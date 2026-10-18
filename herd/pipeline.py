"""Stage orchestration shared by the CLI and the MCP server. Each stage writes its output
under results/<task>/; LLM and Exa calls are cached, so rebuilding is free."""
import hashlib
import json
import os
import threading
from collections import defaultdict
from pathlib import Path

import yaml

from . import label as label_stage
from . import literature as literature_stage
from .cache import JsonlCache
from .exa import Exa
from .herdmap import build_herd_map, find_blind_spots
from .label import EMERGENT, UNLABELED, classify, label_samples, write_audit
from .literature import build_taxonomy, search_literature
from .llm import LLM, BudgetExceeded, Ledger, load_model_catalog
from .reality import reality_check
from .sample import run_sampling

ROOT = Path(__file__).resolve().parent.parent
_caches, _task_locks = {}, defaultdict(threading.Lock)


def load_env():
    """Read KEY=VALUE lines from .env without overriding the real environment."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def shared_cache(path):
    """One in-memory cache per file per process, so concurrent tool calls see each other's writes."""
    if path not in _caches:
        _caches[path] = JsonlCache(path)
    return _caches[path]


class Herd:
    """Wiring for one CLI run or one tool call: config, budget ledger, cached clients.
    budget_usd caps this call's spend; the config's budget_usd caps total spend."""

    def __init__(self, budget_usd=None, progress=print, bill_cached=False):
        load_env()
        self.cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        cache_dir = ROOT / self.cfg["paths"]["cache"]
        llm_cache, exa_cache = shared_cache(cache_dir / "llm.jsonl"), shared_cache(cache_dir / "exa.jsonl")
        spent = llm_cache.total_cost() + exa_cache.total_cost()
        self.ledger = Ledger(self.cfg["budget_usd"], spent, budget_usd, bill_cached)
        self.llm = LLM(llm_cache, load_model_catalog(cache_dir), self.ledger)
        self.exa = Exa(exa_cache, self.ledger, self.cfg["exa"]["cost_per_search_usd"])
        self.progress = progress

    def out_dir(self, task):
        path = ROOT / self.cfg["paths"]["results"] / task.name
        path.mkdir(parents=True, exist_ok=True)
        (path / "task.md").write_text(task.prompt)
        return path


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2))


def inputs_hash(task, cfg):
    """Everything besides models and N that shapes the map: a saved map built from other
    prompts or settings is rebuilt instead of reused."""
    parts = [task.prompt, literature_stage.QUERY_PROMPT, literature_stage.TAXONOMY_PROMPT,
             label_stage.LABEL_PROMPT, label_stage.MERGE_PROMPT,
             json.dumps([cfg["literature"], cfg["labeling"], cfg["blind_spot_max_share"]], sort_keys=True)]
    return hashlib.sha256("\n\0".join(parts).encode()).hexdigest()[:16]


def ensure_map(herd, task, n_per_model=None):
    """SAMPLE -> LITERATURE -> LABEL -> HERD MAP -> BLIND SPOTS, or the saved result if it
    was built with the same models and N. Returns {herd_map, blind_spots, families}."""
    cfg, out = herd.cfg, herd.out_dir(task)
    n = n_per_model or cfg["sampling"]["n_per_model"]
    meta = {"samplers": cfg["models"]["samplers"], "labeler": cfg["models"]["labeler"], "n_per_model": n,
            "inputs_hash": inputs_hash(task, cfg)}
    with _task_locks[task.name]:
        saved = read_json(out / "herd_map.json")
        if saved and saved["meta"] == meta and saved["n_api_errors"] == 0:
            return {"herd_map": saved, "blind_spots": read_json(out / "blind_spots.json"),
                    "families": read_json(out / "families.json")}

        herd.progress(f"sampling {n} per model x {len(meta['samplers'])} models")
        records = run_sampling(task, herd.llm, cfg, out, n, herd.progress)
        if over := [r for r in records if r["status"] == "budget_exceeded"]:
            raise BudgetExceeded(over[0]["error"])
        errors = [r for r in records if r["status"] == "api_error"]
        if len(errors) == len(records):
            raise RuntimeError(f"every sample failed; first error: {errors[0]['error']}")

        herd.progress("literature search")
        literature = search_literature(task, herd.llm, herd.exa, cfg, herd.progress)
        write_json(out / "literature.json", literature)
        taxonomy = build_taxonomy(task, literature, herd.llm, cfg)
        write_json(out / "taxonomy.json", taxonomy)

        herd.progress(f"labeling against {len(taxonomy['families'])} families")
        labels, emergent = label_samples(task, records, taxonomy, herd.llm, cfg, out, herd.progress)
        with (out / "labels.jsonl").open("w") as f:
            for sid, l in labels.items():
                f.write(json.dumps({"sample_id": sid, **l}) + "\n")
        write_audit(out / "labels_audit.csv", records, labels, out)

        families = taxonomy["families"] + emergent
        herd_map = {"task": task.name, "meta": meta, "n_api_errors": len(errors),
                    **build_herd_map(records, labels, families)}
        blind_spots = find_blind_spots(herd_map, families, cfg["blind_spot_max_share"])
        write_json(out / "families.json", families)
        write_json(out / "herd_map.json", herd_map)
        write_json(out / "blind_spots.json", blind_spots)
        return {"herd_map": herd_map, "blind_spots": blind_spots, "families": families}


def family_kind(study, name):
    if name in study["herd_map"]["top3"]:
        return "herd"
    return "blind_spot" if name in {b["family"] for b in study["blind_spots"]} else "other"


def ensure_family_scores(herd, task, study, families, evaluator, k):
    """Reality-check each family (a dict from families.json) unless already done with >= k
    implementations under this evaluator. Returns all saved scores for this evaluator."""
    out = herd.out_dir(task)
    share = {r["family"]: r["share"] for r in study["herd_map"]["families"]}
    with _task_locks[task.name]:
        saved = read_json(out / "family_scores.json", {})
        scores = saved.setdefault(evaluator.key, {})
        for family in families:
            name = family["family"]
            if name in scores and scores[name]["k"] >= k:
                continue
            scores[name] = reality_check(task, family, evaluator, herd.llm, herd.cfg, out, k, herd.progress)
            refresh_kinds(scores, study, share)
            write_json(out / "family_scores.json", saved)  # save after each family
        refresh_kinds(scores, study, share)  # the map may have been rebuilt since scoring
        write_json(out / "family_scores.json", saved)
        return scores


def refresh_kinds(scores, study, share):
    for name, s in scores.items():
        s["kind"], s["share"] = family_kind(study, name), share.get(name, 0.0)


def baseline_scores(task, evaluator, baseline_files=()):
    if not evaluator.command:
        return {name: r["score"] for name, r in task.baselines().items()}
    return {Path(p).stem: evaluator(Path(p))["score"] for p in baseline_files}


def check_approach(herd, task, study, approach, evaluator, k):
    """Place an approach in the herd map; with an evaluator, test its family against the
    blind spots and the top herd families and say which is best."""
    families = study["families"]
    picked = classify(task, approach, families, herd.llm, herd.cfg)
    name = picked["family"]
    row = next((r for r in study["herd_map"]["families"] if r["family"] == name), None)
    blind = {b["family"]: b for b in study["blind_spots"]}
    result = {"family": name, "why": picked.get("justification", ""),
              "share": row["share"] if row else 0.0,
              "rank": study["herd_map"]["families"].index(row) + 1 if row else None,
              "is_blind_spot": name in blind,
              "concentration_top3": study["herd_map"]["concentration_top3"]}
    if name == UNLABELED:
        return {**result, "verdict": "unclassified"}
    if evaluator is None:
        return {**result, "verdict": "untested", "note": "pass evaluator_cmd to test",
                "blind_spots": [b["family"] for b in study["blind_spots"]]}

    by_name = {f["family"]: f for f in families}
    mine = by_name.get(name) or {"family": name, "one_line_mechanism": picked.get("justification", ""),
                                 "source_summary": "No literature source; family named for this approach.",
                                 "source_urls": []}
    to_test = [mine] + [by_name[f] for f in study["herd_map"]["top3"] if f != name] \
        + [b for b in study["blind_spots"] if b["family"] != name]
    scores = ensure_family_scores(herd, task, study, to_test, evaluator, k)
    tested = {f["family"]: scores[f["family"]]["best_score"] for f in to_test
              if scores[f["family"]]["best_score"] is not None}
    result["my_family_best_score"] = tested.get(name)
    result["tested"] = {f: round(s, 6) for f, s in tested.items()}
    if name not in tested:
        return {**result, "verdict": "inconclusive",
                "note": f"no valid implementation of {name} in {k} tries"}

    winners = sorted((f for f in blind if f in tested and evaluator.beats(tested[f], tested[name])),
                     key=lambda f: tested[f], reverse=not evaluator.lower_is_better)
    if winners:
        b, s = blind[winners[0]], scores[winners[0]]
        return {**result, "verdict": "try_blind_spot",
                "blind_spot": {"family": b["family"], "mechanism": b["one_line_mechanism"],
                               "share": b["share"], "best_score": s["best_score"],
                               "code_path": str(herd.out_dir(task) / s["best_code_path"]),
                               "source_urls": b["source_urls"]}}
    best_family = min(tested, key=tested.get) if evaluator.lower_is_better else max(tested, key=tested.get)
    if tested[best_family] == tested[name]:
        common = result["share"] >= herd.cfg["blind_spot_max_share"]
        return {**result, "verdict": "keep",
                "message": "common and best, keep it" if common else "rare and best, keep it"}
    return {**result, "verdict": "try_other_family",
            "better_family": {"family": best_family, "best_score": tested[best_family],
                              "share": scores[best_family]["share"]}}
