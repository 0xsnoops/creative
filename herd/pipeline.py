"""Stage orchestration shared by the CLI and the MCP server. Each stage writes its output
under results/<task>/; LLM and Exa calls are cached, so rebuilding is free."""
import hashlib
import json
import os
import threading
import time
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
from .reality import sandbox as _sandbox

sandbox_redact = _sandbox.redact
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


_tracers = {}


def shared_tracer(trace_dir):
    if trace_dir not in _tracers:
        _tracers[trace_dir] = Tracer(trace_dir)
    return _tracers[trace_dir]


class Tracer:
    """Append-only JSONL of model and search calls (the experiment-phase trace). Records are
    redacted before writing; API keys are never part of them."""

    def __init__(self, trace_dir):
        self.path = Path(trace_dir) / "calls.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def log(self, record):
        text = sandbox_redact(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "phase": "experiment", **record}))
        with self.lock, self.path.open("a") as f:
            f.write(text + "\n")


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
        # HERD_TRACE_DIR set: every model and search call is also appended to <dir>/calls.jsonl.
        self.trace_dir = Path(os.environ["HERD_TRACE_DIR"]) if os.environ.get("HERD_TRACE_DIR") else None
        tracer = shared_tracer(self.trace_dir) if self.trace_dir else None
        self.llm = LLM(llm_cache, load_model_catalog(cache_dir), self.ledger, tracer)
        self.exa = Exa(exa_cache, self.ledger, self.cfg["exa"]["cost_per_search_usd"], tracer)
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


def effective_n(study):
    """Labeled samples per model for the weakest sampler: what confidence should rest on,
    rather than the N that was requested."""
    counts = study["herd_map"]["meta"].get("labeled_per_model")
    return min(counts.values()) if counts else 0


def ensure_map(herd, task, n_per_model=None, retry_failed=False):
    """SAMPLE -> LITERATURE -> LABEL -> HERD MAP -> BLIND SPOTS, or the saved result if it
    was built from the same inputs. A saved map with failed samples is reused (the gaps are
    recorded in its meta) unless retry_failed is set. Returns {herd_map, blind_spots, families}."""
    cfg, out = herd.cfg, herd.out_dir(task)
    n = n_per_model or cfg["sampling"]["n_per_model"]
    inputs = {"samplers": cfg["models"]["samplers"], "labeler": cfg["models"]["labeler"], "n_per_model": n,
              "inputs_hash": inputs_hash(task, cfg)}
    with _task_locks[task.name]:
        saved = read_json(out / "herd_map.json")
        if saved and {k: saved["meta"].get(k) for k in inputs} == inputs and \
                (saved["n_api_errors"] == 0 or not retry_failed):
            return {"herd_map": saved, "blind_spots": read_json(out / "blind_spots.json"),
                    "families": read_json(out / "families.json")}

        # What each stage would cost from a cold cache, for cold-start benchmark accounting.
        build_cost, mark = {}, herd.ledger.recorded

        def stage_done(stage):
            nonlocal mark
            build_cost[stage] = round(herd.ledger.recorded - mark, 6)
            mark = herd.ledger.recorded

        herd.progress(f"sampling {n} per model x {len(inputs['samplers'])} models")
        records = run_sampling(task, herd.llm, cfg, out, n, herd.progress)
        stage_done("sampling")
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
        stage_done("literature")

        herd.progress(f"labeling against {len(taxonomy['families'])} families")
        labels, emergent = label_samples(task, records, taxonomy, herd.llm, cfg, out, herd.progress)
        with (out / "labels.jsonl").open("w") as f:
            for sid, l in labels.items():
                f.write(json.dumps({"sample_id": sid, **l}) + "\n")
        write_audit(out / "labels_audit.csv", records, labels, out)
        stage_done("labeling")

        families = taxonomy["families"] + emergent
        built = build_herd_map(records, labels, families)
        completed = {m: sum(r["model"] == m and bool(r.get("text_path")) for r in records) for m in inputs["samplers"]}
        labeled = {m: built["per_model_counts"].get(m, {}).get("n", 0) for m in inputs["samplers"]}
        meta = {**inputs, "completed_per_model": completed, "labeled_per_model": labeled,
                "min_completed_per_model": min(completed.values()),
                "failed_samples": sorted(r["sample_id"] for r in errors)}
        herd_map = {"task": task.name, "meta": meta, "n_api_errors": len(errors), "build_cost_usd": build_cost,
                    **built}
        blind_spots = find_blind_spots(herd_map, families, cfg["blind_spot_max_share"])
        write_json(out / "families.json", families)
        write_json(out / "herd_map.json", herd_map)
        write_json(out / "blind_spots.json", blind_spots)
        return {"herd_map": herd_map, "blind_spots": blind_spots, "families": families}


def family_kind(study, name):
    if name in study["herd_map"]["top3"]:
        return "herd"
    return "blind_spot" if name in {b["family"] for b in study["blind_spots"]} else "other"


def family_hash(family):
    text = json.dumps([family["family"], family.get("one_line_mechanism"), family.get("source_summary")])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_scores(herd, task, evaluator):
    return read_json(herd.out_dir(task) / "family_scores.json", {}).get(evaluator.key, {})


def record_heldout_use(herd, task, evaluator):
    """Count how often held-out comparisons were shown for this task and evaluator. Returns the
    number of earlier uses; any earlier use makes a new comparison exploratory, because the
    caller may already have adapted to held-out results."""
    path = herd.out_dir(task) / "family_scores.json"
    with _task_locks[task.name]:
        saved = read_json(path, {})
        uses = saved.setdefault("_heldout_uses", {})
        prior = uses.get(evaluator.key, 0)
        uses[evaluator.key] = prior + 1
        write_json(path, saved)
    return prior


def ensure_family_scores(herd, task, study, families, evaluator, k):
    """Reality-check each family (a dict from families.json) unless already done with >= k
    implementations under this evaluator and the same family text. Returns all saved scores
    for this evaluator."""
    out = herd.out_dir(task)
    share = {r["family"]: r["share"] for r in study["herd_map"]["families"]}
    with _task_locks[task.name]:
        saved = read_json(out / "family_scores.json", {})
        scores = saved.setdefault(evaluator.key, {})
        for family in families:
            name, fhash = family["family"], family_hash(family)
            prev = scores.get(name)
            if prev and prev["k"] >= k and "final_score" in prev and prev.get("family_hash") == fhash:
                continue
            scores[name] = {**reality_check(task, family, evaluator, herd.llm, herd.cfg, out, k, herd.progress),
                            "family_hash": fhash}
            refresh_kinds(scores, study, share)
            write_json(out / "family_scores.json", saved)  # save after each family
        refresh_kinds(scores, study, share)  # the map may have been rebuilt since scoring
        write_json(out / "family_scores.json", saved)
        return scores


def refresh_kinds(scores, study, share):
    for name, s in scores.items():
        s["kind"], s["share"] = family_kind(study, name), share.get(name, 0.0)


def baseline_scores(task, evaluator, baseline_files=()):
    """Reference scores on the same split that final scores use."""
    if not evaluator.command:
        return {name: r["score"] for name, r in task.baselines("heldout").items()}
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
    # Cheapest decisive order: your family, then blind spots (ranked), then the top herd families.
    # Each family is scored on its own so a budget stop keeps everything tested so far.
    to_test = [mine] + [b for b in study["blind_spots"] if b["family"] != name] \
        + [by_name[f] for f in study["herd_map"]["top3"] if f != name and f not in blind]
    stopped_at = None
    for family in to_test:
        try:
            ensure_family_scores(herd, task, study, [family], evaluator, k)
        except BudgetExceeded:
            stopped_at = family["family"]
            break
    scores = load_scores(herd, task, evaluator)
    # Each family is represented by its implementation selected on dev, and families are compared
    # on that implementation's held-out score (dev score only for external evaluators).
    tested = {f["family"]: scores[f["family"]]["final_score"] for f in to_test
              if f["family"] in scores and scores[f["family"]]["final_score"] is not None}
    result["complete"], result["stopped_at"] = stopped_at is None, stopped_at
    result["scored_on"] = "heldout" if evaluator.has_heldout else "dev"
    if evaluator.has_heldout and tested:
        prior = record_heldout_use(herd, task, evaluator)
        result["heldout_prior_uses"], result["exploratory"] = prior, prior > 0
    result["my_family_score"] = tested.get(name)
    result["tested"] = {f: round(s, 6) for f, s in tested.items()}
    if name not in tested:
        note = (f"budget ran out before {name} was tested" if stopped_at == name
                else f"no implementation of {name} was valid in {k} tries")
        return {**result, "verdict": "inconclusive", "note": note}

    def describe(f):
        entry, s = by_name.get(f) or blind.get(f) or {}, scores[f]
        return {"family": f, "kind": s["kind"], "share": s["share"], "score": tested[f],
                "mechanism": entry.get("one_line_mechanism", ""), "source_urls": entry.get("source_urls", []),
                "code_path": str(herd.out_dir(task) / s["best_code_path"])}

    def best_of(names):
        return min(names, key=tested.get) if evaluator.lower_is_better else max(names, key=tested.get)

    # Decide on quality first; rarity only explains the winner, it never overrides it.
    best = best_of(tested)
    if not evaluator.beats(tested[best], tested[name]):
        common = result["share"] >= herd.cfg["blind_spot_max_share"]
        return {**result, "verdict": "keep",
                "message": "common and best, keep it" if common else "rare and best, keep it"}
    if best in blind:
        return {**result, "verdict": "try_blind_spot", "blind_spot": describe(best)}
    result = {**result, "verdict": "try_other_family", "better_family": describe(best)}
    rare_winners = [f for f in blind if f in tested and evaluator.beats(tested[f], tested[name])]
    if rare_winners:  # also beats yours, but less than the best family: a different direction to explore
        result["diverse_alternative"] = describe(best_of(rare_winners))
    return result
