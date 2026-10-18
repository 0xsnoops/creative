"""E1, detector validity (offline, $0): for pairs of bin-packing programs, which signal best
predicts "same failure set" -- behavior agreement on the probe states, code similarity, or the
LLM-description strategy similarity? See docs/RESEARCH-loop-escape.md (E1) and docs/RESULTS-E1.md.

    ~/.local/bin/uv run python scripts/e1_detector_validity.py [--reps 300]

No paid calls: the LLM extractor runs under Herd(budget_usd=0.0) with the HTTP call disabled,
so only cached replies are used, and the script asserts herd.ledger.new_spend == 0.
Writes results/e1/results.json."""
import argparse
import hashlib
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from herd import fingerprint as behavioral  # noqa: E402
from herd import strategy  # noqa: E402
from herd.failures import auc, failure_set, jaccard, relative_detail  # noqa: E402
from herd.llm import BudgetExceeded  # noqa: E402
from herd.pipeline import Herd  # noqa: E402
from herd.task import Task  # noqa: E402

TASK_DIR = ROOT / "results" / "online_bin_packing"
OUT = ROOT / "results" / "e1"
SAME = 0.8  # jaccard threshold for "same failure set"
K = 2


def collect():
    """[(source, code)] from every listed origin, before dedupe."""
    found = []
    for line in (TASK_DIR / "samples.jsonl").read_text().splitlines():
        s = json.loads(line)
        if s.get("status") == "valid" and s.get("code_path"):
            found.append((f"samples/{s['sample_id']}", (TASK_DIR / s["code_path"]).read_text()))
    for f in sorted((ROOT / "fixtures" / "trajectories").glob("*.json")):
        for a in json.loads(f.read_text()).get("attempts", []):
            if a.get("code"):
                found.append((f"fixture/{f.stem}/{a.get('id')}", a["code"]))
    for sub in ("demo", "escape", "paraphrase"):
        for p in sorted((TASK_DIR / sub).rglob("*.py")):
            found.append((f"{sub}/{p.relative_to(TASK_DIR / sub)}", p.read_text()))
    return found


def np_auc(scores, labels):
    """Vectorised Mann-Whitney AUC with tie-averaged ranks (same as herd.failures.auc)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s))
    _, start, counts = np.unique(s, return_index=True, return_counts=True)
    avg = start + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pair_matrix(n, fn):
    m = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = fn(i, j)
    return m


def boot(mats, target, members, reps, rng, diff=None):
    """Program-level bootstrap: resample `members` (program indices) with replacement, take all
    pairs of distinct programs among the draws, recompute each AUC. Returns {name: [lo, hi]}
    and, with diff=(a, b), the CI of AUC(a) - AUC(b)."""
    members = np.asarray(members)
    draws = {name: [] for name in mats}
    diffs = []
    for _ in range(reps):
        r = rng.choice(members, size=len(members), replace=True)
        a, b = np.triu_indices(len(r), 1)
        ia, ib = r[a], r[b]
        keep = ia != ib
        ia, ib = ia[keep], ib[keep]
        y = target[ia, ib]
        got = {name: np_auc(m[ia, ib], y) for name, m in mats.items()}
        for name, v in got.items():
            if v is not None:
                draws[name].append(v)
        if diff and got[diff[0]] is not None and got[diff[1]] is not None:
            diffs.append(got[diff[0]] - got[diff[1]])
    ci = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if xs else None
    out = {name: ci(v) for name, v in draws.items()}
    if diff:
        out["diff"] = ci(diffs)
        out["diff_reps"] = len(diffs)
    return out


def analyse(mats, target, members, reps, seed, diff=None):
    members = list(members)
    idx = [(i, j) for x, i in enumerate(members) for j in members[x + 1:]]
    ia, ib = np.array([p[0] for p in idx], int), np.array([p[1] for p in idx], int)
    y = target[ia, ib] if idx else np.array([], bool)
    point = {name: (auc(m[ia, ib].tolist(), y.tolist()) if idx else None) for name, m in mats.items()}
    cis = boot(mats, target, members, reps, np.random.default_rng(seed), diff) if idx else {}
    res = {"n_programs": len(members), "n_pairs": len(idx), "n_positive": int(y.sum()),
           "base_rate": float(y.mean()) if idx else None,
           "auc": {name: {"auc": point[name], "ci95": cis.get(name)} for name in mats}}
    if diff:
        a, b = point[diff[0]], point[diff[1]]
        res["diff"] = {"name": f"{diff[0]} - {diff[1]}", "value": (a - b) if a is not None and b is not None else None,
                       "ci95": cis.get("diff"), "reps": cis.get("diff_reps")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--paid", type=float, default=0.0,
                    help="USD cap for uncached extractor calls (default 0: cache-only, as in the original E1)")
    ap.add_argument("--extractor", default=None, help="extractor model slug (default: config's extractor)")
    args = ap.parse_args()
    started = time.time()
    task = Task.registered("online_bin_packing")

    raw = collect()
    programs, seen = [], {}
    for source, code in raw:
        h = hashlib.sha256(code.encode()).hexdigest()
        if h in seen:
            programs[seen[h]]["sources"].append(source)
        else:
            seen[h] = len(programs)
            programs.append({"id": h[:12], "sha256": h, "sources": [source], "code": code})
    counts = {"collected": len(raw), "unique": len(programs),
              "by_origin": {o: sum(s.startswith(o) for s, _ in raw) for o in ("samples", "fixture", "demo", "escape", "paraphrase")}}
    print(f"{counts['collected']} programs collected, {counts['unique']} unique", flush=True)

    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for p in programs:
            paths[p["id"]] = str(Path(d) / f"{p['id']}.py")
            Path(paths[p["id"]]).write_text(p["code"])
        with ThreadPoolExecutor(8) as pool:
            evals = dict(zip(paths, pool.map(lambda k: task.evaluate(paths[k], split="dev"), paths)))
        print(f"evaluated in {time.time() - started:.0f}s", flush=True)
        valid_ids = [k for k, r in evals.items() if r.get("valid")]
        fps = behavioral.fingerprints(task, {k: paths[k] for k in valid_ids})
        print(f"fingerprinted in {time.time() - started:.0f}s", flush=True)
    best_fit = task.evaluate(str(ROOT / "tasks" / "online_bin_packing" / "baselines" / "best_fit.py"), split="dev")
    assert best_fit["valid"], best_fit

    for p in programs:
        r = evals[p["id"]]
        p["valid_dev"] = bool(r.get("valid"))
        p["dev_score"] = r.get("score")
        p["detail"] = r.get("detail") if r.get("valid") else None
        p["fp"] = fps.get(p["id"])
    kept = [p for p in programs if p["valid_dev"] and p["fp"] is not None]
    counts.update({"invalid_dev": sum(not p["valid_dev"] for p in programs),
                   "invalid_fingerprint": sum(p["valid_dev"] and p["fp"] is None for p in programs),
                   "kept": len(kept)})
    print(json.dumps(counts), flush=True)

    # ---- LLM extractor: cache-only unless --paid gives a cap
    herd = Herd(budget_usd=args.paid, progress=lambda *a, **k: None)
    if args.extractor:
        herd.cfg["models"]["extractor"] = args.extractor
    if not args.paid:
        def no_network(*a, **k):
            raise BudgetExceeded("E1 is cache-only: an uncached model call was attempted")
        herd.llm._post = no_network
    E = strategy.llm_extractor(herd, task)
    blocked = 0

    def extract(code):
        nonlocal blocked
        try:
            return E(code)
        except BudgetExceeded:
            blocked += 1
            return {}
    with ThreadPoolExecutor(8) as pool:  # warm the cache in parallel; the loop below then replays it
        list(pool.map(lambda p: extract(p["code"]), kept))
    for p in kept:
        p["sfp_llm"] = strategy.fingerprint(task, p["code"], extract=extract)
        p["has_llm"] = bool(p["sfp_llm"]["mechanism_family"] or p["sfp_llm"]["decision_rule"]["primary_signal"])
        p["sfp_ast"] = strategy.fingerprint(task, p["code"], extract=None)
    assert args.paid or herd.ledger.new_spend == 0, herd.ledger.new_spend
    llm_ids = [i for i, p in enumerate(kept) if p["has_llm"]]
    print(f"LLM fields for {len(llm_ids)}/{len(kept)} programs (uncached extracts blocked: {blocked}); "
          f"new spend ${herd.ledger.new_spend}", flush=True)

    # ---- pair signals and targets
    n = len(kept)
    fs = [failure_set(p["detail"], K) for p in kept]
    fs_rel = [failure_set(relative_detail(p["detail"], best_fit["detail"]), K) for p in kept]
    mats = {
        "behavior": pair_matrix(n, lambda i, j: 1 - behavioral.distance(kept[i]["fp"], kept[j]["fp"])),
        "code": pair_matrix(n, lambda i, j: strategy.code_similarity(kept[i]["code"], kept[j]["code"])),
        "llm_strategy": pair_matrix(n, lambda i, j: strategy.similarity(kept[i]["sfp_llm"], kept[j]["sfp_llm"])["overall"]),
        "ast_strategy": pair_matrix(n, lambda i, j: strategy.similarity(kept[i]["sfp_ast"], kept[j]["sfp_ast"])["overall"]),
    }
    target = pair_matrix(n, lambda i, j: jaccard(fs[i], fs[j]) >= SAME) == 1
    target_rel = pair_matrix(n, lambda i, j: jaccard(fs_rel[i], fs_rel[j]) >= SAME) == 1
    print(f"pairs built in {time.time() - started:.0f}s", flush=True)

    all_ids = list(range(n))
    no_llm = {k: v for k, v in mats.items() if k != "llm_strategy"}
    results = {
        "primary_all_pairs": analyse(mats, target, all_ids, args.reps, 0, diff=("behavior", "ast_strategy")),
        "primary_llm_pairs": analyse(mats, target, llm_ids, args.reps, 1, diff=("behavior", "llm_strategy")),
        "secondary_vs_best_fit_all_pairs": analyse(no_llm, target_rel, all_ids, args.reps, 2, diff=("behavior", "ast_strategy")),
        "secondary_vs_best_fit_llm_pairs": analyse(mats, target_rel, llm_ids, args.reps, 3, diff=("behavior", "llm_strategy")),
    }
    results["primary_all_pairs"]["diff_code"] = analyse(mats, target, all_ids, args.reps, 0, diff=("behavior", "code"))["diff"]
    # Sanity: the numpy AUC agrees with herd.failures.auc on the full pair set.
    iu = np.triu_indices(n, 1)
    for name, m in mats.items():
        a, b = auc(m[iu].tolist(), target[iu].tolist()), np_auc(m[iu], target[iu])
        assert (a is None and b is None) or abs(a - b) < 1e-9, (name, a, b)

    summary = {
        "settings": {"k": K, "same_threshold": SAME, "reps": args.reps, "probe_states": len(kept[0]["fp"]) if kept else 0,
                     "dev_seeds": [r["seed"] for r in best_fit["detail"]["instances"]],
                     "llm_strategy_behavior_part": "off (strategy.fingerprint behavior=False)"},
        "extractor": herd.cfg["models"].get("extractor"), "paid_cap_usd": args.paid,
        "counts": {**counts, "llm_covered": len(llm_ids), "uncached_extracts_blocked": blocked,
                   "new_spend_usd": herd.ledger.new_spend},
        "failure_set_distribution": {",".join(sorted(s)): fs.count(s) for s in sorted(set(fs), key=sorted)},
        "failure_set_rel_distribution": {",".join(sorted(s)): fs_rel.count(s) for s in sorted(set(fs_rel), key=sorted)},
        "best_fit_dev": best_fit["detail"]["instances"],
        "results": results,
        "programs": [{"id": p["id"], "sources": p["sources"], "dev_score": p["dev_score"],
                      "failure_set": sorted(failure_set(p["detail"], K)),
                      "failure_set_vs_best_fit": sorted(failure_set(relative_detail(p["detail"], best_fit["detail"]), K)),
                      "has_llm": p.get("has_llm"),
                      "mechanism_family": (p.get("sfp_llm") or {}).get("mechanism_family")} for p in programs if p in kept]
                    + [{"id": p["id"], "sources": p["sources"], "dropped": "invalid dev" if not p["valid_dev"] else "invalid fingerprint"}
                       for p in programs if p not in kept],
        "runtime_s": round(time.time() - started, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = "results.json" if not args.extractor else f"results-{args.extractor.replace('/', '_')}.json"
    (OUT / name).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("counts", "results", "runtime_s")}, indent=1, default=str))


if __name__ == "__main__":
    main()
