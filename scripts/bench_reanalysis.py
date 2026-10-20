"""Re-analysis of a finished operator benchmark, offline ($0), with the corrections the review panel asked for
(docs/REVIEW-PANEL-2026-10-20.md):

    uv run python scripts/bench_reanalysis.py cache_eviction [--dir results/cache_eviction/bench] [--json]

- Equal K: per seed, every operator is cut to its first K valid candidates in generation order, K being the
  smallest valid count any operator reached on that seed. best-of-12 against best-of-6 measures throughput,
  not the operator. Best and median dev score at equal K are reported per operator (mean over seeds).
- Strongest baseline: "beats baseline" counts valid candidates whose dev score beats the strongest dev
  baseline, not LRU.
- Cold cost: spent_cold_usd (including the consensus map an operator needs) next to spent_usd (cached).
Reads candidates.jsonl and bench.json written by diverge.bench.run_bench."""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(bench_dir):
    rows = [json.loads(line) for line in (bench_dir / "candidates.jsonl").read_text().splitlines() if line.strip()]
    meta = json.loads((bench_dir / "bench.json").read_text())
    return rows, meta


def analyze(rows, meta, dev_baselines, lower_is_better=True):
    pick = min if lower_is_better else max
    better = (lambda a, b: a < b) if lower_is_better else (lambda a, b: a > b)
    strongest_name = pick(dev_baselines, key=dev_baselines.get)
    strongest = dev_baselines[strongest_name]
    by = {}
    for r in rows:
        if r.get("valid") and r.get("dev_score") is not None:
            by.setdefault((r["seed"], r["operator"]), []).append(r["dev_score"])
    operators = sorted({r["operator"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    out = {"strongest_dev_baseline": {"name": strongest_name, "score": strongest}, "seeds": seeds, "operators": {}}
    k_by_seed = {s: min((len(by.get((s, op), [])) for op in operators), default=0) for s in seeds}
    out["k_by_seed"] = k_by_seed
    spend = {}
    for r in meta.get("per_run", []):
        spend.setdefault(r["operator"], []).append((r.get("spent_usd") or 0.0, r.get("spent_cold_usd") or 0.0))
    for op in operators:
        best_k, median_k, beats, n_valid = [], [], 0, 0
        for s in seeds:
            scores = by.get((s, op), [])
            n_valid += len(scores)
            beats += sum(better(x, strongest) for x in scores)
            k = k_by_seed[s]
            if k:
                best_k.append(pick(scores[:k]))
                median_k.append(statistics.median(scores[:k]))
        cached = sum(a for a, _ in spend.get(op, []))
        cold = sum(b for _, b in spend.get(op, []))
        out["operators"][op] = {
            "n_valid": n_valid, "best_dev_equal_k": statistics.mean(best_k) if best_k else None,
            "median_dev_equal_k": statistics.mean(median_k) if median_k else None,
            "beats_strongest_baseline": beats, "spent_usd": round(cached, 4), "spent_cold_usd": round(cold, 4),
            "beats_per_cold_usd": round(beats / cold, 3) if cold else None}
    return out


def markdown(result):
    b = result["strongest_dev_baseline"]
    lines = [f"Equal K per seed: {result['k_by_seed']}. Strongest dev baseline: {b['name']} {b['score']:.4f}.", "",
             "| operator | valid | best dev (equal K) | median dev (equal K) | beat strongest baseline | $ cached | $ cold | beats per cold $ |",
             "|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.4f}"
    for op, r in sorted(result["operators"].items(), key=lambda kv: (kv[1]["best_dev_equal_k"] is None,
                                                                     kv[1]["best_dev_equal_k"] or 0)):
        lines.append(f"| {op} | {r['n_valid']} | {f(r['best_dev_equal_k'])} | {f(r['median_dev_equal_k'])} | "
                     f"{r['beats_strongest_baseline']} | {r['spent_usd']:.3f} | {r['spent_cold_usd']:.3f} | "
                     f"{'-' if r['beats_per_cold_usd'] is None else r['beats_per_cold_usd']} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from diverge.task import Task
    task = Task.resolve(args.task)
    bench_dir = args.dir or ROOT / "results" / task.name / "bench"
    rows, meta = load(bench_dir)
    dev = {k: r["score"] for k, r in task.baselines("dev").items()}
    result = analyze(rows, meta, dev)
    print(json.dumps(result, indent=2) if args.json else markdown(result))


if __name__ == "__main__":
    main()
