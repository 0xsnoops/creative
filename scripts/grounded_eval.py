"""Grounded alternatives, one preregistered measurement (docs/PREREGISTRATION-grounded.md). Run once:

    DIVERGE_TRACE_DIR=results/traces/grounded uv run python scripts/grounded_eval.py

For each of 5 fixed stuck scenarios (SCENARIOS: an already_tried list, constraints and a context
pack), three arms call diverge_check's free-text path:
  - before: the plain call (my_approach only; the inputs are not passed);
  - quick: the same call with already_tried, constraints and context;
  - deep: the same inputs with depth="deep".
A judge model that generated and screened nothing (JUDGE) labels every returned alternative against the
scenario's already_tried list and constraints. Metrics per arm, pooled over scenarios:
  - dup_rate: alternatives judged the same mechanism as an already-tried approach / alternatives;
  - violation_rate: alternatives judged to break a constraint / alternatives;
  - grounded_share: stated numbers (not single-digit integers) that appear in the context pack /
    stated numbers, over the name, mechanism, strength, trap, how_to_test, grounding and kill_result
    fields (diverge.alternatives.numbers);
  - n_alternatives and spent_usd.
Writes results/grounded_eval/{arms,summary}.json and prints the summary. Descriptive only: 5
scenarios support no significance claim.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from diverge import alternatives as A  # noqa: E402
from diverge.llm import ask_json  # noqa: E402
from diverge.pipeline import Session  # noqa: E402
from diverge.server import free_text_check  # noqa: E402
from diverge.task import Task  # noqa: E402

JUDGE = "deepseek/deepseek-v4-pro-0813"  # not the quick generator, the deep samplers or the screen/critic model
# Per-call caps. Reservations are worst-case, so caps sit well above the expected spend (under $2 in all).
BUDGETS = {"before": 0.1, "quick": 0.3, "deep": 1.0}
ARMS = ("before", "quick", "deep")
FIELDS = ("name", "mechanism", "strength", "trap", "how_to_test", "grounding", "kill_result")

SCENARIOS = [
    {"task": "online_bin_packing", "approach": "best fit: put each item in the open bin it fills most tightly",
     "already_tried": [
         {"approach": "best fit with a tie-break on bin index", "outcome": "dev waste unchanged at 4.02%", "score": 0.0402},
         {"approach": "best fit with a 0.98 tightness threshold before opening a new bin", "outcome": "0.01% better, noise level", "score": 0.0401},
         {"approach": "worst fit to keep bins balanced", "outcome": "worse: 6.8% waste", "score": 0.068}],
     "constraints": ["pure Python standard library only", "one decision per item, no look-ahead at future items",
                     "under 1 ms per decision"],
     "context": "dev waste: best fit 4.02%, first fit 4.31%, worst fit 6.80%\nitem sizes: Weibull, mean 39, p95 58, max 100\n"
                "bins open at the end of a dev instance: median 412"},
    {"task": "cache_eviction", "approach": "LRU with an OrderedDict, evict the least recently used key",
     "already_tried": [
         {"approach": "LRU with a 10% probationary segment (SLRU)", "outcome": "miss ratio 0.3121 vs 0.3164 for LRU", "score": 0.3121},
         {"approach": "LFU with aging every 10000 requests", "outcome": "worse on scan-heavy traces: 0.3410", "score": 0.341}],
     "constraints": ["no per-key memory beyond 16 bytes", "O(1) amortized per request"],
     "context": "dev miss ratio: LRU 0.3164, SLRU 0.3121, LFU-aging 0.3410\ncache size 1000 keys; 4 trace types: "
                "uniform, skewed (zipf 0.9), scan, loop of 1200 keys\nworst trace for every attempt: loop of 1200 keys"},
    {"task": "A relay forwards market data from one exchange feed to 40 subscribers over TCP. Reduce the p99 "
             "delivery delay measured at the subscribers.",
     "approach": "batch messages per subscriber and flush every 5 ms",
     "already_tried": [
         {"approach": "flush every 2 ms instead of 5 ms", "outcome": "p99 61 ms to 58 ms, CPU up 30%", "score": 58},
         {"approach": "one writer thread per subscriber", "outcome": "p99 unchanged, 40 threads contend on the GIL", "score": 60},
         {"approach": "TCP_NODELAY on every socket", "outcome": "p50 better, p99 unchanged", "score": 61}],
     "constraints": ["Python 3.12 only, no C extensions", "one 4-core VM", "subscribers cannot change their client"],
     "context": "p50 22 ms, p99 61 ms at 3,000 msgs/s\n3 of 40 subscribers are on a 140 ms RTT link\n"
                "send buffer full events: 1,912 per hour, all on the 3 slow subscribers"},
    {"task": "Rank 50 candidate search results for a query so that the clicked result is as high as possible, "
             "using a fixed log of 20,000 past queries with clicks for offline replay.",
     "approach": "a weighted sum of BM25, click-through rate and freshness, with weights tuned on the log",
     "already_tried": [
         {"approach": "grid search over the three weights", "outcome": "MRR 0.412 to 0.419", "score": 0.419},
         {"approach": "add document length as a fourth weighted feature", "outcome": "MRR 0.418", "score": 0.418}],
     "constraints": ["no neural models and no GPU", "must explain each ranking with at most 3 features",
                     "re-ranking 50 results under 5 ms"],
     "context": "offline replay MRR: BM25 only 0.371, tuned weighted sum 0.419\nposition bias: 61% of clicks on "
                "rank 1-3\n28% of queries have no click in the top 50"},
    {"task": "Schedule 300 nightly batch jobs on 12 machines to minimize the time the last job finishes. Job "
             "durations vary night to night; history of 90 nights is available.",
     "approach": "longest processing time first, using each job's mean historical duration",
     "already_tried": [
         {"approach": "LPT using the 90th percentile duration instead of the mean", "outcome": "makespan 6.1 h vs 6.2 h, within noise", "score": 6.1},
         {"approach": "LPT with jobs re-sorted every hour", "outcome": "no change: 6.2 h", "score": 6.2}],
     "constraints": ["jobs cannot be preempted", "a dependency list of 41 job pairs must be respected",
                     "the scheduler has 10 seconds to plan"],
     "context": "makespan: mean 6.2 h, best night 5.4 h, worst 8.9 h\n7 jobs account for 38% of total runtime\n"
                "on the 9 worst nights one of 3 jobs ran more than 2x its mean"},
]

JUDGE_PROMPT = """You are judging proposed strategies against an agent's history. You did not write them.

Task:
{task}

Already tried:
{tried}

Hard rules:
{rules}

Proposals:
{proposals}

For each proposal give "duplicate": true if it is the same mechanism as an already-tried approach (same
decision signal and rule, possibly re-weighted, re-tuned or renamed), and "violates": true if it breaks a
hard rule. Reply with only a JSON object:
{{"labels": [{{"index": 0, "duplicate": false, "violates": false}}]}}"""


def run_arm(scenario, arm, budget):
    session = Session(budget, lambda m: None)
    task = Task.resolve(scenario["task"])
    kwargs = {} if arm == "before" else {"already_tried": scenario["already_tried"],
                                         "constraints": scenario["constraints"], "context": scenario["context"],
                                         "depth": "deep" if arm == "deep" else "quick"}
    out = free_text_check(session, task, scenario["approach"], **kwargs)
    alts = out["alternatives"]
    judged = []
    if alts:
        reply = ask_json(session.llm, JUDGE, JUDGE_PROMPT.format(
            task=task.prompt, tried="\n".join(f"[{i}] {A.tried_line(t)}" for i, t in
                                              enumerate(A.clean_tried(scenario["already_tried"]))),
            rules="\n".join(f"- {c}" for c in scenario["constraints"]),
            proposals="\n".join(f"[{i}] {a['name']}: {a['mechanism']}" for i, a in enumerate(alts))),
            max_tokens=4000, key=f"grounded_eval:{arm}", required=("labels",))
        labels = {r.get("index"): r for r in reply["labels"] if isinstance(r, dict)}
        judged = [labels.get(i, {}) for i in range(len(alts))]
    have = A.numbers(scenario["context"])
    stated = [x for a in alts for x in set().union(*(A.numbers(a.get(k)) for k in FIELDS))
              if not (x.is_integer() and 0 <= x <= 9)]
    return {"task": task.name, "arm": arm, "n": len(alts),
            "duplicates": sum(bool(j.get("duplicate")) for j in judged),
            "violations": sum(bool(j.get("violates")) for j in judged),
            "numbers_stated": len(stated), "numbers_in_context": sum(x in have for x in stated),
            "filtered_out": out.get("filtered_out"), "alternatives": alts, "judge": judged,
            "spent_usd": round(session.ledger.spent_this_run, 4)}


def summarize(rows):
    out = {}
    for arm in ARMS:
        r = [x for x in rows if x["arm"] == arm]
        n, stated = sum(x["n"] for x in r), sum(x["numbers_stated"] for x in r)
        out[arm] = {"scenarios": len(r), "n_alternatives": n,
                    "dup_rate": round(sum(x["duplicates"] for x in r) / n, 3) if n else None,
                    "violation_rate": round(sum(x["violations"] for x in r) / n, 3) if n else None,
                    "grounded_share": round(sum(x["numbers_in_context"] for x in r) / stated, 3) if stated else None,
                    "numbers_stated": stated, "spent_usd": round(sum(x["spent_usd"] for x in r), 4)}
    return out


def main():
    out = ROOT / "results" / "grounded_eval"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        for arm in ARMS:
            try:
                rows.append(run_arm(scenario, arm, BUDGETS[arm]))
            except Exception as e:  # recorded, not retried: the protocol runs once
                rows.append({"task": scenario["task"][:60], "arm": arm, "error": f"{type(e).__name__}: {e}"[:300],
                             "n": 0, "duplicates": 0, "violations": 0, "numbers_stated": 0,
                             "numbers_in_context": 0, "spent_usd": 0.0})
            (out / "arms.json").write_text(json.dumps(rows, indent=2))
    summary = {"summary": summarize(rows), "errors": [r for r in rows if r.get("error")]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
