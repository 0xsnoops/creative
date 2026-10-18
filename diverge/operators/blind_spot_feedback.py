"""blind_spot_feedback: implement the blind-spot families round by round, as blind_spot does,
but each family's next attempt sees what happened to its previous attempt on the dev split:
the (redacted) error if it was invalid, or its dev score against the best dev baseline and its
weakest dev instance if it was valid, plus the previous code. Held-out results are never shown.

blind_spot_feedback_sham is its control: the same note structure and length, but taken from a
different family's previous attempt. blind_spot itself (identical-prompt retries, i.e. i.i.d.
resampling at the same budget) is the main baseline: published work (Olausson et al., arXiv
2306.09896) finds resampling often matches self-repair at small budgets.
(Model-review clusters C1 and C3: anthropic, google, qwen, mistralai, z-ai, minimax, openai,
deepseek; the sham control follows x-ai's critique.)"""
from ..reality import sandbox

NOTE_INVALID = """Previous attempt at this family (development instances only):
It was invalid. Error: {error}
Previous code:
```python
{code}
```
Fix the cause of that failure while keeping the family's mechanism."""

NOTE_VALID = """Previous attempt at this family (development instances only):
It scored {score:.4g} (lower is better); the best reference baseline scores {baseline:.4g}.
Its weakest development case: {weakest}.
Previous code:
```python
{code}
```
Improve on it while keeping the family's mechanism."""


def gaps(detail, baseline_detail):
    """Per-case gaps between a candidate and the best baseline on the same dev case."""
    if not detail or not baseline_detail:
        return []
    if "instances" in detail:  # bin packing: per seed, % excess bins
        base = {r["seed"]: r["excess_pct"] for r in baseline_detail.get("instances", [])}
        return [(r["excess_pct"] - base.get(r["seed"], 0.0), f"instance seed {r['seed']}: {r['excess_pct']:.3g}% "
                 f"excess bins vs {base.get(r['seed'], 0.0):.3g}% for the best baseline") for r in detail["instances"]]
    if "runs" in detail:  # cache eviction: per trace and cache size, miss ratio
        key = lambda r: (r["trace"], r["cache_size"])
        base = {key(r): r["miss_ratio"] for r in baseline_detail.get("runs", [])}
        return [(r["miss_ratio"] - base.get(key(r), 0.0), f"trace {r['trace']} at cache size {r['cache_size']}: "
                 f"miss ratio {r['miss_ratio']:.3f} vs {base.get(key(r), 0.0):.3f} for the best baseline")
                for r in detail["runs"]]
    return []


def note_for(candidate, ctx, baseline):
    """A short dev-only note about one previous attempt; None if there is nothing to say."""
    if not candidate or not candidate.get("code_path"):
        return None
    code = (ctx.herd.out_dir(ctx.task) / candidate["code_path"]).read_text()[:4000]
    if not candidate["valid"]:
        error = sandbox.redact(str(candidate.get("error") or "unknown error"))[:300]
        return NOTE_INVALID.format(error=error, code=code)
    worst = max(gaps(candidate.get("dev_detail"), baseline["detail"]), default=None)
    return NOTE_VALID.format(score=candidate["score"], baseline=baseline["score"],
                             weakest=worst[1] if worst else "not available", code=code)


def best_dev_baseline(ctx):
    results = [r for r in ctx.task.baselines("dev").values() if r["valid"]]
    best = min(results, key=lambda r: r["score"])
    return {"score": best["score"], "detail": best["detail"]}


def _jobs(ctx, sham):
    spots = ctx.study["blind_spots"]
    ctx.notes["families"] = [b["family"] for b in spots]
    ctx.notes["feedback"] = "sham" if sham else "own"
    if sham and len(spots) < 2:
        ctx.notes["sham_degenerate"] = "fewer than two families: the sham note comes from the same family"
    if not spots:
        return
    baseline = best_dev_baseline(ctx)
    last = {}  # family -> its most recent completed candidate (this run only)
    round_ = 0
    while True:
        for i, family in enumerate(spots):
            source = spots[(i + 1) % len(spots)]["family"] if sham else family["family"]
            note = note_for(last.get(source), ctx, baseline) if round_ else None
            name = family["family"]
            yield ctx.family_job(family, note or "",
                                 meta={"feedback_round": round_, "feedback_from": source if note else None},
                                 on_done=lambda c, name=name: last.__setitem__(name, c))
        yield None  # barrier: next round's notes need this round's outcomes
        round_ += 1


def jobs(ctx):
    return _jobs(ctx, sham=False)


def sham_jobs(ctx):
    return _jobs(ctx, sham=True)
