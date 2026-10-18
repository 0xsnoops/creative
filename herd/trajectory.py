"""Trajectory analysis: is an agent's sequence of attempts different code but the same strategy?

loop_score = 0.6 * strategy_similarity + 0.3 * stall + 0.1 * shared_weak, where
- strategy_similarity is the mean pairwise strategy.similarity()["overall"];
- stall is 1 when the best score improves on the first by less than STALL (2%), falls linearly
  to 0 at 10% improvement, and is 0.5 when progress cannot be measured (fewer than two scores);
- shared_weak is 1 when every attempt with dev detail is worse than the baseline on a common case.
A single attempt scores 0. loop_detected needs two or more attempts, strategy_similarity of at
least SAME_STRATEGY and stalled (or unknown) progress; loop_score only ranks how strong it is."""
from itertools import combinations

from .strategy import code_similarity, fingerprint, similarity

STALL = 0.02
FULL_PROGRESS = 0.10
SAME_STRATEGY = 0.7
WEIGHTS = {"strategy": 0.6, "stall": 0.3, "weak": 0.1}


def pairwise(items, fn):
    pairs = list(combinations(items, 2))
    return sum(fn(a, b) for a, b in pairs) / len(pairs) if pairs else None


def common(lists):
    """Items present in every list (case-insensitive), in the first list's order."""
    lists = list(lists)
    if not lists:
        return []
    rest = [{str(x).strip().lower() for x in l} for l in lists[1:]]
    return [x for x in dict.fromkeys(lists[0]) if all(str(x).strip().lower() in r for r in rest)]


def agreed(values):
    keys = {str(v).strip().lower() if v is not None else None for v in values}
    return values[0] if values and len(keys) == 1 else None


def progress(scores, lower_is_better):
    scores = [s for s in scores if s is not None]
    if not scores:
        return {"first": None, "best": None, "relative_improvement": None}
    first, best = scores[0], (min if lower_is_better else max)(scores)
    gain = (first - best) if lower_is_better else (best - first)
    rel = gain / abs(first) if len(scores) >= 2 and first != 0 else None
    return {"first": first, "best": best, "relative_improvement": rel}


def shared_items(fps, has_detail):
    fs = list(fps.values())
    return {"inputs_used": common(f["inputs_used"] for f in fs),
            "inputs_unused": common(f["inputs_unused"] for f in fs),
            "state_persistent": agreed([f["state"]["persistent"] for f in fs]),
            "primary_signal": agreed([f["decision_rule"]["primary_signal"] for f in fs]),
            "mechanism_family": agreed([f["mechanism_family"] for f in fs]),
            "assumptions": common(f["assumptions"] for f in fs),
            "weak_regimes": common(fps[i]["failure_signature"]["weak_regimes"] for i in has_detail)}


def stall_of(rel):
    if rel is None:
        return 0.5
    return min(1.0, max(0.0, 1 - (rel - STALL) / (FULL_PROGRESS - STALL)))


def reasons_for(n, fps, shared, sims, prog, has_detail):
    all_n = f"{n}/{n} attempts"
    out = []
    used_sets = {frozenset(f["inputs_used"]) for f in fps.values()}
    if shared["inputs_used"]:
        only = "only " if len(used_sets) == 1 else ""
        out.append(f"{all_n} use {only}inputs {', '.join(shared['inputs_used'])}")
    if shared["inputs_unused"]:
        out.append(f"{all_n} ignore inputs {', '.join(shared['inputs_unused'])}")
    if shared["state_persistent"] is not None:
        features = common(f["state"]["features"] for f in fps.values())
        out.append(f"{all_n} keep state across calls" + (f" ({', '.join(features)})" if features else "")
                   if shared["state_persistent"] else f"{all_n} keep no state across calls")
    if shared["primary_signal"]:
        out.append(f"{all_n} decide mainly by {shared['primary_signal']}")
    if shared["mechanism_family"]:
        out.append(f"{all_n} implement {shared['mechanism_family']}")
    for a in shared["assumptions"][:3]:
        out.append(f"{all_n} assume: {a}")
    if shared["weak_regimes"]:
        k = len(has_detail)
        out.append(f"{k}/{k} attempts with dev results are worse than the best baseline on "
                   f"{', '.join(shared['weak_regimes'][:3])}")
    if sims["strategy"] is not None:
        out.append(f"mean pairwise strategy similarity {sims['strategy']:.2f} against code similarity "
                   f"{sims['code']:.2f}")
    rel = prog["relative_improvement"]
    if rel is None:
        out.append("progress unknown: fewer than two scored attempts")
    else:
        verdict = "stalled" if rel < STALL else "improving"
        out.append(f"best score {prog['best']:.4g} vs first {prog['first']:.4g}: {rel:.1%} better ({verdict})")
    return out


def analyze(task, attempts, extract=None, behavior=False, lower_is_better=True, baseline_detail=None):
    """attempts: [{"id", "code", "score", "dev_detail"}] in chronological order."""
    n = len(attempts)
    fps = {a["id"]: fingerprint(task, a["code"], extract=extract, behavior=behavior,
                                dev_detail=a.get("dev_detail"), baseline_detail=baseline_detail)
           for a in attempts}
    has_detail = [a["id"] for a in attempts if a.get("dev_detail") and baseline_detail]
    sims = {"code": pairwise([a["code"] for a in attempts], code_similarity),
            "strategy": pairwise(list(fps.values()), lambda x, y: similarity(x, y)["overall"])}
    shared = shared_items(fps, has_detail)
    prog = progress([a.get("score") for a in attempts], lower_is_better)
    rel = prog["relative_improvement"]
    stalled = rel is None or rel < STALL
    loop_score = 0.0 if n < 2 else (WEIGHTS["strategy"] * sims["strategy"] + WEIGHTS["stall"] * stall_of(rel)
                                    + WEIGHTS["weak"] * bool(shared["weak_regimes"]))
    return {"n": n, "fingerprints": fps, "code_similarity": sims["code"], "strategy_similarity": sims["strategy"],
            "shared": shared, "progress": prog, "loop_score": round(loop_score, 4),
            "loop_detected": n >= 2 and sims["strategy"] >= SAME_STRATEGY and stalled,
            "reasons": reasons_for(n, fps, shared, sims, prog, has_detail) if n else []}
