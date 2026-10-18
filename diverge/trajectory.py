"""Trajectory analysis: is an agent's sequence of attempts different code but the same strategy?

loop_score = 0.6 * strategy_similarity + 0.3 * stall + 0.1 * shared_weak, where
- strategy_similarity is the mean pairwise strategy.similarity()["overall"];
- stall is 1 when the best score improves on the first by less than STALL (2%), falls linearly
  to 0 at 10% improvement, and is 0.5 when progress cannot be measured (fewer than two scores);
- shared_weak is 1 when every attempt with dev detail is worse than the baseline on a common case.
A single attempt scores 0. loop_detected needs two or more attempts, strategy_similarity of at
least SAME_STRATEGY and stalled (or unknown) progress; loop_score only ranks how strong it is.

BEHAVIOR DETECTOR (detector "behavior"; used whenever diverge.fingerprint.supported(task), i.e. the
task's evaluator can fingerprint candidates, unless behavior_loop=False). The loop decision then
rests only on measurements; the strategy fields an LLM extractor fills in (decision rule,
assumptions, mechanism family) stay in the report as a caption and never enter it:
(a) same behavior: the RECENT (3) most recent attempts all fall in one behavior cluster
    (diverge.fingerprint.clusters, single linkage at fingerprint.THRESHOLD over all attempts);
(b) same failures: every pair of those clustered attempts has failure-set Jaccard >= SAME_FAILURES
    (0.8), failure sets being diverge.failures.failure_set(dev_detail, k=FAILURE_K): the k worst dev
    cases. All pairs, not the mean, so one attempt that fails elsewhere breaks the loop; with
    k = 2 this means identical failure sets;
(c) stalled: no dev improvement greater than EPSILON over the last 3 attempts. The best score of
    the last two attempts is compared with the best score of every attempt before them; the
    improvement is relative to that best earlier score, and up to EPSILON (1%) counts as none.
    Unknown (fewer than two scores) counts as stalled, as in the rule above.
loop_detected = a and b and c. Attempts without dev_detail are scored on dev with the given
evaluator (or Evaluator(task)), sandboxed; fingerprints are computed in the sandbox too."""
import hashlib
import os
import tempfile
from itertools import combinations
from pathlib import Path

from . import fingerprint as behavioral
from .failures import failure_set, jaccard
from .strategy import code_similarity, fingerprint, similarity

STALL = 0.02
FULL_PROGRESS = 0.10
SAME_STRATEGY = 0.7
WEIGHTS = {"strategy": 0.6, "stall": 0.3, "weak": 0.1}
RECENT = 3            # (a): the most recent attempts that must share one behavior cluster
SAME_FAILURES = 0.8   # (b): minimum pairwise failure-set Jaccard among them
FAILURE_K = 2         # (b): worst dev cases per failure set
EPSILON = 0.01        # (c): relative improvement (of the best earlier score) that still counts as none


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


# ---------------------------------------------------------------- behavior detector
def improvement_last3(scores, lower_is_better):
    """Relative improvement of the best of the last two scores over the best of all earlier
    ones (so over the last 3 attempts when there are 3 or more); None with fewer than 2 scores."""
    scores = [s for s in scores if s is not None]
    if len(scores) < 2:
        return None
    earlier, recent = (scores[:-2], scores[-2:]) if len(scores) >= 3 else (scores[:1], scores[1:])
    pick = min if lower_is_better else max
    before, after = pick(earlier), pick(recent)
    gain = (before - after) if lower_is_better else (after - before)
    return gain / abs(before) if before != 0 else (0.0 if gain == 0 else float("inf") if gain > 0 else float("-inf"))


def _dev_results(task, attempts, evaluator):
    """{id: (score, dev_detail)} for every attempt; attempts without dev_detail are scored on dev
    (sandboxed) with the evaluator, or Evaluator(task) when none is given."""
    from .reality import Evaluator, dev_detail
    out, missing = {}, []
    for a in attempts:
        if a.get("dev_detail"):
            out[a["id"]] = (a.get("score"), a["dev_detail"])
        else:
            missing.append(a)
    if missing:
        evaluator = evaluator or Evaluator(task)
        with tempfile.TemporaryDirectory() as d:
            for a in missing:
                path = os.path.join(d, f"{hashlib.sha256(a['code'].encode()).hexdigest()[:12]}.py")
                with open(path, "w") as f:
                    f.write(a["code"])
                result = evaluator(Path(path))
                out[a["id"]] = (result["score"] if result["valid"] else None, dev_detail(result))
    return out


def behavior_loop_of(task, attempts, evaluator=None, lower_is_better=True):
    """The three measured conditions of the behavior detector (see the module docstring)."""
    ids = [a["id"] for a in attempts]
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for i, a in enumerate(attempts):
            paths[a["id"]] = os.path.join(d, f"attempt-{i}.py")
            with open(paths[a["id"]], "w") as f:
                f.write(a["code"])
        fps = behavioral.fingerprints(task, paths) if paths else {}
    groups = behavioral.clusters(fps)
    cluster_of = {k: i for i, g in enumerate(groups) for k in g}
    recent = ids[-RECENT:]
    recent_clusters = [cluster_of.get(k) for k in recent]
    top = max(set(c for c in recent_clusters if c is not None), key=recent_clusters.count, default=None)
    clustered = [k for k, c in zip(recent, recent_clusters) if c is not None and c == top]
    a_holds = len(clustered) >= RECENT
    spread = max((behavioral.distance(fps[x], fps[y]) for x, y in combinations(clustered, 2)), default=0.0)

    results = _dev_results(task, attempts, evaluator) if attempts else {}
    sets = {k: failure_set(results[k][1], k=FAILURE_K) if results[k][1] and "error" not in results[k][1] else None
            for k in ids}
    pairs = [jaccard(sets[x], sets[y]) if sets[x] is not None and sets[y] is not None else 0.0
             for x, y in combinations(clustered, 2)]
    b_holds = bool(pairs) and min(pairs) >= SAME_FAILURES

    scores = [a.get("score") if a.get("score") is not None else results[a["id"]][0] for a in attempts]
    imp = improvement_last3(scores, lower_is_better)
    c_holds = imp is None or imp <= EPSILON
    return {"clusters": groups, "cluster_of_recent": dict(zip(recent, recent_clusters)),
            "clustered_recent": clustered, "max_probe_disagreement": round(spread, 4),
            "failure_sets": {k: sorted(v) if v is not None else None for k, v in sets.items()},
            "failure_jaccard": {"min": min(pairs) if pairs else None,
                                "mean": sum(pairs) / len(pairs) if pairs else None,
                                "pairs": {f"{x}|{y}": j for (x, y), j in zip(combinations(clustered, 2), pairs)}},
            "improvement_last3": imp, "epsilon": EPSILON, "scores": scores,
            "conditions": {"a": a_holds, "b": b_holds, "c": c_holds},
            "signatures": fps}


def behavior_reasons(b):
    out = []
    recent = list(b["cluster_of_recent"])
    k = len(b["clustered_recent"])
    if b["conditions"]["a"]:
        out.append(f"{k}/{len(recent)} recent attempts in one behavior cluster (max probe disagreement "
                   f"{b['max_probe_disagreement']:.3f} <= {behavioral.THRESHOLD})")
    else:
        out.append(f"only {k}/{len(recent)} recent attempts share a behavior cluster (need {RECENT}; "
                   f"{len(b['clusters'])} clusters over all attempts)")
    sets = [b["failure_sets"][x] for x in b["clustered_recent"]]
    jmin = b["failure_jaccard"]["min"]
    if b["conditions"]["b"]:
        same = sets[0] if sets else []
        identical = "identical" if jmin == 1.0 else "nearly identical"
        out.append(f"failure sets {identical} (Jaccard {jmin:.2f}): {', '.join(same)}")
    elif jmin is not None:
        listing = "; ".join(f"{x}: {', '.join(s or ['invalid'])}" for x, s in zip(b["clustered_recent"], sets))
        out.append(f"failure sets differ (min Jaccard {jmin:.2f} < {SAME_FAILURES}): {listing}")
    else:
        out.append("failure sets not compared: fewer than two recent attempts share a behavior cluster")
    imp = b["improvement_last3"]
    if imp is None:
        out.append("dev improvement unknown: fewer than two scored attempts (counts as stalled)")
    elif b["conditions"]["c"]:
        out.append(f"no dev improvement over the last 3 attempts ({imp:+.1%} vs epsilon {EPSILON:.0%})")
    else:
        out.append(f"dev improvement over the last 3 attempts {imp:+.1%} (> epsilon {EPSILON:.0%})")
    return out


def analyze(task, attempts, extract=None, behavior=False, lower_is_better=True, baseline_detail=None,
            behavior_loop=None, evaluator=None):
    """attempts: [{"id", "code", "score", "dev_detail"}] in chronological order. behavior_loop:
    None = use the behavior detector when the task supports fingerprints, True/False force it
    on/off (on needs a supported task). evaluator scores attempts that lack dev_detail."""
    n = len(attempts)
    use_behavior = behavioral.supported(task) and behavior_loop is not False
    b = behavior_loop_of(task, attempts, evaluator, lower_is_better) if use_behavior else None
    fps = {a["id"]: fingerprint(task, a["code"], extract=extract, behavior=behavior and b is None,
                                dev_detail=a.get("dev_detail"), baseline_detail=baseline_detail)
           for a in attempts}
    if b is not None and behavior:
        for k, fp in fps.items():
            fp["behavior_signature"] = b["signatures"].get(k)
    has_detail = [a["id"] for a in attempts if a.get("dev_detail") and baseline_detail]
    sims = {"code": pairwise([a["code"] for a in attempts], code_similarity),
            "strategy": pairwise(list(fps.values()), lambda x, y: similarity(x, y)["overall"])}
    shared = shared_items(fps, has_detail)
    prog = progress([a.get("score") for a in attempts], lower_is_better)
    rel = prog["relative_improvement"]
    stalled = rel is None or rel < STALL
    loop_score = 0.0 if n < 2 else (WEIGHTS["strategy"] * sims["strategy"] + WEIGHTS["stall"] * stall_of(rel)
                                    + WEIGHTS["weak"] * bool(shared["weak_regimes"]))
    reasons = reasons_for(n, fps, shared, sims, prog, has_detail) if n else []
    if b is None:
        llm_loop = n >= 2 and sims["strategy"] >= SAME_STRATEGY and stalled
        return {"n": n, "fingerprints": fps, "code_similarity": sims["code"], "strategy_similarity": sims["strategy"],
                "shared": shared, "progress": prog, "loop_score": round(loop_score, 4),
                "loop_detected": llm_loop, "detector": "llm", "behavior": None, "reasons": reasons}
    report = {k: v for k, v in b.items() if k != "signatures"}
    return {"n": n, "fingerprints": fps, "code_similarity": sims["code"], "strategy_similarity": sims["strategy"],
            "shared": shared, "progress": prog, "loop_score": round(loop_score, 4),
            "loop_detected": all(b["conditions"].values()), "detector": "behavior", "behavior": report,
            "reasons": reasons + (behavior_reasons(b) if n else [])}
