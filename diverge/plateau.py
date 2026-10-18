"""Plateau check: is the stalled best attempt sitting at the best known built-in baseline?

Measured context (docs/RESULTS.md, mini rescue test, preliminary): three of four stuck states sat
exactly at the best built-in baseline's dev score, and no method (escape, try_different, random_lever,
fresh_start) rescued any of those three. When attempts stall AT the reference score, the agent may have
reached what this framing allows: more variations rarely help, and the right move is to reframe the
problem or escalate to a human.

plateau(attempts, baselines, lower_is_better=True, eps=0.005) -> dict | None
    attempts:  [{id, score, valid}] (the diverge_escape result's "attempts" list)
    baselines: task.baselines("dev") -> {name: {"score": float, "valid": bool}} (plain numbers accepted)
    Returns None without baselines or without a valid scored attempt. Otherwise
    {"at_reference", "reference", "reference_score", "best_attempt", "gap", "eps", "split"} where gap is
    best_attempt - reference_score (signed) and at_reference is True when |gap| <= eps * |reference_score|:
    the best attempt matches the best baseline, or beats it by less than eps (relative).

attach(result, baselines) sets result["plateau"] from the result's attempts and lower_is_better.
evidence_item(result, verdict) / action_extension(result, verdict): the contract's extra MEASURED line and
next_action sentence, non-empty only when the verdict is escape or repair, the trajectory is stalled and
plateau.at_reference is True. The numbers in that text come from result["plateau"]."""
from .trajectory import STALL

EPS = 0.005
RESCUE_FACT = "preliminary rescue test: 0/3 such states rescued by any method"
APPLIES_TO = ("escape", "repair")


def _score(entry):
    if isinstance(entry, dict):
        if entry.get("valid") is False:
            return None
        return entry.get("score")
    return entry


def plateau(attempts, baselines, lower_is_better=True, eps=EPS):
    if not baselines:
        return None
    refs = {name: s for name, s in ((n, _score(b)) for n, b in baselines.items())
            if isinstance(s, (int, float)) and not isinstance(s, bool)}
    scores = [a.get("score") for a in attempts or []
              if a.get("valid", True) and isinstance(a.get("score"), (int, float)) and not isinstance(a.get("score"), bool)]
    if not refs or not scores:
        return None
    pick = min if lower_is_better else max
    reference = pick(refs, key=refs.get)
    ref_score, best = refs[reference], pick(scores)
    gap = best - ref_score
    at_reference = abs(gap) <= eps * max(abs(ref_score), 1e-12)
    return {"at_reference": at_reference, "reference": reference, "reference_score": ref_score,
            "best_attempt": best, "gap": gap, "eps": eps, "split": "dev"}


def attach(result, baselines, eps=EPS):
    """Compute the plateau for an escape result and store it under result["plateau"]."""
    result["plateau"] = plateau(result.get("attempts") or [], baselines,
                                result.get("lower_is_better", True) is not False, eps)
    return result["plateau"]


def stalled(result):
    """The trajectory's own stall rule: best improves on first by less than STALL, or unknown."""
    p = ((result.get("diagnosis") or {}).get("evidence") or {}).get("progress") or {}
    rel = p.get("relative_improvement")
    return rel is None or rel < STALL


def applies(result, verdict):
    p = result.get("plateau")
    return bool(p and p.get("at_reference") and verdict in APPLIES_TO and stalled(result))


def _num(x):
    return f"{x:.6g}" if isinstance(x, (int, float)) else str(x)


def evidence_item(result, verdict):
    """MEASURED evidence line, or None when the plateau rule does not apply."""
    if not applies(result, verdict):
        return None
    p = result["plateau"]
    relation = "equals" if p["gap"] == 0 else f"matches (within {p['eps']:.1%})"
    return {"claim": f"best attempt {_num(p['best_attempt'])} {relation} the best known baseline {p['reference']} "
                     f"{_num(p['reference_score'])} on {p['split']}; more variations of this framing rarely help "
                     f"({RESCUE_FACT})",
            "class": "MEASURED", "detail": {"gap": p["gap"], "eps": p["eps"]}}


def action_extension(result, verdict):
    """Sentence appended to next_action, or "" when the plateau rule does not apply."""
    if not applies(result, verdict):
        return ""
    p = result["plateau"]
    what = "lever" if verdict == "escape" else "repair"
    return (f" If the {what} does not beat {p['reference']} ({_num(p['reference_score'])} on {p['split']}), stop "
            f"varying: reframe the problem (restate the objective/constraints, change the representation) or "
            f"escalate to a human.")
