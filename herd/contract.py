"""The output contract of the herd_escape MCP tool: a one-word verdict an agent can branch on,
then one compact JSON object with the chosen lever, a falsifiable next experiment and evidence
where every claim is labeled MEASURED, SOURCE_BACKED or HYPOTHESIZED.

verdict(result) takes an escape_for_agent result dict (herd.server) and returns one of
keep | repair | escape | escalate | inconclusive. Rules, applied in this order:

1. inconclusive: the result has "error".
2. escalate (contradictory evaluator/task, A): the result lists its attempts, every attempt is
   invalid, at least one escape candidate was generated and every candidate is invalid too.
3. inconclusive: fewer than 2 scored attempts (valid, with a score), or no current score
   (evaluation.before is None).
4. escalate (contradictory evaluator/task, B): at least 2 scored attempts, all scores equal
   (within 1e-9 relative), while their strategies differ (strategy_similarity is known and below
   herd.trajectory.SAME_STRATEGY, 0.7): the evaluator does not tell different strategies apart.
5. keep: diagnosis action "continue".
6. repair: action "repair".
7. escape: action "switch_strategy" and a lever was chosen (overlooked_levers non-empty and
   recommended_switch.mechanism set).
8. escalate: action "switch_strategy" but no lever is available.
Anything else is inconclusive.

Scored attempts come from result["attempts"] ([{id, score, valid}], added by the herd_escape
tool); without that list, two scores are inferred from diagnosis.evidence.progress (a relative
improvement is only computed from two or more scores)."""
import json

from .trajectory import SAME_STRATEGY

VERDICTS = ("keep", "repair", "escape", "escalate", "inconclusive")
MEASURED, SOURCE_BACKED, HYPOTHESIZED = "MEASURED", "SOURCE_BACKED", "HYPOTHESIZED"
MARGIN = 0.01        # the escape must beat the best attempt by 1% to count as a success
DEFAULT_TRIES = 3    # candidates to allow when none was generated yet
# trajectory reasons that restate model-extracted fingerprint fields (assumptions, primary
# signal, mechanism family): never MEASURED
MODEL_REASONS = (" assume: ", " decide mainly by ", " implement ")


def _progress(result):
    return ((result.get("diagnosis") or {}).get("evidence") or {}).get("progress") or {}


def scored_attempts(result):
    """Scores of valid attempts, in order; None when the result does not list its attempts."""
    attempts = result.get("attempts")
    if attempts is None:
        return None
    return [a["score"] for a in attempts if a.get("valid", True) and a.get("score") is not None]


def _n_scored(result):
    scores = scored_attempts(result)
    if scores is not None:
        return len(scores)
    p = _progress(result)
    if p.get("relative_improvement") is not None:
        return 2
    return 1 if p.get("best") is not None else 0


def _all_equal(scores):
    return len(scores) >= 2 and all(abs(s - scores[0]) <= 1e-9 * max(1.0, abs(scores[0])) for s in scores)


def _lever_chosen(result):
    return bool(result.get("overlooked_levers")) and bool((result.get("recommended_switch") or {}).get("mechanism"))


def verdict(result):
    if not isinstance(result, dict) or result.get("error"):
        return "inconclusive"
    attempts, candidates = result.get("attempts") or [], result.get("candidates") or []
    if attempts and candidates and all(a.get("valid") is False for a in attempts) \
            and all(c.get("valid") is False for c in candidates):
        return "escalate"
    if _n_scored(result) < 2 or (result.get("evaluation") or {}).get("before") is None:
        return "inconclusive"
    scores, sim = scored_attempts(result), result.get("strategy_similarity")
    if scores and _all_equal(scores) and sim is not None and sim < SAME_STRATEGY:
        return "escalate"
    action = result.get("action") or (result.get("diagnosis") or {}).get("action")
    if action == "continue":
        return "keep"
    if action == "repair":
        return "repair"
    if action == "switch_strategy":
        return "escape" if _lever_chosen(result) else "escalate"
    return "inconclusive"


# ------------------------------------------------------------------ render
def _num(x):
    return f"{x:.6g}" if isinstance(x, (int, float)) else str(x)


def _lower(result):
    return result.get("lower_is_better", True) is not False


def best_score(result):
    scores = scored_attempts(result)
    if scores:
        return (min if _lower(result) else max)(scores)
    return _progress(result).get("best")


def chosen_lever(result):
    if not _lever_chosen(result):
        return None
    lever = result["overlooked_levers"][0]
    return {k: lever.get(k) for k in ("type", "name", "mechanism", "why")}


def _subject(lever):
    if lever["type"] == "unused_signal":
        return f"Making decisions depend on {lever['name']}"
    if lever["type"] == "assumption_break":
        return f'Dropping the assumption "{lever["name"]}"'
    return f"Switching to {lever['name']}"


def next_experiment(result, v, lever):
    best, lower = best_score(result), _lower(result)
    direction = "lower is better" if lower else "higher is better"
    tries = len(result.get("candidates") or []) or DEFAULT_TRIES
    if best is None or v == "inconclusive":
        out = {"hypothesis": "Not enough scored attempts to judge the strategy: score at least two attempts "
                             "on the dev split and call herd_escape again.",
               "success_threshold": None, "failure_threshold": None}
    elif v == "escalate":
        out = {"hypothesis": "The evaluator or task is inconsistent with the attempts (identical scores across "
                             "different strategies, or nothing is valid); check it before trying more code.",
               "success_threshold": "a deliberately different, valid solution gets a different dev score",
               "failure_threshold": f"dev score stays {_num(best)} or invalid for every solution tried"}
    else:
        target = best - MARGIN * abs(best) if lower else best + MARGIN * abs(best)
        op = ("<" if lower else ">") if best == 0 else ("<=" if lower else ">=")
        if lever:
            subject = _subject(lever)
        elif v == "keep":
            subject = "Continuing the current strategy"
        else:
            subject = "Repairing the current strategy (keeping its approach)"
        out = {"hypothesis": f"{subject} will beat the best attempt's dev score {_num(best)}.",
               "success_threshold": f"dev score {op} {_num(target)} ({direction}) and valid",
               "failure_threshold": f"dev score no better than {_num(best)} after {tries} candidate{'s' if tries != 1 else ''}, or invalid"}
    ev = result.get("evaluation") or {}
    if result.get("candidates"):
        out["measured"] = {k: ev.get(k) for k in ("before", "after", "improved", "split")}
    return out


def _family_index(families):
    out = {}
    for f in families or []:
        if isinstance(f, dict) and f.get("family"):
            urls = [u for u in f.get("source_urls") or [] if isinstance(u, str) and u.startswith("http")]
            if urls:
                out.setdefault(f["family"].casefold(), {**f, "source_urls": urls})
    return out


def _sourced(lever, index):
    """SOURCE_BACKED claim for a lever that is a taxonomy family with source URLs, else None."""
    if lever.get("type") not in ("blind_spot", "minority_mechanism"):
        return None
    fam = index.get(str(lever.get("name")).casefold())
    urls = list(fam["source_urls"]) if fam else \
        [u for u in (lever.get("detail") or {}).get("source_urls") or [] if isinstance(u, str) and u.startswith("http")]
    if not urls:
        return None
    mech = (fam or {}).get("one_line_mechanism") or (lever.get("detail") or {}).get("one_line_mechanism")
    claim = f"{lever['name']} is a family described in the literature" + (f": {mech}" if mech else "")
    item = {"claim": claim, "class": SOURCE_BACKED, "url": urls[0]}
    if (fam or {}).get("source_summary"):
        item["detail"] = fam["source_summary"]
    return item


def _herd_counts(lever):
    """MEASURED herd counts carried by a lever's detail, if any."""
    d, name, t = lever.get("detail") or {}, lever.get("name"), lever.get("type")
    if t == "unused_signal" and d.get("used_by") is not None and d.get("decided"):
        item = {"claim": f"{name} used by {d['used_by']}/{d['decided']} herd samples", "class": MEASURED}
        if d.get("used_by_models"):
            item["detail"] = {"used_by_models": d["used_by_models"]}
        return item
    if t == "assumption_break" and d.get("held_by") is not None and d.get("checked"):
        return {"claim": f'assumption "{name}" held by {d["held_by"]}/{d["checked"]} checked herd samples',
                "class": MEASURED, "detail": "per-sample holds are judged by a checker model, with quoted evidence"}
    if t == "minority_mechanism" and d.get("models"):
        n = len(d["models"]) + len(d.get("absent_from") or [])
        return {"claim": f"{name} produced by {len(d['models'])}/{n} sampler models ({d.get('count')} samples)",
                "class": MEASURED, "detail": {"models": d["models"]}}
    if t == "blind_spot" and (d.get("observed") or d.get("share") is not None):
        seen = d.get("observed") or f"share {d['share']}"
        return {"claim": f"{name} observed in the herd: {seen}", "class": MEASURED}
    return None


def evidence(result, lever, families=None):
    out, index = [], _family_index(families)
    ss, cs = result.get("strategy_similarity"), result.get("code_similarity")
    if ss is not None and cs is not None:
        out.append({"claim": f"mean pairwise strategy similarity {ss:.2f} against code similarity {cs:.2f}",
                    "class": MEASURED})
    if result.get("attempts"):
        out.append({"claim": "attempt dev scores: " + ", ".join(
            f"{a.get('id')}={_num(a.get('score')) if a.get('valid', True) else 'invalid'}" for a in result["attempts"]),
            "class": MEASURED})
    else:
        p = _progress(result)
        if p.get("best") is not None:
            out.append({"claim": f"best attempt dev score {_num(p['best'])} vs first {_num(p.get('first'))}",
                        "class": MEASURED})
    for r in result.get("reasons") or []:
        if r.startswith("mean pairwise strategy similarity") and ss is not None:
            continue  # already stated above
        model = any(m in f" {r} " for m in MODEL_REASONS)
        item = {"claim": r, "class": HYPOTHESIZED if model else MEASURED}
        if model:
            item["detail"] = "restates a strategy property extracted by a model from the code"
        out.append(item)
    if result.get("herd_prevalence") is not None:
        out.append({"claim": f"at most {result['herd_prevalence']:.0%} of herd samples share the attempts' input profile",
                    "class": MEASURED})
    levers = result.get("overlooked_levers") or []
    if lever and levers:
        first = levers[0]
        if counts := _herd_counts(first):
            out.append(counts)
        if src := _sourced(first, index):
            out.append(src)
        out.append({"claim": f"Expected effect: {first.get('mechanism')}", "class": HYPOTHESIZED,
                    "detail": "the recommended mechanism; untested until a candidate is scored"})
    for alt in levers[1:] if lever else levers:
        if alt.get("type") == "blind_spot" and (src := _sourced(alt, index)):
            out.append(src)
    ev = result.get("evaluation") or {}
    if result.get("candidates") and ev.get("after") is not None:
        out.append({"claim": f"escape candidate dev score {_num(ev['after'])} vs current {_num(ev.get('before'))} "
                             f"({'improved' if ev.get('improved') else 'not improved'})", "class": MEASURED})
    if s := result.get("counterexample_summary"):
        out.append({"claim": "counterexamples: inputs where every attempt decides alike and the escape does not",
                    "class": MEASURED, "detail": s})
    return out


def scope_note(paraphrase=None, stability=None):
    if paraphrase:
        p, r, v = paraphrase.get("mean_P"), paraphrase.get("max_R"), paraphrase.get("verdict")
    elif stability:
        p, r, v = stability.get("paraphrase_tvd"), stability.get("resampling_tvd_max"), stability.get("verdict")
    else:
        return "Herd shares are for this prompt wording; robustness to rewording was not measured for this task."
    if p is None or r is None:
        return f"Herd shares are for this prompt wording (paraphrase verdict: {v})."
    moved = "rewording moved the map" if p > r else "rewording moved the map no more than resampling"
    return (f"Herd shares are for this prompt wording: {moved} (paraphrase TVD {p:.2f} vs at most {r:.2f} "
            f"between resamples; verdict: {v}).")


def contract(result, overlook=None, paraphrase=None, families=None):
    """The JSON object render() prints under the verdict line."""
    v = verdict(result)
    if result.get("error"):
        return v, {k: result.get(k) for k in ("error", "detail", "spent_usd") if k in result}
    lever = chosen_lever(result)
    out = {"lever": lever, "next_experiment": next_experiment(result, v, lever),
           "evidence": evidence(result, lever, families),
           "scope_note": scope_note(paraphrase, result.get("herd_stability"))}
    for k in ("spent_usd", "counterexamples"):
        if k in result:
            out[k] = result[k]
    out["candidate_path"] = result.get("candidate_path")
    return v, out


def render(result, overlook=None, paraphrase=None, families=None):
    """`VERDICT: <verdict>` then the contract as indented JSON. overlook is accepted for callers
    that have it; the lever counts already come from it via the result."""
    v, body = contract(result, overlook, paraphrase, families)
    return f"VERDICT: {v}\n" + json.dumps(body, indent=2, default=str)


def parse(text):
    """(verdict, body) from render() output."""
    head, _, rest = text.partition("\n")
    return head.removeprefix("VERDICT: ").strip(), json.loads(rest)


# ------------------------------------------------------------------ agent fields
# What an agent should do next for each verdict. herd_escape (herd.server) adds these fields to
# the contract body and also returns them as MCP structuredContent.
MIN_SCORED = 2  # rule 3 above: fewer scored attempts is inconclusive
NEXT_ACTION = {
    "keep": "Continue the existing strategy; do not switch yet. Score each change with the evaluator and "
            "call herd_escape again if scores stall.",
    "repair": "Fix the existing strategy and preserve its mechanism (do not switch approach); re-score it "
              "with the evaluator.",
    "escape": "Implement the named lever (start from candidate_path when one exists), run the evaluator, and "
              "compare against the before score and next_experiment's success and failure thresholds.",
    "escalate": "Stop autonomous retries and ask a human: HERD has no safe next move to name, or the evaluator "
                "looks contradictory (identical scores for different strategies, or nothing is valid).",
    "inconclusive": f"Get at least {MIN_SCORED} valid, scored attempts (earlier ones as previous_attempts, the "
                    f"latest as current_solution), then call herd_escape again.",
}
AUTONOMOUS = {v: v != "escalate" for v in VERDICTS}


def _inconclusive_action(result):
    n = _n_scored(result)
    need = max(0, MIN_SCORED - n)
    if need:
        return (f"Only {n} valid, scored attempt{'s' if n != 1 else ''}: get {need} more (at least {MIN_SCORED} in "
                f"total; earlier ones as previous_attempts, the latest as current_solution), then call herd_escape "
                f"again.")
    if (result.get("evaluation") or {}).get("before") is None:
        return ("current_solution has no valid score: make it pass the evaluator (or pass the latest valid "
                "attempt as current_solution), then call herd_escape again.")
    return NEXT_ACTION["inconclusive"]


def agent_fields(result, v, body):
    """{next_action, can_continue_autonomously, complete, missing_requirements, candidate_available,
    evidence_summary} for verdict v; error results carry their own next_action (herd.server.failure)."""
    if result.get("error"):
        return {"next_action": result.get("next_action") or NEXT_ACTION["inconclusive"],
                "can_continue_autonomously": bool(result.get("can_continue_autonomously", False)),
                "complete": False, "missing_requirements": list(result.get("missing") or []),
                **{k: result[k] for k in ("live_calls_available", "offline_demo_available", "partial")
                   if k in result}}
    action = _inconclusive_action(result) if v == "inconclusive" else NEXT_ACTION[v]
    ev = result.get("evaluation") or {}
    if v == "escape" and result.get("candidates") and not ev.get("improved"):
        action += (" HERD's own candidate did not beat the before score on dev, so the lever is still untested "
                   "by a working implementation.")
    counts = {c: 0 for c in (MEASURED, SOURCE_BACKED, HYPOTHESIZED)}
    for e in body.get("evidence") or []:
        counts[e["class"]] = counts.get(e["class"], 0) + 1
    return {"next_action": action, "can_continue_autonomously": AUTONOMOUS[v], "complete": True,
            "missing_requirements": [], "candidate_available": bool(result.get("candidate_path")),
            "evidence_summary": counts}


def agent_result(result, overlook=None, paraphrase=None, families=None):
    """(text, structured) for the herd_escape tool. text is `VERDICT: <verdict>` then one JSON object:
    the agent fields first, then the contract body (parse() reads it). structured is the same object
    plus "verdict", for MCP clients that read structuredContent."""
    v, body = contract(result, overlook, paraphrase, families)
    merged = {**agent_fields(result, v, body), **body}
    return f"VERDICT: {v}\n" + json.dumps(merged, indent=2, default=str), \
        json.loads(json.dumps({"verdict": v, **merged}, default=str))
