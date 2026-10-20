"""The output contract of the diverge_escape MCP tool: a one-word verdict an agent can branch on,
then one compact JSON object with the chosen lever, a falsifiable next experiment and evidence
where every claim is labeled MEASURED, SOURCE_BACKED or HYPOTHESIZED.

verdict(result) takes an escape_for_agent result dict (diverge.server) and returns one of
keep | repair | escape | escalate | inconclusive. Rules, applied in this order:

1. inconclusive: the result has "error".
2. escalate (contradictory evaluator/task, A): the result lists its attempts, every attempt is
   invalid, at least one escape candidate was generated and every candidate is invalid too.
3. inconclusive: fewer than 2 scored attempts (valid, with a score), or no current score
   (evaluation.before is None).
4. escalate (contradictory evaluator/task, B): at least 2 scored attempts, all scores equal
   (within 1e-9 relative), while their strategies differ (strategy_similarity is known and below
   diverge.trajectory.SAME_STRATEGY, 0.7): the evaluator does not tell different strategies apart.
5. keep: diagnosis action "continue".
6. repair: action "repair".
7. escape: action "switch_strategy" and a lever was chosen (overlooked_levers non-empty and
   recommended_switch.mechanism set).
8. escalate: action "switch_strategy" but no lever is available.
Anything else is inconclusive.

Scored attempts come from result["attempts"] ([{id, score, valid}], added by the diverge_escape
tool); without that list, two scores are inferred from diagnosis.evidence.progress (a relative
improvement is only computed from two or more scores)."""
import json

from . import plateau as _plateau
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
                             "on the dev split and call diverge_escape again.",
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


def _consensus_counts(lever):
    """MEASURED consensus counts carried by a lever's detail, if any."""
    d, name, t = lever.get("detail") or {}, lever.get("name"), lever.get("type")
    if t == "unused_signal" and d.get("used_by") is not None and d.get("decided"):
        item = {"claim": f"{name} used by {d['used_by']}/{d['decided']} consensus samples", "class": MEASURED}
        if d.get("used_by_models"):
            item["detail"] = {"used_by_models": d["used_by_models"]}
        return item
    if t == "assumption_break" and d.get("held_by") is not None and d.get("checked"):
        return {"claim": f'assumption "{name}" held by {d["held_by"]}/{d["checked"]} checked consensus samples',
                "class": MEASURED, "detail": "per-sample holds are judged by a checker model, with quoted evidence"}
    if t == "minority_mechanism" and d.get("models"):
        n = len(d["models"]) + len(d.get("absent_from") or [])
        return {"claim": f"{name} produced by {len(d['models'])}/{n} sampler models ({d.get('count')} samples)",
                "class": MEASURED, "detail": {"models": d["models"]}}
    if t == "blind_spot" and (d.get("observed") or d.get("share") is not None):
        seen = d.get("observed") or f"share {d['share']}"
        return {"claim": f"{name} observed in the consensus: {seen}", "class": MEASURED}
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
        out.append({"claim": f"at most {result['herd_prevalence']:.0%} of consensus samples share the attempts' input profile",
                    "class": MEASURED})
    levers = result.get("overlooked_levers") or []
    if lever and levers:
        first = levers[0]
        if counts := _consensus_counts(first):
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
        return "Consensus shares are for this prompt wording; robustness to rewording was not measured for this task."
    if p is None or r is None:
        return f"Consensus shares are for this prompt wording (paraphrase verdict: {v})."
    moved = "rewording moved the map" if p > r else "rewording moved the map no more than resampling"
    return (f"Consensus shares are for this prompt wording: {moved} (paraphrase TVD {p:.2f} vs at most {r:.2f} "
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
    if item := _plateau.evidence_item(result, v):  # stalled at the best known baseline (diverge.plateau)
        out["evidence"].append(item)
        out["evidence"].append(_plateau.hypothesis_item(result, v))
    if "plateau" in result:
        out["plateau"] = result["plateau"]
    for k in ("spent_usd", "counterexamples", "agent_inputs"):
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
    """(verdict, body) from render() or agent_result() output. A markdown block may sit between the
    VERDICT line and the JSON object; the JSON starts at the first line that starts with "{" (markdown
    lines never do)."""
    head, _, rest = text.partition("\n")
    lines = rest.split("\n")
    start = next((i for i, line in enumerate(lines) if line.startswith("{")), 0)
    return head.removeprefix("VERDICT: ").strip(), json.loads("\n".join(lines[start:]))


# ------------------------------------------------------------------ agent fields
# What an agent should do next for each verdict. diverge_escape (diverge.server) adds these fields to
# the contract body and also returns them as MCP structuredContent.
MIN_SCORED = 2  # rule 3 above: fewer scored attempts is inconclusive
NEXT_ACTION = {
    "keep": "Continue the existing strategy; do not switch yet. Score each change with the evaluator and "
            "call diverge_escape again if scores stall.",
    "repair": "Fix the existing strategy and preserve its mechanism (do not switch approach); re-score it "
              "with the evaluator.",
    "escape": "Implement the named lever (start from candidate_path when one exists), run the evaluator, and "
              "compare against the before score and next_experiment's success and failure thresholds.",
    "escalate": "Stop autonomous retries and ask a human: DIVERGE has no safe next move to name, or the evaluator "
                "looks contradictory (identical scores for different strategies, or nothing is valid).",
    "inconclusive": f"Get at least {MIN_SCORED} valid, scored attempts (earlier ones as previous_attempts, the "
                    f"latest as current_solution), then call diverge_escape again.",
}
AUTONOMOUS = {v: v != "escalate" for v in VERDICTS}


def _inconclusive_action(result):
    n = _n_scored(result)
    need = max(0, MIN_SCORED - n)
    if need:
        return (f"Only {n} valid, scored attempt{'s' if n != 1 else ''}: get {need} more (at least {MIN_SCORED} in "
                f"total; earlier ones as previous_attempts, the latest as current_solution), then call diverge_escape "
                f"again.")
    if (result.get("evaluation") or {}).get("before") is None:
        return ("current_solution has no valid score: make it pass the evaluator (or pass the latest valid "
                "attempt as current_solution), then call diverge_escape again.")
    return NEXT_ACTION["inconclusive"]


def agent_fields(result, v, body):
    """{next_action, can_continue_autonomously, complete, missing_requirements, candidate_available,
    evidence_summary} for verdict v; error results carry their own next_action (diverge.server.failure)."""
    if result.get("error"):
        return {"next_action": result.get("next_action") or NEXT_ACTION["inconclusive"],
                "can_continue_autonomously": bool(result.get("can_continue_autonomously", False)),
                "complete": False, "missing_requirements": list(result.get("missing") or []),
                **{k: result[k] for k in ("live_calls_available", "offline_demo_available", "partial")
                   if k in result}}
    action = _inconclusive_action(result) if v == "inconclusive" else NEXT_ACTION[v]
    ev = result.get("evaluation") or {}
    if v == "escape" and result.get("candidates") and not ev.get("improved"):
        action += (" DIVERGE's own candidate did not beat the before score on dev, so the lever is still untested "
                   "by a working implementation.")
    action += _plateau.action_extension(result, v)
    counts = {c: 0 for c in (MEASURED, SOURCE_BACKED, HYPOTHESIZED)}
    for e in body.get("evidence") or []:
        counts[e["class"]] = counts.get(e["class"], 0) + 1
    return {"next_action": action, "can_continue_autonomously": AUTONOMOUS[v], "complete": True,
            "missing_requirements": [], "candidate_available": bool(result.get("candidate_path")),
            "evidence_summary": counts}


def agent_result(result, overlook=None, paraphrase=None, families=None):
    """(text, structured) for the diverge_escape tool. text is `VERDICT: <verdict>` then one JSON object:
    the agent fields first, then the contract body (parse() reads it). structured is the same object
    plus "verdict", for MCP clients that read structuredContent."""
    v, body = contract(result, overlook, paraphrase, families)
    merged = {**agent_fields(result, v, body), **body}
    structured = json.loads(json.dumps({"verdict": v, **merged}, default=str))
    return f"VERDICT: {v}\n" + escape_markdown(structured) + "\n" + json.dumps(merged, indent=2, default=str), \
        structured


# ------------------------------------------------------------------ readable markdown
# A short block between the VERDICT line and the JSON. Every value in it is copied from the result
# object (numbers via json.dumps, so they read exactly as in the JSON); nothing is recomputed.
def _md_text(x, limit=300):
    s = " ".join(str(x).split())  # one line: a markdown line never starts with "{" (see parse)
    return s if len(s) <= limit else s[:limit - 3] + "..."


def _md_num(x):
    return json.dumps(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else _md_text(x)


def _yes(x):
    return "yes" if x else "no"


def escape_markdown(obj):
    """Markdown summary of a diverge_escape structured object: verdict, lever, traps, next test."""
    lines = [f"## DIVERGE escape: {_md_text(obj.get('verdict'))}"]
    if obj.get("error"):
        lines.append(f"**Error:** {_md_text(obj['error'])}: {_md_text(obj.get('detail', ''))}")
    else:
        lever = obj.get("lever")
        lines.append(f"**Lever:** {_md_text(lever.get('name'))} ({_md_text(lever.get('type'))}): "
                     f"{_md_text(lever.get('mechanism'))}" if lever else "**Lever:** none chosen")
        ne = obj.get("next_experiment") or {}
        m = ne.get("measured")
        if m:
            lines.append(f"**Measured ({_md_text(m.get('split'))}):** candidate {_md_num(m.get('after'))} vs current "
                         f"{_md_num(m.get('before'))} (improved: {_yes(m.get('improved'))})")
        traps = [f"{_md_text(e['claim'])} (HYPOTHESIZED: not measured)" for e in obj.get("evidence") or []
                 if e.get("class") == HYPOTHESIZED]
        if ne.get("failure_threshold"):
            traps.append(f"fails if: {_md_text(ne['failure_threshold'])}")
        if traps:
            lines += ["**Traps:**", *(f"- {t}" for t in traps)]
        if ne.get("hypothesis"):
            test = _md_text(ne["hypothesis"])
            if ne.get("success_threshold"):
                test += f" Success: {_md_text(ne['success_threshold'])}."
            lines.append(f"**Next test:** {test}")
    lines.append(f"**Next action:** {_md_text(obj.get('next_action'), 600)}")
    return "\n".join(lines) + "\n"


def _family_line(f):
    parts = [f"**{_md_text(f.get('family'))}**"]
    if f.get("score") is not None:
        parts.append(f"score {_md_num(f['score'])}")
    if f.get("share") is not None:
        parts.append(f"consensus share {_md_num(f['share'])}")
    return ", ".join(parts) + (f": {_md_text(f['mechanism'])}" if f.get("mechanism") else "")


def check_markdown(obj):
    """Markdown summary of a diverge_check structured object: verdict, consensus placement, shortlist,
    traps, next test."""
    lines = [f"## DIVERGE check: {_md_text(obj.get('verdict'))}"]
    if obj.get("error"):
        lines.append(f"**Error:** {_md_text(obj['error'])}: {_md_text(obj.get('detail', ''))}")
        lines.append(f"**Next action:** {_md_text(obj.get('next_action'), 600)}")
        return "\n".join(lines) + "\n"
    consensus = obj.get("herd_comparison") if obj.get("mode") == "free_text" else obj
    if consensus and consensus.get("family"):
        line = f"**Your approach:** {_md_text(consensus['family'])}, consensus share {_md_num(consensus.get('share'))}"
        if consensus.get("rank") is not None:
            line += f", rank {_md_num(consensus['rank'])}"
        if "in_top_families" in consensus:
            line += f" (one of the consensus top families: {_yes(consensus['in_top_families'])})"
        lines.append(line)
        if consensus.get("top_families"):
            lines.append("**Consensus top families:** " + ", ".join(
                f"{_md_text(f)} ({_md_num(s)})" for f, s in consensus["top_families"]))
    elif obj.get("herd_note"):
        lines.append(f"**Consensus comparison:** {_md_text(obj['herd_note'])}")
    if obj.get("my_family_score") is not None:
        lines.append(f"**Your family's score ({_md_text(obj.get('scored_on'))}):** {_md_num(obj['my_family_score'])}")
    shortlist = [obj[k] for k in ("blind_spot", "better_family", "diverse_alternative") if obj.get(k)]
    if shortlist:
        lines += ["**Shortlist (MEASURED):**", *(f"- {_family_line(f)}" for f in shortlist)]
    alts = obj.get("alternatives") or []
    if alts:
        lines.append("**Alternatives (HYPOTHESIZED, untested):**")
        for a in alts:
            line = (f"- **{_md_text(a.get('name'))}**: {_md_text(a.get('mechanism'))} "
                    f"Strength: {_md_text(a.get('strength'))} Test: {_md_text(a.get('how_to_test'))}")
            if a.get("test_cost"):
                line += (f" Cost: {_md_text(a['test_cost'])}" + (", decisive" if a.get("decisive") else "")
                         + (f". Wrong if: {_md_text(a['kill_result'])}" if a.get("kill_result") else ""))
            if a.get("ungrounded_numbers"):
                line += " Numbers not in your context: " + ", ".join(_md_num(x) for x in a["ungrounded_numbers"])
            lines.append(line)
    if obj.get("filtered_out"):
        lines.append("**Filtered out:** " + "; ".join(
            f"{_md_text(f.get('name'))} ({_md_text(f.get('reason'))}: {_md_text(f.get('of'), 120)})"
            for f in obj["filtered_out"]))
    if obj.get("relax_hint"):
        lines.append(f"**Relax hint:** {_md_text(obj['relax_hint'])}")
    traps = [f"{_md_text(a.get('name'))}: {_md_text(a.get('trap'))}" for a in alts if a.get("trap")]
    if obj.get("note"):
        traps.append(_md_text(obj["note"]))
    if obj.get("exploratory"):
        traps.append("held-out results were shown before for this evaluator: treat this comparison as exploratory")
    if obj.get("complete") is False and obj.get("stopped_at"):
        traps.append(f"budget stopped testing at {_md_text(obj['stopped_at'])}: the comparison is partial")
    if traps:
        lines += ["**Traps:**", *(f"- {t}" for t in traps)]
    lines.append(f"**Next test:** {_md_text(obj.get('next_action'), 600)}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ diverge_check
FREE_TEXT_ACTION = (
    "These alternatives are HYPOTHESIZED: DIVERGE has not implemented or scored any of them. Test before "
    "committing: implement the promising ones and score each with your evaluator (each how_to_test says what to "
    "measure), or call diverge_check again with evaluator_cmd (a command that takes a code file path as its last "
    'argument and prints {"valid": bool, "score": float}) so DIVERGE implements and scores the consensus families and '
    "blind spots itself.")
CHECK_NEXT_ACTION = {
    "keep": "Keep your approach: no tested family beat it. Build on it, score each change, and call diverge_escape "
            "if scored attempts stall.",
    "try_blind_spot": "Implement blind_spot (start from its code_path, DIVERGE's best implementation) and score it "
                      "with your evaluator before committing to it.",
    "try_other_family": "Switch to better_family (start from its code_path) and score it with your evaluator; "
                        "diverse_alternative, when present, is a rarer family that also beat yours.",
    "inconclusive": "Your family was not scored (see note). Call diverge_check again with a larger k or budget_usd "
                    "(ask the user before raising spend), or check the evaluator.",
    "unclassified": "my_approach matched no family in the consensus map: describe its mechanism more concretely and "
                    "call diverge_check again.",
    "untested": FREE_TEXT_ACTION,
}


def check_result(result):
    """(text, structured) for the diverge_check tool: `VERDICT: <verdict>`, a markdown block, then one
    JSON object (parse() reads it). structured is the same object (it carries verdict, complete and
    next_action; error results keep their own next_action)."""
    v = "inconclusive" if result.get("error") else (result.get("verdict") or "inconclusive")
    obj = {"verdict": v, "complete": result.get("complete", not result.get("error")),
           "next_action": result.get("next_action") or CHECK_NEXT_ACTION.get(v, NEXT_ACTION["inconclusive"]),
           "can_continue_autonomously": result.get("can_continue_autonomously", True),
           **{k: x for k, x in result.items() if k not in ("verdict", "complete", "next_action",
                                                         "can_continue_autonomously")}}
    obj = json.loads(json.dumps(obj, default=str))
    return f"VERDICT: {v}\n" + check_markdown(obj) + "\n" + json.dumps(obj, indent=2), obj
