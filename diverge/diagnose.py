"""DIAGNOSE: given an agent's attempts (a trajectory report from diverge.trajectory), decide whether
to keep going, repair the current strategy, or switch to a different one, and say what the
attempts share. The decision is deterministic; the optional `phrase` callable only rewords text.

Also consensus anchors: implementation details most consensus samples keep although the task does not
require them (a first, purely lexical version)."""
import re

IMPROVING = 0.02     # relative improvement of the best attempt over the first that counts as progress
ANCHOR_SHARE = 0.8   # share of consensus texts that must retain a phrase for it to be a suspected anchor
REQUIREMENT = re.compile(r"\b(must|required|only|exactly)\b", re.I)


def text_of(x):
    """An assumption or regime as text, whether the report gives strings or dicts."""
    if isinstance(x, dict):
        return str(x.get("assumption") or x.get("regime") or x.get("text") or x.get("name") or "")
    return str(x)


def join(items):
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


STOP = {"the", "and", "all", "any", "are", "for", "with", "from", "that", "this", "each", "its", "their",
        "they", "does", "every", "into", "than", "which", "solution", "solutions", "code", "attempt", "attempts"}


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP}


def similar(a, b, threshold=0.6):
    """Loose match of two assumption texts: one contains the other, or they share at least
    `threshold` of the shorter one's content words."""
    na, nb = " ".join(a.lower().split()), " ".join(b.lower().split())
    if na and nb and (na in nb or nb in na):
        return True
    wa, wb = words(a), words(b)
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= threshold


def attempt_count(trajectory):
    return trajectory.get("n") or len(trajectory.get("fingerprints") or {})


def unused_inputs(trajectory, overlook=None):
    """Inputs none of the attempts use: the report's shared inputs_unused, plus audited overlook
    inputs that no attempt's fingerprint uses (or leaves ambiguous)."""
    out = list((trajectory.get("shared") or {}).get("inputs_unused") or [])
    fps = list((trajectory.get("fingerprints") or {}).values())
    if fps and overlook:
        seen = set().union(*(set(fp.get("inputs_used") or []) | set(fp.get("inputs_ambiguous") or []) for fp in fps))
        out += [r["input"] for r in overlook.get("inputs", []) if r["input"] not in seen and r["input"] not in out]
    return out


def consensus_prevalence(trajectory, overlook):
    """(prevalence, evidence): how much of the consensus shares the attempts' input profile.
    For every input the overlook audit decided (share not None) that all attempts use (shared
    inputs_used) or that no attempt uses (unused_inputs), the consensus marginal is `share` or
    `1 - share`. Without per-sample joint data, the share of consensus samples matching ALL of these
    is at most the smallest marginal (exact with a single input), which is what we report; the
    Frechet lower bound max(0, sum(m) - (len(m) - 1)) is kept in the evidence."""
    if not overlook:
        return None, {}
    used = set((trajectory.get("shared") or {}).get("inputs_used") or [])
    unused = set(unused_inputs(trajectory, overlook))
    marginals = {}
    for r in overlook.get("inputs", []):
        if r.get("share") is None:
            continue
        if r["input"] in used:
            marginals[f"uses {r['input']}"] = r["share"]
        elif r["input"] in unused:
            marginals[f"does not use {r['input']}"] = 1 - r["share"]
    if not marginals:
        return None, {}
    upper = min(marginals.values())
    lower = max(0.0, sum(marginals.values()) - (len(marginals) - 1))
    return round(upper, 3), {"marginals": {k: round(v, 3) for k, v in marginals.items()},
                             "bounds": [round(lower, 3), round(upper, 3)], "herd_samples": (overlook.get("herd") or {}).get("valid"),
                             "consensus_samples": (overlook.get("herd") or {}).get("valid")}


def shared_cause(trajectory):
    n, shared = attempt_count(trajectory), trajectory.get("shared") or {}
    fps = list((trajectory.get("fingerprints") or {}).values())
    parts = []
    if used := list(shared.get("inputs_used") or []):
        union = set().union(*(set(fp.get("inputs_used") or []) for fp in fps)) if fps else None
        parts.append(f"decide using {'only ' if union == set(used) else ''}{join(used)}")
    if shared.get("state_persistent") is False:
        parts.append("keep no state")
    elif shared.get("state_persistent") is True:
        parts.append("keep persistent state")
    if shared.get("primary_signal"):
        parts.append(f"primary signal {shared['primary_signal']}")
    if shared.get("mechanism_family"):
        parts.append(f"mechanism family {shared['mechanism_family']}")
    if not parts:
        return f"no shared strategy properties found across {n} attempts"
    return f"all {n} attempts " + ", ".join(parts)


def target_properties(trajectory, overlook=None):
    """What an escape must do differently, read off the shared strategy. Assumptions the task
    forces (overlook's hard assumptions) are never targets."""
    shared = trajectory.get("shared") or {}
    by_share = {r["input"]: r["share"] for r in (overlook or {}).get("inputs", []) if r.get("share") is not None}
    unused = sorted(unused_inputs(trajectory, overlook), key=lambda i: (by_share.get(i, 1.0), i))
    out = [f"use an input the attempts never use: {i}" for i in unused]
    if shared.get("state_persistent") is False:
        out.append("keep persistent state across calls")
    if shared.get("primary_signal"):
        out.append(f"a primary signal other than {shared['primary_signal']}")
    if shared.get("mechanism_family"):
        out.append(f"a mechanism outside {shared['mechanism_family']}")
    hard = [a for a in ((overlook or {}).get("assumptions") or {}).get("assumptions", []) if a.get("kind") == "hard"]
    for a in shared.get("assumptions") or []:
        if not any(similar(text_of(a), h["assumption"]) for h in hard):
            out.append(f"break assumption: {text_of(a)}")
    if weak := [text_of(w) for w in shared.get("weak_regimes") or []]:
        out.append(f"do better where every attempt is weak: {'; '.join(weak)}")
    return out


def polish(phrase, text):
    if not phrase or not text:
        return text
    try:
        out = phrase(text)
    except Exception:
        return text
    return out if isinstance(out, str) and out.strip() else text


def behavior_decision(behavior, weak):
    """(action, repairability, reason) from the behavior detector's measured conditions
    (diverge.trajectory): switch_strategy when (a) same behavior cluster, (b) same failure sets and
    (c) stalled all hold; repair when stalled but (a) or (b) fails; continue when improving."""
    cond = behavior["conditions"]
    if all(cond.values()):
        return ("switch_strategy", "low" if weak else "medium",
                "behavior loop: the recent attempts make the same decisions on the probes, fail on the same "
                "dev cases and have stopped improving: switch strategy")
    if cond["c"]:
        failed = [name for key, name in (("a", "they do not share one behavior cluster"),
                                         ("b", "their failure sets differ")) if not cond[key]]
        return ("repair", "medium" if weak else "high",
                f"stalled, but no behavior loop ({join(failed)}): fix the current strategy")
    return ("continue", "high",
            f"dev improvement {behavior['improvement_last3']:+.1%} over the last 3 attempts "
            f"(> {behavior['epsilon']:.0%}): keep going")


UNDETERMINED_REASON = ("stalled, but DIVERGE cannot tell a strategy loop from a fixable stall: the task has no "
                       "behavior probes and the evaluator reports no per-case results, so shared failures cannot be "
                       "measured (model-written strategy similarity was at chance for this in docs/RESULTS-E1.md and "
                       "is not used)")


def failure_decision(f, weak):
    """(action, repairability, reason) from the failure detector's conditions (diverge.trajectory):
    switch_strategy when (b) same failures and (c) stalled; repair when only stalled; else continue."""
    cond = f["conditions"]
    if cond["b"] and cond["c"]:
        return ("switch_strategy", "low" if weak else "medium",
                "failure loop: the recent attempts fail on the same worst dev cases and have stopped improving: "
                "switch strategy")
    if cond["c"]:
        return ("repair", "medium" if weak else "high",
                "stalled, but the recent attempts fail on different dev cases: fix the current strategy")
    return ("continue", "high", f"dev improvement {f['improvement_last3']:+.1%} over the last 3 attempts "
                                f"(> {f['epsilon']:.0%}): keep going")


def diagnose(task, trajectory, overlook=None, phrase=None):
    """With the behavior detector (trajectory["detector"] == "behavior"), behavior_decision on its
    measured conditions; with the failure detector ("failures"), failure_decision. With no detector
    ("none"): continue while improving, else "undetermined" (the verdict is inconclusive). Reports
    without a detector field (older callers): continue if the best attempt is >= 2% better than the
    first; else switch_strategy if loop_detected; else repair."""
    n, shared = attempt_count(trajectory), trajectory.get("shared") or {}
    progress = trajectory.get("progress") or {}
    gain = progress.get("relative_improvement")
    weak = [text_of(w) for w in shared.get("weak_regimes") or []]
    reasons = list(trajectory.get("reasons") or [])
    behavior = trajectory.get("behavior") if trajectory.get("detector") == "behavior" else None
    detector = trajectory.get("detector")
    f = trajectory.get("failures") or {}
    if behavior:
        action, repairability, line = behavior_decision(behavior, weak)
        reasons.append(line)
    elif detector == "failures":
        action, repairability, line = failure_decision(f, weak)
        reasons.append(line)
    elif detector == "none":
        if f and not f["conditions"]["c"]:
            action, repairability = "continue", "high"
            reasons.append(f"dev improvement {f['improvement_last3']:+.1%} over the last 3 attempts "
                           f"(> {f['epsilon']:.0%}): keep going")
        else:
            action, repairability = "undetermined", None
            reasons.append(UNDETERMINED_REASON)
    elif gain is not None and gain >= IMPROVING:
        action, repairability = "continue", "high"
        reasons.append(f"the best attempt is {gain:.1%} better than the first (>= {IMPROVING:.0%}): keep going")
    elif trajectory.get("loop_detected"):
        action, repairability = "switch_strategy", "low" if weak else "medium"
        reasons.append("loop: the attempts repeat one strategy without progress"
                       + ("; the same weak regimes persist in every attempt" if weak else ""))
    else:
        action, repairability = "repair", "medium" if weak else "high"
        reasons.append("stalled, but the attempts do not repeat one strategy: fix the current one")
    if weak:
        failure = f"all {n} attempts are weak on: {'; '.join(weak)}"
    elif (behavior or f) and action == "continue":
        imp = (behavior or f)["improvement_last3"]
        failure = f"none yet: {imp:+.1%} dev improvement over the last 3 attempts"
    elif action == "continue":
        failure = f"none yet: the best attempt is {gain:.1%} better than the first"
    else:
        failure = f"no progress after {n} attempts"
    cause = shared_cause(trajectory)
    prevalence, prevalence_evidence = consensus_prevalence(trajectory, overlook)
    evidence = {k: trajectory.get(k) for k in ("n", "loop_score", "loop_detected", "strategy_similarity",
                                               "code_similarity", "progress", "detector", "behavior", "failures")}
    evidence.update(shared=shared, herd_prevalence=prevalence_evidence, consensus_prevalence=prevalence_evidence, task=getattr(task, "name", task))
    return {"action": action, "repairability": repairability,
            "failure": polish(phrase, failure), "shared_cause": polish(phrase, cause),
            "target_properties": target_properties(trajectory, overlook),
            "herd_prevalence": prevalence, "evidence": evidence, "reasons": reasons}


# ---------------------------------------------------------------- consensus anchors
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[.\-+#/][A-Za-z0-9_]+)*(?:\+\+|#)?")
FENCED = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def candidate_phrases(text):
    """Backticked terms, then tokens that look like names: acronyms (SHA-256, LRU), words with
    digits or underscores, camelCase, dotted names (Node.js), and capitalized words that do not
    start a sentence (Redis, Weibull)."""
    found = re.findall(r"`([^`\n]+)`", text)
    for sentence in sentences(text):
        sentence = sentence.strip().lstrip("#*->| ")
        for i, m in enumerate(TOKEN.finditer(sentence)):
            w = m.group()
            if (len(re.findall("[A-Z]", w)) >= 2 or re.search(r"\d", w) or "_" in w or re.search(r"[a-z][A-Z]", w)
                    or re.search(r"\w{2,}\.\w{2,}", w) or (i > 0 and len(w) > 1 and w[0].isupper())):
                found.append(w)
    seen, out = set(), []
    for w in (f.strip() for f in found):
        if w and w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out


def mentions(phrase, text):
    return re.search(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)", text.lower()) is not None


def anchors(task_text, consensus_texts, required_text=None, extract=None):
    """Phrases from the task that most consensus texts retain, with whether the task requires them.
    A phrase is required if it appears in a sentence of required_text (default: the task text)
    containing must/required/only/exactly, or inside one of its fenced code blocks (the stated
    interface). Counting is case-insensitive whole-phrase matching; it does not see synonyms."""
    phrases = list(extract(task_text)) if extract else candidate_phrases(task_text)
    required_text = task_text if required_text is None else required_text
    rules = [s for s in sentences(FENCED.sub(" ", required_text)) if REQUIREMENT.search(s)]
    rules += FENCED.findall(required_text)
    total = len(consensus_texts)
    rows = []
    for p in phrases:
        retained = sum(mentions(p, t) for t in consensus_texts)
        required = any(mentions(p, r) for r in rules)
        rows.append({"anchor": p, "retained_by": retained, "total": total, "required": required,
                     "suspected_anchor": bool(total) and retained / total >= ANCHOR_SHARE and not required})
    return sorted(rows, key=lambda r: -r["retained_by"])
