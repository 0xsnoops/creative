"""ESCAPE: when an agent's attempts are different code but the same strategy, pick ONE lever the
attempts (and most of the consensus samples) leave unused and implement a solution built on it.

Levers, in order of preference: an input the consensus overlooks and the attempts never use; an
incidental assumption shared by the consensus and by every attempt; a family only one or two sampler
models produced; a blind-spot family. Evidence comes from the overlook study and dev-split
scores only: held-out results never enter a prompt. Candidate code only runs through the
Evaluator (sandboxed)."""
import hashlib
import inspect
import json
import re
from pathlib import Path

from .diagnose import diagnose, similar, text_of, unused_inputs
from .llm import BudgetExceeded, message_text
from .reality import dev_detail, sandbox, slug
from .sample import extract_code

OVERLOOKED_SHARE = 0.10  # same cut as overlook.OVERLOOKED_SHARE
CODE_LIMIT = 4000


def _pct(x):
    return f"{x:.0%}"


def levers(diagnosis, trajectory, overlook, families=None):
    """Every available lever, in preference order; each carries its mechanism and why."""
    overlook, shared = overlook or {}, trajectory.get("shared") or {}
    n = trajectory.get("n") or len(trajectory.get("fingerprints") or {})
    mine = shared.get("mechanism_family")
    out = []

    unused = unused_inputs(trajectory, overlook)
    rows = [r for r in overlook.get("inputs", []) if r.get("overlooked") and r.get("share") is not None
            and r["share"] <= OVERLOOKED_SHARE and r["input"] in unused]
    for r in sorted(rows, key=lambda r: (r["share"], r["input"])):
        out.append({"type": "unused_signal", "name": r["input"],
                    "detail": {k: r.get(k) for k in ("description", "share", "used_by", "decided", "used_by_models")}
                    | {"evidence": (r.get("evidence") or [])[:3]},
                    "mechanism": f"Build the decision rule around {r['input']} ({r.get('description') or 'unused input'}), "
                                 f"so that different values of it change which option is chosen.",
                    "why": f"{r['input']} is used by {r.get('used_by')}/{r.get('decided')} consensus samples "
                           f"({_pct(r['share'])}); none of your {n} attempts use it"})

    mine_assumptions = [text_of(a) for a in shared.get("assumptions") or []]
    for a in (overlook.get("assumptions") or {}).get("assumptions", []):
        if a.get("kind") != "incidental" or not a.get("shared"):
            continue
        match = next((m for m in mine_assumptions if similar(m, a["assumption"])), None)
        if match is None:
            continue
        out.append({"type": "assumption_break", "name": a["assumption"],
                    "detail": {"held_by": a.get("held_by"), "checked": a.get("checked"), "matched_attempt_assumption": match,
                               "evidence": (a.get("evidence") or [])[:3]},
                    "mechanism": f'Drop the assumption "{a["assumption"]}" and let the decisions exploit '
                                 f"what becomes possible without it.",
                    "why": f"held by {a.get('held_by')}/{a.get('checked')} checked consensus samples and by all {n} of your "
                           f"attempts; the task statement does not force it"})

    for f in (overlook.get("minority") or {}).get("families", []):
        if mine and f["item"].casefold() == str(mine).casefold():
            continue
        n_models = len(f["models"]) + len(f.get("absent_from") or [])
        out.append({"type": "minority_mechanism", "name": f["item"],
                    "detail": {k: f.get(k) for k in ("models", "absent_from", "count")},
                    "mechanism": f"Implement the {f['item']} approach, which only {', '.join(f['models'])} produced.",
                    "why": f"produced by {len(f['models'])}/{n_models} sampler models ({f.get('count')} samples)"
                           + (f"; your attempts use {mine}" if mine else "")})

    for fam in families or []:
        fam = fam if isinstance(fam, dict) else {"family": str(fam)}
        if mine and fam["family"].casefold() == str(mine).casefold():
            continue
        seen = fam.get("observed") or (f"share {_pct(fam['share'])}" if fam.get("share") is not None else "rarely observed")
        out.append({"type": "blind_spot", "name": fam["family"],
                    "detail": {k: fam.get(k) for k in ("one_line_mechanism", "share", "observed", "source_urls")},
                    "mechanism": f"{fam['family']}: {fam.get('one_line_mechanism') or 'a sourced family the consensus rarely uses'}.",
                    "why": f"a sourced family the consensus rarely produces ({seen})"})
    return out


LEVER_POLICIES = ("chosen", "random")
RANDOM_POOL_SHARE = 0.5  # the rescue test's random_lever control drew from inputs this share of the consensus uses


def attempt_levers(trajectory):
    """Levers from the attempts alone, for tasks without an overlook study: inputs of the task
    interface that no attempt uses (read by the extractor model, so HYPOTHESIZED)."""
    n = trajectory.get("n") or len(trajectory.get("fingerprints") or {})
    return [{"type": "attempt_unused_input", "name": name, "detail": {},
             "mechanism": f"Build the decision rule around {name}, so that different values of it change which "
                          f"option is chosen.",
             "why": f"none of your {n} attempts use {name} (read from the code by a model; no consensus data for "
                    f"this task)"}
            for name in (trajectory.get("shared") or {}).get("inputs_unused") or []]


def random_lever(trajectory, overlook, found, attempts, seed=0):
    """A random lever, seeded by the attempts' code (same attempts, same draw). With an overlook study:
    an input at least RANDOM_POOL_SHARE of consensus samples use, as the rescue test's random_lever
    control arm (3/4 stuck states rescued vs 1/4 for the chosen lever, docs/RESULTS.md round 2);
    else any available lever, else None."""
    import random
    pool = [r["input"] for r in (overlook or {}).get("inputs", []) if (r.get("share") or 0) >= RANDOM_POOL_SHARE]
    digest = hashlib.sha256("\0".join(a.get("code") or "" for a in attempts).encode()).hexdigest()
    rng = random.Random(int(digest[:12], 16) + seed)
    if pool:
        lever = _lever_v2(rng.choice(sorted(pool)), overlook)
        lever["why"] = (f"drawn at random ({lever['why']}); in the rescue test a random lever rescued more stuck "
                        f"states than the chosen one (docs/RESULTS.md)")
        return {**lever, "type": "random_input"}
    if found:
        lever = dict(rng.choice(found))
        return {**lever, "why": f"drawn at random from the available levers ({lever.get('why')})"}
    return None


def hard_constraints(overlook):
    quotes = [a["prompt_quote"] for a in ((overlook or {}).get("assumptions") or {}).get("assumptions", [])
              if a.get("kind") == "hard" and a.get("prompt_quote")]
    return list(dict.fromkeys(quotes))


def choose(diagnosis, trajectory, overlook, families=None):
    """One lever (the first available in preference order) plus up to 3 alternatives."""
    found = levers(diagnosis, trajectory, overlook, families)
    evidence = {"hard_constraints": hard_constraints(overlook),
                "unused_inputs": unused_inputs(trajectory, overlook),
                "attempt_mechanism": (trajectory.get("shared") or {}).get("mechanism_family"),
                "n_attempts": trajectory.get("n"), "herd": (overlook or {}).get("herd"),
                "consensus": (overlook or {}).get("herd"),
                "paraphrase_verdict": ((overlook or {}).get("paraphrase_test") or {}).get("verdict")}
    if not found:
        return {"lever": None, "mechanism": None, "evidence": evidence, "alternatives": [],
                "why": "no lever found: no overlooked unused input, shared incidental assumption, "
                       "minority family or blind-spot family is available"}
    first = found[0]
    return {"lever": {k: first[k] for k in ("type", "name", "detail")}, "mechanism": first["mechanism"],
            "why": first["why"], "evidence": evidence, "alternatives": found[1:4]}


ESCAPE_MARKER = "Strategy to move away from"
ESCAPE_TEMPLATE = """{task}

---
""" + ESCAPE_MARKER + """: {shared_cause}.
Earlier attempts changed the code but kept this strategy. Observed failure: {failure}.

Keep these requirements of the task (quoted from it):
{constraints}

Pursue this one change instead:
Lever ({lever_type}): {lever}
Mechanism: {mechanism}
Evidence: {why}{notes}

The current solution, shown only as what to move away from (do not tune it):
```python
{code}
```

Write a complete solution whose decisions materially depend on {lever}: changing it must change
which option is chosen. Keep the task's interface and rules.
Reply with your complete solution in a single Python code block."""


def escape_prompt(task, choice, diagnosis, current_code, notes=""):
    lever = choice["lever"]
    if lever is None:
        raise ValueError("no lever to pursue")
    quotes = choice.get("evidence", {}).get("hard_constraints") or []
    constraints = "\n".join(f'- "{q}"' for q in quotes) or "- (none recorded beyond the interface and rules above)"
    return ESCAPE_TEMPLATE.format(task=task.prompt, shared_cause=diagnosis["shared_cause"],
                                  failure=diagnosis["failure"], constraints=constraints,
                                  lever_type=lever["type"].replace("_", " "), lever=lever["name"],
                                  mechanism=choice["mechanism"], why=choice["why"],
                                  notes=f"\n\n{notes}" if notes else "", code=(current_code or "")[:CODE_LIMIT])


def generate(session, task, choice, diagnosis, current_code, evaluator, k=1, seed=0, notes=""):
    """k implementations of the escape prompt, each scored on the dev split."""
    prompt = escape_prompt(task, choice, diagnosis, current_code, notes)
    code_dir = session.out_dir(task) / "escape"
    code_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for i in range(k):
        try:
            entry = session.llm.chat(session.cfg["models"]["implementer"], [{"role": "user", "content": prompt}],
                                  temperature=1.0, max_tokens=session.cfg["reality_check"]["max_tokens"],
                                  sample_idx=f"escape:{seed}:{i}")
        except BudgetExceeded:
            raise
        except Exception as e:
            candidates.append({"path": None, "valid": False, "score": None, "error": sandbox.redact(str(e))[:500]})
            continue
        code = extract_code(message_text(entry), task.entrypoint)
        if code is None:
            candidates.append({"path": None, "valid": False, "score": None, "error": "no code in reply"})
            continue
        path = code_dir / f"{slug(choice['lever']['name'])}-{seed}-{i}.py"
        path.write_text(code)
        result = evaluator(path)  # dev split
        error = (result.get("detail") or {}).get("error")
        candidates.append({"path": str(path), "valid": result["valid"], "score": result["score"],
                           "error": sandbox.redact(str(error))[:500] if error else None, "code": code})
    valid = [c for c in candidates if c["valid"]]
    best = None
    if valid:
        top = evaluator.best([c["score"] for c in valid])
        best = next(c for c in valid if c["score"] == top)
    return {"candidates": [{k: v for k, v in c.items() if k != "code"} for c in candidates], "best": best,
            "prompt": prompt}


def _attempt(a, i):
    if isinstance(a, str):
        return {"id": f"attempt-{i}", "code": a, "score": None}
    return {"id": a.get("id") or f"attempt-{i}", **{k: v for k, v in a.items() if k != "id"}}


def baseline_detail(task, evaluator, attempts):
    """Per-case dev results of the task's best built-in baseline, for weak-regime comparison;
    None for external evaluators or when no attempt has dev detail."""
    if getattr(evaluator, "command", None) or not task.evaluator or not any(a.get("dev_detail") for a in attempts):
        return None
    valid = [r for r in task.baselines("dev").values() if r["valid"]]
    return min(valid, key=lambda r: r["score"])["detail"] if valid else None


def _analyze(task, attempts, evaluator, extract):
    """diverge.trajectory.analyze, called with whichever of these arguments it takes by name."""
    from . import trajectory  # imported late: written in parallel with this module
    params = inspect.signature(trajectory.analyze).parameters
    if "attempts" not in params:
        return trajectory.analyze(task, attempts)
    offered = {"task": task, "attempts": attempts, "evaluator": evaluator, "extract": extract,
               "lower_is_better": getattr(evaluator, "lower_is_better", True)}
    if "baseline_detail" in params:
        offered["baseline_detail"] = baseline_detail(task, evaluator, attempts)
    return trajectory.analyze(**{k: v for k, v in offered.items() if k in params})


def escape(session, task, current_solution, previous_attempts, evaluator, overlook=None, families=None,
           extract=None, k=1, notes=""):
    """Diagnose the attempts and, if they loop on one strategy, implement one escape lever. notes (the
    agent's already-tried outcomes, hard rules and context; diverge.alternatives.escape_notes) go into
    the implementation prompt."""
    attempts = [_attempt(a, i) for i, a in enumerate([*previous_attempts, current_solution])]
    score_dir = session.out_dir(task) / "escape" / "attempts"
    for a in attempts:  # attempts given without a score are scored on dev, sandboxed
        if a.get("score") is None and a.get("code") and "valid" not in a:
            score_dir.mkdir(parents=True, exist_ok=True)
            path = score_dir / f"{hashlib.sha256(a['code'].encode()).hexdigest()[:12]}.py"
            path.write_text(a["code"])
            result = evaluator(path)
            a.update(score=result["score"], valid=result["valid"], dev_detail=dev_detail(result))
    current = attempts[-1]
    report = _analyze(task, attempts, evaluator, extract)
    diagnosis = diagnose(task, report, overlook)
    choice = choose(diagnosis, report, overlook, families)
    generated = None
    if diagnosis["action"] == "switch_strategy" and choice["lever"] is not None:
        generated = generate(session, task, choice, diagnosis, current["code"], evaluator, k=k, notes=notes)
    best = (generated or {}).get("best")
    before, after = current.get("score"), best["score"] if best else None
    improved = after is not None and (before is None or evaluator.beats(after, before))
    lever = [{k: choice["lever"][k] for k in ("type", "name", "detail")} | {"mechanism": choice["mechanism"],
                                                                            "why": choice["why"]}] if choice["lever"] else []
    return {"loop_detected": report.get("loop_detected"), "strategy_similarity": report.get("strategy_similarity"),
            "code_similarity": report.get("code_similarity"), "shared_strategy": diagnosis["shared_cause"],
            "shared_assumptions": (report.get("shared") or {}).get("assumptions", []),
            "herd_prevalence": diagnosis["herd_prevalence"],
            "consensus_prevalence": diagnosis["herd_prevalence"],  # same value; current name
            "overlooked_levers": lever + choice["alternatives"],
            "recommended_switch": {"mechanism": choice["mechanism"], "why": choice["why"]},
            "candidate": best["code"] if best else None, "candidate_path": best["path"] if best else None,
            "candidates": generated["candidates"] if generated else [],
            "evaluation": {"before": before, "after": after, "improved": improved, "split": "dev"},
            "action": diagnosis["action"], "diagnosis": diagnosis, "reasons": diagnosis["reasons"]}


# ================================================================ escape v2
# Research-guided escape (docs/RESEARCH-loop-escape.md: consolidated restart, agreement
# counterexample, idea-level search, admission gate). Fresh contexts only: no attempt code ever
# enters a v2 prompt; the agent sees a failure summary, one lever with its consensus evidence and one
# agreement counterexample, proposes mechanism sketches, and only sketches that differ from the
# stuck strategy are implemented. Every implementation must pass admission before dev scoring.
CX_CASES = 1000  # better-than-stuck cases are rare on tiny inputs (about 1 in 1000 vs best fit)
MAX_ALTERNATIVES = 5
NO_FUTURE = ("The alternative decided using only the current item and the current bin state "
             "(no future items).")
NO_FUTURE_CACHE = ("The alternative decided using only the requests seen so far and the current cache "
                   "contents (no future requests).")

SKETCH_MARKER = "Propose mechanism sketches"
SKETCH_TEMPLATE = """{task}

---
""" + SKETCH_MARKER + """ for a fresh solution. Earlier attempts are summarised below; their code is
deliberately not shown.

Failure summary (dev split):
{summary}

Lever to build on: {lever} ({description}).
Evidence: {why}{evidence}

{counterexample}

Keep these requirements of the task (quoted from it):
{constraints}

Propose exactly {k} mechanism sketches in plain language (no code). Each sketch must make its
decisions depend on {lever}, so that changing {lever} changes which option is chosen, and must be a
different mechanism from the shared strategy above (not a re-weighting or re-tuning of it).
Reply with only a JSON object:
{{"sketches": [{{"name": "...", "state": "what is kept across calls, or none",
"decision_rule": "how each decision is made", "uses_lever": "how the decision depends on {lever}"}}]}}"""

JUDGE_MARKER = "Same mechanism?"
JUDGE_TEMPLATE = JUDGE_MARKER + """ Compare a proposed mechanism with the shared strategy of an agent's stuck attempts.

Shared strategy of the stuck attempts: {shared}

Proposed mechanism:
- name: {name}
- state: {state}
- decision rule: {decision_rule}
- uses {lever}: {uses_lever}

Is the proposal the same mechanism as the shared strategy (same decision signal and rule, possibly
re-weighted, re-scaled or re-tuned)? Reply with only a JSON object:
{{"same_mechanism": true or false, "reason": "one sentence"}}"""

USES_MARKER = "Uses the lever?"
USES_TEMPLATE = USES_MARKER + """ Does this proposed mechanism make its decisions depend on the named lever?

Lever: {lever}{description}

Proposed mechanism:
- name: {name}
- state: {state}
- decision rule: {decision_rule}
- how it uses the lever: {uses_lever}

Answer true only if changing what the lever refers to would change which option the decision rule
picks (for a family lever: the rule implements that family's mechanism). Reply with only a JSON object:
{{"uses_lever": true or false, "reason": "one sentence"}}"""

IMPLEMENT_MARKER = "Implement this mechanism"
IMPLEMENT_TEMPLATE = """{task}

---
""" + IMPLEMENT_MARKER + """ (a fresh design; no earlier code is shown on purpose):
Name: {name}
State kept across calls: {state}
Decision rule: {decision_rule}
How the decision depends on {lever}: {uses_lever}

Lever: {lever} ({description}). The decisions must materially depend on {lever}: changing it
must change which option is chosen.

Keep these requirements of the task (quoted from it):
{constraints}

Write a complete solution that implements this mechanism faithfully, keeping the task's
interface and rules. Reply with your complete solution in a single Python code block."""


class _Counted:
    """diverge.llm with a per-role call counter (ask_json may retry, so count at chat level)."""

    def __init__(self, llm, calls, role):
        self.llm, self.calls, self.role = llm, calls, role

    def chat(self, *args, **kwargs):
        self.calls[self.role] += 1
        return self.llm.chat(*args, **kwargs)


def _lever_v2(lever, overlook):
    """A lever dict {type, name, detail, mechanism, why} from a name (an input of the overlook
    study, e.g. a control arm's random used input) or a dict."""
    if isinstance(lever, dict):
        name = lever["name"]
        lever = {"type": "input", "detail": {}, **lever}
    else:
        name, lever = str(lever), None
    row = next((r for r in (overlook or {}).get("inputs", []) if r.get("input") == name), None)
    if lever is None:
        detail = {k: (row or {}).get(k) for k in ("description", "share", "used_by", "decided")}
        detail["evidence"] = ((row or {}).get("evidence") or [])[:3]
        share = f"{row['used_by']}/{row['decided']} consensus samples" if row and row.get("decided") else "unknown"
        lever = {"type": "input", "name": name, "detail": detail,
                 "mechanism": f"Build the decision rule around {name}, so that different values of it change "
                              f"which option is chosen.",
                 "why": f"{name} is used by {share}"}
    lever.setdefault("mechanism", f"Build the decision rule around {name}.")
    lever.setdefault("why", "given explicitly")
    if row and not (lever.get("detail") or {}).get("description"):
        lever["detail"] = {**(lever.get("detail") or {}), "description": row.get("description")}
    return lever


def _lever_samples(lever, overlook):
    """Consensus sample ids that use the lever: the overlook input row's evidence, else the lever's own."""
    row = next((r for r in (overlook or {}).get("inputs", []) if r.get("input") == lever["name"]), None)
    evidence = (row or {}).get("evidence") or (lever.get("detail") or {}).get("evidence") or []
    ids = [e.get("sample_id") if isinstance(e, dict) else e for e in evidence]
    return list(dict.fromkeys(i for i in ids if isinstance(i, str)))


def _consensus_programs(session, task, sample_ids, limit=MAX_ALTERNATIVES):
    """{sample_id: code} for up to limit valid consensus samples among sample_ids."""
    out_dir = session.out_dir(task)
    path = out_dir / "samples.jsonl"
    if not sample_ids or not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows[r.get("sample_id")] = r
    out = {}
    for sid in sample_ids:
        r = rows.get(sid)
        if not r or r.get("status") != "valid" or not r.get("code_path"):
            continue
        code_path = Path(r["code_path"])
        code_path = code_path if code_path.is_absolute() else out_dir / code_path
        if code_path.exists():
            out[sid] = code_path.read_text()
        if len(out) >= limit:
            break
    return out


def _bin_state(items, chosen, capacity):
    """Free space of each open bin after placing items at the bins with the chosen free space."""
    remaining = []
    for item, free in zip(items, chosen):
        i = remaining.index(free) if free < capacity and free in remaining else len(remaining)
        if i == len(remaining):
            remaining.append(capacity)
        remaining[i] -= item
    return remaining


def describe_counterexample(task_name, cx, lever_name, n_attempts):
    """The counterexample as an online decision: {"step", "decision", "decision_text", "outcome_text",
    "text"}. The decision part is built only from what an online policy has at that step (items
    so far, the current item, the current bin/cache state); later items appear only in the outcome."""
    from . import counterexample as ce
    who = {1: "the attempt", 2: "both attempts"}.get(n_attempts, f"all {n_attempts} attempts")
    case, a, b = cx["case"], cx["attempts_output"], cx["alternative_output"]
    alt = f"the alternative (consensus sample {cx['alternative_id']}, which uses {lever_name})"
    if task_name == ce.BIN:
        cap, items = ce.CAPACITY, list(case["items"])
        i = next(j for j, (x, y) in enumerate(zip(a["chosen"], b["chosen"])) if x != y)
        state = _bin_state(items[:i], a["chosen"][:i], cap)

        def where(out):
            free = out["chosen"][i]
            if free == cap:
                return "a new bin"
            idx = (out.get("bins") or [None] * len(items))[i]
            return f"bin {idx + 1} ({free} free)" if idx is not None else f"the bin with {free} free"
        decision = {"step": i + 1, "item": items[i], "placed_so_far": items[:i], "bins_free": state,
                    "attempts_choice": where(a), "alternative_choice": where(b)}
        bins = ", ".join(f"bin {k + 1}: {f} free" for k, f in enumerate(state)) or "no open bins"
        decision_text = (f"Decision at step {i + 1}: current item size {items[i]}; items already placed: "
                         f"{', '.join(map(str, items[:i])) or 'none'}; open bins before placing ({bins}; "
                         f"capacity {cap}).\n- {who} put it in {decision['attempts_choice']}.\n"
                         f"- {alt} puts it in {decision['alternative_choice']}.\n{NO_FUTURE}")
        outcome_text = (f"Outcome (items arriving after step {i + 1} were unknown at that decision: "
                        f"{', '.join(map(str, items[i + 1:])) or 'none'}): bins used over the whole sequence: "
                        f"attempts {a['n_bins']}, alternative {b['n_bins']}.")
        header = f"Item sequence in arrival order (one at a time, online): {', '.join(map(str, items))}."
    else:
        trace, capacity = list(case["trace"]), case["capacity"]
        i = next(j for j, (x, y) in enumerate(zip(a["evictions"], b["evictions"])) if x != y)
        cache, queue, step = [], list(a["evictions"]), None
        for s, key in enumerate(trace, 1):
            if key in cache:
                continue
            if len(cache) >= capacity:
                if len(a["evictions"]) - len(queue) == i:
                    step = s
                    break
                cache.remove(queue.pop(0))
            cache.append(key)
        step = step or len(trace)
        decision = {"step": step, "request": trace[step - 1], "seen_so_far": trace[:step - 1],
                    "cache": list(cache), "attempts_choice": a["evictions"][i], "alternative_choice": b["evictions"][i]}
        decision_text = (f"Decision at step {step}: request for key {trace[step - 1]} misses; requests so far: "
                         f"{' '.join(map(str, trace[:step - 1]))}; cache (capacity {capacity}) holds "
                         f"{', '.join(map(str, cache))}.\n- {who} evict key {a['evictions'][i]}.\n"
                         f"- {alt} evicts key {b['evictions'][i]}.\n{NO_FUTURE_CACHE}")
        outcome_text = (f"Outcome (requests after step {step} were unknown at that decision: "
                        f"{' '.join(map(str, trace[step:])) or 'none'}): misses over the whole trace: "
                        f"attempts {a['misses']}, alternative {b['misses']}.")
        header = f"Request trace in arrival order (online): {' '.join(map(str, trace))}."
    text = f"Agreement counterexample (every stuck attempt makes the same choices on this case):\n{header}\n" \
           f"{decision_text}\n{outcome_text}"
    return {"step": decision["step"], "decision": decision, "decision_text": decision_text,
            "outcome_text": outcome_text, "text": text}


def _counterexample_v2(session, task, attempts, lever, overlook, seed):
    """The best agreement counterexample whose alternative (a consensus program using the lever) is
    better, or None; plus a note on what was searched."""
    from . import counterexample as ce
    if task.name not in (ce.BIN, ce.CACHE) or not getattr(task, "evaluator", None):
        return None, "no counterexample: this task has no probe mode"
    alternatives = _consensus_programs(session, task, _lever_samples(lever, overlook))
    if not alternatives:
        return None, f"no counterexample: no valid consensus program that uses {lever['name']} was found"
    stuck = {a["id"]: a["code"] for a in attempts if a.get("code")}
    try:
        found = ce.find(task, stuck, alternatives, n=CX_CASES, seed=seed)
    except BudgetExceeded:
        raise
    except Exception as e:  # probes are sandboxed; a probe failure only means no counterexample
        return None, f"no counterexample: probe failed ({sandbox.redact(str(e))[:200]})"
    better = [c for c in found["counterexamples"] if c["alternative_better"]]
    if not better:
        return None, (f"no counterexample: on {found['summary']['cases']} tiny cases no consensus program using "
                      f"{lever['name']} beat the attempts where they all agree")
    cx = better[0]
    described = describe_counterexample(task.name, cx, lever["name"], found["summary"]["valid_attempts"])
    return ({"case": cx["case"], "alternative_id": cx["alternative_id"], "attempts_output": cx["attempts_output"],
             "alternative_output": cx["alternative_output"], "alternative_better": True,
             "summary": found["summary"], **described}, None)


def failure_summary(attempts, diagnosis, evaluator, k=2, notes=""):
    """Dev scores, the best score, the shared failure set and the shared strategy, then the agent's notes
    (diverge.alternatives.escape_notes; plain language); never code."""
    from .failures import failure_set, jaccard
    scored = [a for a in attempts if a.get("score") is not None]
    lines = ["Attempt dev scores (" + ("lower" if getattr(evaluator, "lower_is_better", True) else "higher")
             + " is better): " + (", ".join(f"{a['id']}={a['score']:.4g}" for a in scored) or "none valid")]
    if scored:
        lines.append(f"Best dev score so far: {evaluator.best([a['score'] for a in scored]):.4g}")
    sets = {a["id"]: failure_set(a.get("dev_detail"), k) for a in attempts}
    sets = {i: s for i, s in sets.items() if s}
    if sets:
        shared = frozenset.intersection(*sets.values())
        if shared and len(sets) == len(attempts):
            lines.append(f"All {len(attempts)} attempts are worst on dev instances {', '.join(sorted(shared))}.")
        else:
            ids = list(sets)
            pairs = [jaccard(sets[x], sets[y]) for n, x in enumerate(ids) for y in ids[n + 1:]]
            overlap = f"; mean pairwise overlap {sum(pairs) / len(pairs):.2f}" if pairs else ""
            lines.append("Worst dev instances per attempt: "
                         + "; ".join(f"{i}: {', '.join(sorted(s))}" for i, s in sets.items()) + overlap + ".")
    lines.append(f"Shared strategy of the attempts: {diagnosis['shared_cause']}.")
    if diagnosis.get("failure"):
        lines.append(f"Observed failure: {diagnosis['failure']}.")
    if notes:
        lines.append(notes)
    return "\n".join(lines)


def _lever_text(lever):
    detail = lever.get("detail") or {}
    description = detail.get("description") or lever.get("mechanism") or "the lever"
    evidence = [e for e in detail.get("evidence") or [] if isinstance(e, dict) and e.get("evidence")]
    ev = "".join(f"\n- {e.get('model') or e.get('sample_id')}: {e['evidence']}" for e in evidence[:3])
    return description, (f"\nHow consensus programs use it:{ev}" if ev else "")


def _constraints(overlook):
    return "\n".join(f'- "{q}"' for q in hard_constraints(overlook)) or "- (none recorded beyond the interface and rules above)"


def sketch_prompt(task, summary, lever, counterexample, overlook, k=3, no_cx_note=None):
    description, evidence = _lever_text(lever)
    cx = counterexample["text"] if counterexample else (no_cx_note or "No agreement counterexample was found.")
    return SKETCH_TEMPLATE.format(task=task.prompt, summary=summary, lever=lever["name"], description=description,
                                  why=lever.get("why") or "", evidence=evidence, counterexample=cx,
                                  constraints=_constraints(overlook), k=k)


def implementation_prompt_v2(task, sketch, lever, overlook):
    description, _ = _lever_text(lever)
    return IMPLEMENT_TEMPLATE.format(task=task.prompt, name=sketch.get("name", ""), state=sketch.get("state", "none"),
                                     decision_rule=sketch.get("decision_rule", ""), uses_lever=sketch.get("uses_lever", ""),
                                     lever=lever["name"], description=description, constraints=_constraints(overlook))


def lever_terms(lever):
    """Strings whose presence in a decision rule counts as using the lever."""
    name = str(lever["name"])
    terms = {name, name.replace("_", " ")}
    description = (lever.get("detail") or {}).get("description")
    if description:
        terms.add(description)
        terms.update(m for m in re.findall(r"\(([^()]*\([^()]*\)[^()]*|[^()]+)\)", description) if len(m) > 3)
    return {t.casefold() for t in terms if t.strip()}


def mentions_lever(sketch, lever):
    rule = str(sketch.get("decision_rule") or "").casefold()
    return any(t in rule for t in lever_terms(lever))


def _admit(task, code, stuck_codes, lever_name):
    try:
        from . import admission
    except ImportError:
        return {"admitted": True, "disagreement": None, "sensitivity": None, "perturbation": None,
                "reasons": ["admission module unavailable: not checked"], "valid": None}
    if hasattr(admission, "supported") and not admission.supported(task):
        # No probes for this task (cache_eviction, every evaluator_cmd): the behavioral gate cannot run.
        # Rejecting here used to drop every candidate on those tasks; the dev score decides instead.
        return {"admitted": True, "disagreement": None, "sensitivity": None, "perturbation": None,
                "reasons": ["no behavior probes for this task: admission not checked"], "valid": None}
    try:
        return admission.admit(task, code, stuck_codes, lever_name)
    except BudgetExceeded:
        raise
    except Exception as e:
        return {"admitted": False, "disagreement": None, "sensitivity": None, "perturbation": None,
                "reasons": [f"admission failed: {sandbox.redact(str(e))[:300]}"], "valid": False}


def escape_v2(session, task, current_solution, previous_attempts, evaluator, overlook=None, families=None, lever=None,
              k_sketches=3, max_resample=2, seed=0, log=None, extract=None, feedback_on_reject=False, notes="",
              lever_policy="chosen", lever_fallback=None, only_on_switch=False):
    """Escape v2: failure summary + one lever + agreement counterexample -> k mechanism sketches
    (fresh context, no code) -> keep sketches that use the lever and differ from the stuck
    strategy -> implement each, admit (resampling up to max_resample times) -> best on dev.
    feedback_on_reject=True appends the admission reasons to each resample prompt (as v3 does);
    the default keeps v2's prompts and cache keys unchanged.
    lever_policy "chosen" (the first lever in preference order) or "random" (random_lever). When no
    lever is found (no overlook study, e.g. a new task): attempt_levers, then lever_fallback(diagnosis,
    report) (a callable returning a lever dict or None), only when the diagnosis is switch_strategy.
    only_on_switch=True generates nothing unless the diagnosis is switch_strategy (diverge_escape)."""
    from .llm import ask_json
    events, calls = [], {"implementer": 0, "extractor": 0}

    def note(event):
        events.append(event)
        if log:
            log(event)

    attempts = [_attempt(a, i) for i, a in enumerate([*previous_attempts, current_solution])]
    score_dir = session.out_dir(task) / "escape_v2" / "attempts"
    for a in attempts:  # scored on dev (sandboxed) when the score or the per-case detail is missing
        if a.get("code") and (a.get("score") is None or not a.get("dev_detail")):
            score_dir.mkdir(parents=True, exist_ok=True)
            path = score_dir / f"{hashlib.sha256(a['code'].encode()).hexdigest()[:12]}.py"
            path.write_text(a["code"])
            result = evaluator(path)
            a.update(score=result["score"] if result["valid"] else None, valid=result["valid"],
                     dev_detail=dev_detail(result))
    current = attempts[-1]
    report = _analyze(task, attempts, evaluator, extract)
    diagnosis = diagnose(task, report, overlook)
    if lever_policy not in LEVER_POLICIES:
        raise ValueError(f"lever_policy must be one of {list(LEVER_POLICIES)}, got {lever_policy!r}")
    alternatives, source = [], "given"
    if lever is None:
        choice = choose(diagnosis, report, overlook, families)
        chosen = choice["lever"] and {**choice["lever"], "mechanism": choice["mechanism"], "why": choice["why"]}
        alternatives, source = choice["alternatives"], "chosen"
        if lever_policy == "random":
            chosen, source = random_lever(report, overlook, levers(diagnosis, report, overlook, families),
                                          attempts, seed), "random"
        if chosen is None and diagnosis["action"] == "switch_strategy":
            found = attempt_levers(report)
            if found:
                chosen, alternatives, source = found[0], found[1:4], "attempts"
            elif lever_fallback is not None:
                chosen, source = lever_fallback(diagnosis, report), "fallback"
    else:
        chosen = _lever_v2(lever, overlook)
    before = current.get("score")
    out = {"version": 2, "lever": chosen, "lever_policy": lever_policy, "lever_source": source if chosen else None,
           "lever_alternatives": alternatives, "failure_summary": None, "counterexample": None, "sketches": [],
           "kept_sketches": [], "rejections": [], "admitted": [], "candidate": None, "candidate_path": None,
           "evaluation": {"before": before, "after": None, "improved": False, "split": "dev"},
           "calls": calls, "diagnosis": diagnosis, "analysis": report, "action": diagnosis["action"], "log": events}
    summary = failure_summary(attempts, diagnosis, evaluator, notes=notes)
    out["failure_summary"] = summary
    if chosen is None or (only_on_switch and diagnosis["action"] != "switch_strategy"):
        note({"stage": "lever", "event": "no lever available" if chosen is None else
              f"no candidate generated: diagnosis is {diagnosis['action']}"})
        out["spent_usd"] = session.ledger.spent_this_run
        return out

    cx, no_cx = _counterexample_v2(session, task, attempts, chosen, overlook, seed)
    out["counterexample"] = cx
    note({"stage": "counterexample", "found": cx is not None, "note": no_cx})

    implementer = session.cfg["models"]["implementer"]
    extractor = session.cfg["models"].get("extractor") or session.cfg["models"].get("labeler") or implementer
    impl_llm, judge_llm = _Counted(session.llm, calls, "implementer"), _Counted(session.llm, calls, "extractor")

    prompt = sketch_prompt(task, summary, chosen, cx, overlook, k=k_sketches, no_cx_note=no_cx)
    try:  # an unusable sketch reply ends this escape without a candidate, not the whole tool call
        reply = ask_json(impl_llm, implementer, prompt, max_tokens=session.cfg["reality_check"]["max_tokens"],
                         key=f"escape_v2:sketch:{seed}", temperature=1.0, required=("sketches",))
    except BudgetExceeded:
        raise
    except Exception as e:
        out["rejections"].append({"stage": "sketch", "sketch": None, "reasons": [sandbox.redact(str(e))[:300]]})
        note(out["rejections"][-1])
        reply = {}
    sketches = [s for s in reply.get("sketches") or [] if isinstance(s, dict)][:k_sketches]
    out["sketches"], out["sketch_prompt"] = sketches, prompt

    kept = []
    for i, s in enumerate(sketches):
        # The literal check misses paraphrases ("a new bin" for new_bin_slot) and family levers,
        # whose names never appear in a rule; the extractor model judges those cases.
        try:  # one unparsable judge reply drops this sketch, not the whole run
            uses = mentions_lever(s, chosen) or ask_json(judge_llm, extractor, USES_TEMPLATE.format(
            lever=chosen["name"], description=(f" ({(chosen.get('detail') or {}).get('description')})"
                                               if (chosen.get("detail") or {}).get("description") else ""),
            name=s.get("name", ""), state=s.get("state", ""), decision_rule=s.get("decision_rule", ""),
            uses_lever=s.get("uses_lever", "")),
            max_tokens=2000, key=f"escape_v2:uses:{seed}:{i}", required=("uses_lever",)).get("uses_lever") is True
            verdict = ask_json(judge_llm, extractor, JUDGE_TEMPLATE.format(
                shared=diagnosis["shared_cause"], name=s.get("name", ""), state=s.get("state", ""),
                decision_rule=s.get("decision_rule", ""), lever=chosen["name"], uses_lever=s.get("uses_lever", "")),
                max_tokens=2000, key=f"escape_v2:same:{seed}:{i}", required=("same_mechanism",)) if uses else {}
        except BudgetExceeded:
            raise
        except Exception as e:
            out["rejections"].append({"stage": "sketch_filter", "sketch": s.get("name"),
                                      "reasons": [f"judge failed: {type(e).__name__}: {str(e)[:200]}"]})
            note(out["rejections"][-1])
            continue
        if not uses:
            reason = f"decision rule does not use {chosen['name']}"
            out["rejections"].append({"stage": "sketch_filter", "sketch": s.get("name"), "reasons": [reason]})
            note(out["rejections"][-1])
            continue
        if verdict.get("same_mechanism") is not False:
            reason = f"same mechanism as the shared strategy: {verdict.get('reason') or 'judged the same'}"
            out["rejections"].append({"stage": "sketch_filter", "sketch": s.get("name"), "reasons": [reason]})
            note(out["rejections"][-1])
            continue
        kept.append({**s, "index": i, "difference": verdict.get("reason")})
    out["kept_sketches"] = kept

    code_dir = session.out_dir(task) / "escape_v2"
    code_dir.mkdir(parents=True, exist_ok=True)
    stuck_codes = [a["code"] for a in attempts if a.get("code")]
    prompts = []
    for s in kept:
        impl_prompt = implementation_prompt_v2(task, s, chosen, overlook)
        prompts.append(impl_prompt)
        feedback = ""  # opt-in: why the previous candidate was rejected, so a resample is not blind
        for attempt in range(max_resample + 1):
            base = {"stage": "admission", "sketch": s.get("name"), "attempt": attempt}
            prompt = impl_prompt + feedback
            if feedback:
                prompts.append(prompt)
            try:
                entry = impl_llm.chat(implementer, [{"role": "user", "content": prompt}], temperature=1.0,
                                      max_tokens=session.cfg["reality_check"]["max_tokens"],
                                      sample_idx=f"escape_v2:impl:{seed}:{s['index']}:{attempt}")
            except BudgetExceeded:
                raise
            except Exception as e:
                out["rejections"].append({**base, "stage": "implement", "reasons": [sandbox.redact(str(e))[:300]]})
                note(out["rejections"][-1])
                continue
            code = extract_code(message_text(entry), task.entrypoint)
            if code is None:
                out["rejections"].append({**base, "stage": "implement", "reasons": ["no code in reply"]})
                note(out["rejections"][-1])
                continue
            verdict = _admit(task, code, stuck_codes, chosen["name"])
            if not verdict.get("admitted"):
                out["rejections"].append({**base, "reasons": list(verdict.get("reasons") or []),
                                          "disagreement": verdict.get("disagreement"),
                                          "sensitivity": verdict.get("sensitivity"),
                                          "perturbation": verdict.get("perturbation")})
                note(out["rejections"][-1])
                if feedback_on_reject:
                    feedback = ADMISSION_FEEDBACK.format(reasons="\n".join(
                        f"- {r}" for r in (verdict.get("reasons") or ["not admitted"])), lever=chosen["name"])
                continue
            path = code_dir / f"{slug(chosen['name'])}-{seed}-{s['index']}-{attempt}.py"
            path.write_text(code)
            try:
                result = evaluator(path)  # dev split only
            except BudgetExceeded:
                raise
            except Exception as e:
                out["rejections"].append({"stage": "evaluate", "sketch": s.get("name"), "attempt": attempt,
                                          "reasons": [f"{type(e).__name__}: {str(e)[:200]}"]})
                note(out["rejections"][-1])
                break
            error = (result.get("detail") or {}).get("error")
            admitted = {"sketch": s.get("name"), "attempt": attempt, "path": str(path), "valid": result["valid"],
                        "score": result["score"] if result["valid"] else None,
                        "disagreement": verdict.get("disagreement"), "sensitivity": verdict.get("sensitivity"),
                        "error": sandbox.redact(str(error))[:500] if error else None, "code": code}
            out["admitted"].append(admitted)
            note({"stage": "admitted", **{k: v for k, v in admitted.items() if k != "code"}})
            break
    out["implementation_prompts"] = prompts

    valid = [c for c in out["admitted"] if c["valid"] and c["score"] is not None]
    if valid:
        top = evaluator.best([c["score"] for c in valid])
        best = next(c for c in valid if c["score"] == top)
        after = best["score"]
        out.update(candidate=best["code"], candidate_path=best["path"])
        out["evaluation"] = {"before": before, "after": after,
                             "improved": before is None or evaluator.beats(after, before), "split": "dev"}
    out["admitted"] = [{k: v for k, v in c.items() if k != "code"} for c in out["admitted"]]
    out["spent_usd"] = session.ledger.spent_this_run
    return out


# ================================================================ escape v3
# Diverge -> critic -> deepen. The failure summary is first stripped of incidental details of the
# stuck attempts (anchors) while hard task constraints are kept only with a verbatim quote from the
# task. Each sketch is proposed in its own isolated call on its own lever (no other sketch, no
# scores, no code visible); a separate critic model scores them, trap / restated sketches are
# dropped, and only the top few are deepened, implemented (fresh context), admitted and dev-scored.
ANCHOR_MARKER = "Remove anchoring details"
ANCHOR_TEMPLATE = ANCHOR_MARKER + """ from a summary of an agent's stuck attempts at the task below.

Task statement:
<<<
{task}
>>>

Summary of the stuck attempts (dev split):
<<<
{summary}
>>>

Rewrite the summary so that a fresh designer is not anchored on how the attempts were built: remove
parameter values, constants, data structures, variable or function names, and the names of the
attempts' particular tricks. Keep scores, which dev instances are hardest, and what kind of failure
was observed, in neutral words.
Separately list the hard constraints of the task that any solution must respect. For each one,
copy the exact sentence fragment from the task statement that imposes it (verbatim, no paraphrase);
leave out anything the task statement does not literally require.
Reply with only a JSON object:
{{"summary": "...", "constraints": [{{"constraint": "...", "quote": "exact text from the task"}}]}}"""

SKETCH3_MARKER = "Propose one mechanism"
SKETCH3_TEMPLATE = """{task}

---
""" + SKETCH3_MARKER + """ for a fresh solution. A previous line of work got stuck; what is known
about it is summarised below (its code is deliberately not shown):
{summary}

Hard requirements of the task (quoted from it):
{constraints}

Build your mechanism on this one starting point: {lever} ({description}).
Why this starting point: {why}

Describe a single mechanism in plain language (no code) whose decisions depend on {lever}, so that
changing {lever} changes which option is chosen. Aim for a genuinely different way of deciding, not
a re-weighting of a familiar rule. Reply with only a JSON object:
{{"name": "...", "state": "what is kept across calls, or none", "decision_rule": "how each decision is made",
"uses_lever": "how the decision depends on {lever}"}}"""

CRITIC_MARKER = "Critique candidate mechanisms"
CRITIC_TEMPLATE = CRITIC_MARKER + """ for the task below. You are a reviewer, not their author.

Task statement:
<<<
{task}
>>>

The strategy a previous line of work got stuck on: {stuck}

Candidate mechanisms:
{sketches}

For each candidate give integer scores 0-10: "viability" (can it be implemented correctly within the
task's interface and rules), "fit" (does it exploit structure this task actually has), "distance" (how
different its decision signal and rule are from the stuck strategy). Name one concrete "strength" and
one concrete "trap" (a specific way it would fail or quietly degrade). Set "is_trap" true if the trap
is likely to make it no better than the stuck strategy or invalid, and "restates_stuck" true if it is
the stuck strategy re-weighted, re-scaled, re-tuned or renamed. Reply with only a JSON object:
{{"scores": [{{"index": 0, "viability": 0, "fit": 0, "distance": 0, "strength": "...", "trap": "...",
"is_trap": false, "restates_stuck": false}}]}}"""

DEEPEN_MARKER = "Work out this mechanism"
DEEPEN_TEMPLATE = """{task}

---
""" + DEEPEN_MARKER + """ before it is implemented (no earlier code is shown on purpose):
Name: {name}
State kept across calls: {state}
Decision rule: {decision_rule}
How the decision depends on {lever}: {uses_lever}
A reviewer's concern: {trap}

Hard requirements of the task (quoted from it):
{constraints}

Make the mechanism precise: what exactly is computed at each decision and from which inputs, the
single biggest risk to its score or validity and how the design avoids it, and the first concrete
step an implementation should take. Reply with only a JSON object:
{{"mechanism": "...", "main_risk": "...", "first_step": "..."}}"""

IMPLEMENT3_MARKER = "Implement this worked-out mechanism"
ADMISSION_FEEDBACK = """

---
Your previous implementation of this mechanism was rejected before scoring, because a behavioral
check on fixed decision states found:
{reasons}
Write a new implementation in which the decisions really follow the mechanism above and really
depend on {lever}, so that they differ from the stuck attempts' decisions."""

IMPLEMENT3_TEMPLATE = """{task}

---
""" + IMPLEMENT3_MARKER + """ (a fresh design; no earlier code is shown on purpose):
Name: {name}
State kept across calls: {state}
Mechanism: {mechanism}
Main risk to avoid: {main_risk}
First step: {first_step}

The decisions must materially depend on {lever} ({description}): changing it must change which
option is chosen.

Hard requirements of the task (quoted from it):
{constraints}

Write a complete solution that implements this mechanism faithfully, keeping the task's interface
and rules. Reply with your complete solution in a single Python code block."""

CRITIC_AXES = ("viability", "fit", "distance")


def _norm(text):
    return " ".join(str(text).split())


def anchor_strip(llm, model, task, summary, seed=0, max_tokens=4000):
    """(stripped summary, kept constraints, dropped constraints). Constraints are kept only when
    their quote occurs verbatim (whitespace-normalised) in the task statement. Raises on failure."""
    from .llm import ask_json
    reply = ask_json(llm, model, ANCHOR_TEMPLATE.format(task=task.prompt, summary=summary), max_tokens=max_tokens,
                     key=f"escape_v3:anchor:{seed}", required=("summary", "constraints"))
    stripped = reply.get("summary")
    if not isinstance(stripped, str) or not stripped.strip():
        raise ValueError("anchor strip returned no summary")
    prompt, kept, dropped = _norm(task.prompt), [], []
    for c in reply.get("constraints") or []:
        if not isinstance(c, dict):
            continue
        quote = _norm(c.get("quote") or "")
        (kept if quote and quote in prompt else dropped).append({"constraint": str(c.get("constraint") or ""),
                                                                 "quote": quote})
    return stripped.strip(), kept, dropped


def lever_pool(diagnosis, trajectory, overlook, families=None):
    """Levers for v3 sketches, in order: levers() (overlooked unused inputs, assumptions,
    minority and blind-spot families), then other inputs the attempts never use, then rarely
    used overlook inputs (share <= 0.5), rarest first."""
    pool = [dict(lv) for lv in levers(diagnosis, trajectory, overlook, families)]
    names = {str(lv["name"]) for lv in pool}
    unused = unused_inputs(trajectory, overlook)
    rows = {r["input"]: r for r in (overlook or {}).get("inputs", []) if r.get("input")}
    share = lambda n: rows.get(n, {}).get("share") if rows.get(n, {}).get("share") is not None else 1.0
    rare = [n for n in rows if n not in unused and share(n) <= 0.5]
    for name in sorted(unused, key=lambda n: (share(n), n)) + sorted(rare, key=lambda n: (share(n), n)):
        if name not in names:
            names.add(name)
            pool.append(_lever_v2(name, overlook))
    return pool


def _constraint_lines(kept, overlook):
    quotes = list(dict.fromkeys([c["quote"] for c in kept] + hard_constraints(overlook)))
    return "\n".join(f'- "{q}"' for q in quotes) or "- (none recorded beyond the interface and rules above)"


def sketch_prompt_v3(task, summary, constraints, lever):
    description, _ = _lever_text(lever)
    return SKETCH3_TEMPLATE.format(task=task.prompt, summary=summary, constraints=constraints, lever=lever["name"],
                                   description=description, why=lever.get("why") or "")


def critic_prompt(task, stuck, sketches):
    lines = "\n".join(f"[{i}] {s.get('name', '')}\n    state: {s.get('state', '')}\n    decision rule: "
                      f"{s.get('decision_rule', '')}\n    uses {s['lever']}: {s.get('uses_lever', '')}"
                      for i, s in enumerate(sketches))
    return CRITIC_TEMPLATE.format(task=task.prompt, stuck=stuck, sketches=lines)


def _score(v):
    try:
        return max(0.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _critic_model(models):
    impl = models["implementer"]
    for m in (models.get("labeler"), models.get("extractor")):
        if m and m != impl:
            return m
    return models.get("labeler") or models.get("extractor") or impl


def escape_v3(session, task, current_solution, previous_attempts, evaluator, overlook=None, families=None, lever=None,
              n_sketches=5, n_deepen=2, max_resample=2, seed=0, log=None, extract=None, feedback_on_reject=True,
              notes=""):
    """Escape v3: anchor-stripped failure summary -> n_sketches isolated sketches, each on its own
    lever -> critic (a different model) drops traps / restatements and ranks -> deepen the top
    n_deepen -> implement (fresh context), admit (up to max_resample resamples) -> best on dev.
    Per-branch failures are logged and skipped."""
    from .llm import ask_json
    events = []
    calls = {"implementer": 0, "extractor": 0, "critic": 0}

    def note(event):
        events.append(event)
        if log:
            log(event)

    def reject(stage, sketch, reason, **extra):
        out["rejections"].append({"stage": stage, "sketch": sketch, "reasons": [sandbox.redact(str(reason))[:300]],
                                  **extra})
        note(out["rejections"][-1])

    attempts = [_attempt(a, i) for i, a in enumerate([*previous_attempts, current_solution])]
    score_dir = session.out_dir(task) / "escape_v3" / "attempts"
    for a in attempts:  # scored on dev (sandboxed) when the score or the per-case detail is missing
        if a.get("code") and (a.get("score") is None or not a.get("dev_detail")):
            score_dir.mkdir(parents=True, exist_ok=True)
            path = score_dir / f"{hashlib.sha256(a['code'].encode()).hexdigest()[:12]}.py"
            path.write_text(a["code"])
            result = evaluator(path)
            a.update(score=result["score"] if result["valid"] else None, valid=result["valid"],
                     dev_detail=dev_detail(result))
    current = attempts[-1]
    report = _analyze(task, attempts, evaluator, extract)
    diagnosis = diagnose(task, report, overlook)
    pool = [_lever_v2(lever, overlook)] if lever is not None else lever_pool(diagnosis, report, overlook, families)
    before = current.get("score")
    models = session.cfg["models"]
    implementer = models["implementer"]
    extractor = models.get("extractor") or models.get("labeler") or implementer
    critic = _critic_model(models)
    out = {"version": 3, "lever": pool[0] if pool else None, "levers": pool, "failure_summary": None,
           "anchor_strip": {"constraints": [], "dropped": [], "fallback": False, "model": extractor},
           "sketches": [], "critic": {"model": critic, "scores": []}, "kept_sketches": [], "deepened": [],
           "rejections": [], "admitted": [], "candidate": None, "candidate_path": None,
           "evaluation": {"before": before, "after": None, "improved": False, "split": "dev"},
           "calls": calls, "diagnosis": diagnosis, "analysis": report, "action": diagnosis["action"], "log": events}
    if not pool:
        out["failure_summary"] = failure_summary(attempts, diagnosis, evaluator, notes=notes)
        note({"stage": "lever", "event": "no lever available"})
        out["spent_usd"] = session.ledger.spent_this_run
        return out

    impl_llm = _Counted(session.llm, calls, "implementer")
    ext_llm, critic_llm = _Counted(session.llm, calls, "extractor"), _Counted(session.llm, calls, "critic")
    max_tokens = session.cfg["reality_check"]["max_tokens"]

    # a. anchor strip
    summary_v2 = failure_summary(attempts, diagnosis, evaluator, notes=notes)
    try:
        summary, kept_c, dropped_c = anchor_strip(ext_llm, extractor, task, summary_v2, seed=seed)
        out["anchor_strip"].update(constraints=kept_c, dropped=dropped_c, original=summary_v2)
        note({"stage": "anchor_strip", "kept": len(kept_c), "dropped": len(dropped_c)})
    except BudgetExceeded:
        raise
    except Exception as e:
        summary, kept_c = summary_v2, []
        out["anchor_strip"].update(fallback=True, error=sandbox.redact(str(e))[:300])
        note({"stage": "anchor_strip", "event": "fallback to v2 summary", "error": out["anchor_strip"]["error"]})
    out["failure_summary"] = summary
    constraints = _constraint_lines(kept_c, overlook)

    # b. diverge: isolated sketches, one lever each
    prompts = []
    for i in range(n_sketches):
        lv = pool[i % len(pool)]
        prompt = sketch_prompt_v3(task, summary, constraints, lv)
        prompts.append(prompt)
        try:
            s = ask_json(impl_llm, implementer, prompt, max_tokens=max_tokens, key=f"escape_v3:sketch:{seed}:{i}",
                         temperature=1.0, required=("name", "decision_rule"))
        except BudgetExceeded:
            raise
        except Exception as e:
            reject("sketch", None, e, index=i, lever=lv["name"])
            continue
        out["sketches"].append({k: str(s.get(k) or "") for k in ("name", "state", "decision_rule", "uses_lever")}
                               | {"index": i, "lever": lv["name"]})
    out["sketch_prompts"] = prompts

    # c. critic: one call over all sketches, a different model when available
    sketches = out["sketches"]
    ranked = []
    if sketches:
        try:
            reply = ask_json(critic_llm, critic, critic_prompt(task, diagnosis["shared_cause"], sketches),
                             max_tokens=max_tokens, key=f"escape_v3:critic:{seed}", required=("scores",))
            by_pos = {}
            for r in reply.get("scores") or []:
                if isinstance(r, dict) and isinstance(r.get("index"), int) and 0 <= r["index"] < len(sketches):
                    by_pos[r["index"]] = r
            for pos, s in enumerate(sketches):
                r = by_pos.get(pos)
                if r is None:
                    reject("critic", s["name"], "critic gave no score", index=s["index"])
                    continue
                row = {"index": s["index"], "sketch": s["name"], **{k: _score(r.get(k)) for k in CRITIC_AXES},
                       "strength": str(r.get("strength") or ""), "trap": str(r.get("trap") or ""),
                       "is_trap": r.get("is_trap") is True, "restates_stuck": r.get("restates_stuck") is True}
                row["total"] = sum(row[k] for k in CRITIC_AXES)
                out["critic"]["scores"].append(row)
                if row["is_trap"] or row["restates_stuck"]:
                    why = "restates the stuck strategy" if row["restates_stuck"] else f"trap: {row['trap']}"
                    reject("critic", s["name"], why, index=s["index"])
                    continue
                ranked.append((row["total"], -pos, {**s, "critic": row}))
        except BudgetExceeded:
            raise
        except Exception as e:  # no critique: keep every sketch, in proposal order
            out["critic"]["error"] = sandbox.redact(str(e))[:300]
            note({"stage": "critic", "event": "critic failed; sketches kept unranked", "error": out["critic"]["error"]})
            ranked = [(0.0, -pos, {**s, "critic": None}) for pos, s in enumerate(sketches)]
    kept = [s for _, _, s in sorted(ranked, key=lambda t: (t[0], t[1]), reverse=True)]
    out["kept_sketches"] = kept

    # d. deepen the top n_deepen, implement, admit, score on dev
    levers_by_name = {str(lv["name"]): lv for lv in pool}
    code_dir = session.out_dir(task) / "escape_v3"
    code_dir.mkdir(parents=True, exist_ok=True)
    stuck_codes = [a["code"] for a in attempts if a.get("code")]
    impl_prompts = []
    for s in kept[:n_deepen]:
        lv = levers_by_name[s["lever"]]
        description, _ = _lever_text(lv)
        try:
            d = ask_json(impl_llm, implementer, DEEPEN_TEMPLATE.format(
                task=task.prompt, name=s["name"], state=s["state"] or "none", decision_rule=s["decision_rule"],
                lever=lv["name"], uses_lever=s["uses_lever"], trap=((s.get("critic") or {}).get("trap") or "none given"),
                constraints=constraints), max_tokens=max_tokens, key=f"escape_v3:deepen:{seed}:{s['index']}",
                required=("mechanism", "main_risk", "first_step"))
        except BudgetExceeded:
            raise
        except Exception as e:
            reject("deepen", s["name"], e, index=s["index"])
            continue
        deep = {"sketch": s["name"], "index": s["index"], "lever": lv["name"],
                **{k: str(d.get(k) or "") for k in ("mechanism", "main_risk", "first_step")}}
        out["deepened"].append(deep)
        note({"stage": "deepened", "sketch": s["name"], "lever": lv["name"]})
        impl_prompt = IMPLEMENT3_TEMPLATE.format(task=task.prompt, name=s["name"], state=s["state"] or "none",
                                                 mechanism=deep["mechanism"], main_risk=deep["main_risk"],
                                                 first_step=deep["first_step"], lever=lv["name"],
                                                 description=description, constraints=constraints)
        impl_prompts.append(impl_prompt)
        feedback = ""  # P2: why the previous candidate was rejected, so a resample is not blind
        for attempt in range(max_resample + 1):
            base = {"sketch": s["name"], "index": s["index"], "attempt": attempt}
            prompt = impl_prompt + feedback
            if feedback:
                impl_prompts.append(prompt)
            try:
                entry = impl_llm.chat(implementer, [{"role": "user", "content": prompt}], temperature=1.0,
                                      max_tokens=max_tokens, sample_idx=f"escape_v3:impl:{seed}:{s['index']}:{attempt}")
                code = extract_code(message_text(entry), task.entrypoint)
                if code is None:
                    raise ValueError("no code in reply")
            except BudgetExceeded:
                raise
            except Exception as e:
                reject("implement", s["name"], e, index=s["index"], attempt=attempt)
                continue
            verdict = _admit(task, code, stuck_codes, lv["name"])
            if not verdict.get("admitted"):
                out["rejections"].append({"stage": "admission", **base, "reasons": list(verdict.get("reasons") or []),
                                          "disagreement": verdict.get("disagreement"),
                                          "sensitivity": verdict.get("sensitivity"),
                                          "perturbation": verdict.get("perturbation")})
                note(out["rejections"][-1])
                if feedback_on_reject:
                    feedback = ADMISSION_FEEDBACK.format(reasons="\n".join(
                        f"- {r}" for r in (verdict.get("reasons") or ["not admitted"])), lever=lv["name"])
                continue
            path = code_dir / f"{slug(str(lv['name']))}-{seed}-{s['index']}-{attempt}.py"
            path.write_text(code)
            try:
                result = evaluator(path)  # dev split only
            except BudgetExceeded:
                raise
            except Exception as e:
                reject("evaluate", s["name"], e, index=s["index"], attempt=attempt)
                break
            error = (result.get("detail") or {}).get("error")
            admitted = {**base, "lever": lv["name"], "path": str(path), "valid": result["valid"],
                        "score": result["score"] if result["valid"] else None,
                        "disagreement": verdict.get("disagreement"), "sensitivity": verdict.get("sensitivity"),
                        "error": sandbox.redact(str(error))[:500] if error else None, "code": code}
            out["admitted"].append(admitted)
            note({"stage": "admitted", **{k: v for k, v in admitted.items() if k != "code"}})
            break
    out["implementation_prompts"] = impl_prompts

    valid = [c for c in out["admitted"] if c["valid"] and c["score"] is not None]
    if valid:
        top = evaluator.best([c["score"] for c in valid])
        best = next(c for c in valid if c["score"] == top)
        out.update(candidate=best["code"], candidate_path=best["path"], lever=levers_by_name[best["lever"]])
        out["evaluation"] = {"before": before, "after": best["score"],
                             "improved": before is None or evaluator.beats(best["score"], before), "split": "dev"}
    out["admitted"] = [{k: v for k, v in c.items() if k != "code"} for c in out["admitted"]]
    out["spent_usd"] = session.ledger.spent_this_run
    return out


MODES = {"v1": escape, "v2": escape_v2, "v3": escape_v3}
DEFAULT_MODE = "v2"


def run_escape(mode=DEFAULT_MODE, *args, **kwargs):
    """Run escape version `mode` ("v1", "v2" or "v3"); keyword arguments the chosen version does
    not take (e.g. lever/seed for v1, n_sketches for v2) are dropped."""
    fn = MODES[mode]
    params = inspect.signature(fn).parameters
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


# ---------------------------------------------------------------------------------------------
# Escape v4: v3 with (1) an implementation prompt that keeps the lever's mechanism load-bearing
# but allows complementary sibling mechanisms, (2) a critic rubric applied as hard filters,
# (3) a compile + entrypoint check with exactly one minimal repair per candidate, and (4) one
# evolve step: the top two dev-scored candidates each get one mutated child. Dev split only.
CRITIC4_MARKER = "Review candidate mechanisms against a rubric"
CRITIC4_TEMPLATE = CRITIC4_MARKER + """ for the task below. You review them; you did not write them.

Task statement:
<<<
{task}
>>>

The strategy a previous line of work got stuck on: {stuck}

Candidate mechanisms (each was asked to build on a named target lever):
{sketches}

For every candidate fill in this rubric:
- "uses_target_mechanism": true only if its decisions really depend on its target lever (changing the
  lever would change which option it picks); false if the lever is decoration or it falls back to the
  stuck strategy.
- "strategy_distinctness" 0-10: how different its decision signal and rule are from the stuck strategy.
- "feasibility" 0-10: how likely a careful programmer implements it correctly within the task's
  interface, rules and time limits.
- "constraint_fit" 0-10: how well it respects the task's hard requirements.
- "trap": the most likely concrete way it fails or quietly degrades, or null if you see none.
- "fatal_trap": true if that trap would make it invalid or no better than the stuck strategy.
- "strength": one concrete reason it could beat the stuck strategy.
Reply with only a JSON object:
{{"scores": [{{"index": 0, "uses_target_mechanism": true, "strategy_distinctness": 0, "feasibility": 0,
"constraint_fit": 0, "trap": null, "fatal_trap": false, "strength": "..."}}]}}"""

RUBRIC_AXES = ("strategy_distinctness", "feasibility", "constraint_fit")
MIN_FEASIBILITY, MIN_CONSTRAINT_FIT = 4, 5

IMPLEMENT4_MARKER = "Implement a solution whose core is this mechanism"
IMPLEMENT4_TEMPLATE = """{task}

---
""" + IMPLEMENT4_MARKER + """ (a fresh design; no earlier code is shown on purpose):
Name: {name}
State kept across calls: {state}
Mechanism: {mechanism}
Main risk to avoid: {main_risk}
First step: {first_step}
Known risks from review: {risks}

This mechanism is load-bearing: the solution's decisions must be driven by it and must materially
depend on {lever} ({description}), so that changing {lever} changes which option is chosen.
You may combine it with complementary mechanisms when that helps correctness or performance, for
example ideas from these sibling designs:
{siblings}
Helpers like these may support, refine or break ties for the core mechanism, but must not replace
it. Do not fall back to the conventional strategy the previous work got stuck on: {stuck}.

Hard requirements of the task (quoted from it):
{constraints}

Write a complete solution, keeping the task's interface and rules. Reply with your complete
solution in a single Python code block."""

REPAIR_MARKER = "Fix this code with the smallest possible change"
REPAIR_TEMPLATE = REPAIR_MARKER + """ so that it compiles and defines `{entrypoint}` at the top
level of the module.

Problem found:
{error}

Code:
```python
{code}
```

Rules: change only what is needed to fix the problem above. Do not redesign it, do not change its
strategy or its decision rule, do not rename things that already work. Reply with the complete fixed
code in a single Python code block."""

EVOLVE_MARKER = "Improve this candidate once"
EVOLVE_TEMPLATE = """{task}

---
""" + EVOLVE_MARKER + """. It was built on the target lever {lever} ({description}) and scored on the
dev split ({direction} is better).

Its dev score: {score}
Its per-case dev results (worst first):
{cases}
Its worst dev cases: {worst}

Its code:
```python
{code}
```

Write one improved version. Rules:
- keep the mechanism built on {lever} central: decisions must still materially depend on it;
- you may fix defects, combine it with compatible mechanisms, change how state is represented, or
  change how the signal from {lever} is used;
- you may not revert to the conventional strategy the previous work got stuck on: {stuck}.
Hard requirements of the task (quoted from it):
{constraints}

Reply with your complete solution in a single Python code block."""


def syntax_check(code, entrypoint):
    """None if `code` compiles and defines `entrypoint` (def, class or assignment) at the top
    level; otherwise a short description of the problem."""
    import ast
    try:
        compile(code, "<candidate>", "exec")
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    except Exception as e:  # e.g. ValueError for null bytes
        return f"{type(e).__name__}: {e}"
    if not entrypoint:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == entrypoint:
            return None
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == entrypoint for t in targets):
                return None
    return f"`{entrypoint}` is not defined at the top level of the module"


def _code_from(entry, entrypoint):
    text = message_text(entry)
    return extract_code(text, entrypoint) or extract_code(text, None)


def critic_prompt_v4(task, stuck, sketches):
    lines = "\n".join(f"[{i}] {s.get('name', '')}\n    target lever: {s['lever']}\n    state: {s.get('state', '')}"
                      f"\n    decision rule: {s.get('decision_rule', '')}\n    uses {s['lever']}: {s.get('uses_lever', '')}"
                      for i, s in enumerate(sketches))
    return CRITIC4_TEMPLATE.format(task=task.prompt, stuck=stuck, sketches=lines)


def rubric_row(r):
    """Normalised rubric row and the reason it fails the hard filters (None if it passes)."""
    trap = r.get("trap")
    trap = str(trap).strip() if trap not in (None, "", False) and str(trap).strip().lower() not in ("null", "none") else None
    row = {"uses_target_mechanism": r.get("uses_target_mechanism") is True,
           **{k: _score(r.get(k)) for k in RUBRIC_AXES},
           "trap": trap, "fatal_trap": bool(trap) and r.get("fatal_trap") is True,
           "strength": str(r.get("strength") or "")}
    row["total"] = sum(row[k] for k in RUBRIC_AXES)
    if not row["uses_target_mechanism"]:
        why = "does not use the target mechanism"
    elif row["feasibility"] < MIN_FEASIBILITY:
        why = f"feasibility {row['feasibility']:g} < {MIN_FEASIBILITY}"
    elif row["constraint_fit"] < MIN_CONSTRAINT_FIT:
        why = f"constraint_fit {row['constraint_fit']:g} < {MIN_CONSTRAINT_FIT}"
    elif row["fatal_trap"]:
        why = f"fatal trap: {trap}"
    else:
        why = None
    return row, why


def _sibling_lines(kept, s, limit=4):
    sibs = [k for k in kept if k["index"] != s["index"]][:limit]
    return "\n".join(f"- {k['name']}: {k['decision_rule']}" for k in sibs) or "- (none available)"


def _case_lines(detail, limit=12):
    from .failures import _cases
    cases = sorted(_cases(detail), key=lambda c: (-c[1], c[0]))
    if not cases:
        return "- (no per-case detail available)"
    return "\n".join(f"- {case}: {value:.4g}" for case, value in cases[:limit])


def escape_v4(session, task, current_solution, previous_attempts, evaluator, overlook=None, families=None, lever=None,
              n_sketches=5, n_deepen=2, max_resample=2, seed=0, log=None, extract=None, feedback_on_reject=True,
              n_parents=2, notes=""):
    """Escape v4: v3's anchor strip and isolated sketches -> rubric critic with hard filters ->
    deepen the top n_deepen (non-fatal traps passed on as risks) -> implement with the lever's
    mechanism load-bearing and sibling mechanisms allowed -> compile/entrypoint check with at
    most one minimal repair -> admit (up to max_resample resamples) -> dev score -> one evolve
    step (one child each for the top n_parents) -> best on dev among parents and children.
    Per-branch failures are logged and skipped."""
    from .failures import _cases, failure_set
    from .llm import ask_json
    events = []
    calls = {"implementer": 0, "extractor": 0, "critic": 0, "repair": 0}

    def note(event):
        events.append(event)
        if log:
            log(event)

    def reject(stage, sketch, reason, **extra):
        out["rejections"].append({"stage": stage, "sketch": sketch, "reasons": [sandbox.redact(str(reason))[:300]],
                                  **extra})
        note(out["rejections"][-1])

    attempts = [_attempt(a, i) for i, a in enumerate([*previous_attempts, current_solution])]
    score_dir = session.out_dir(task) / "escape_v4" / "attempts"
    for a in attempts:
        if a.get("code") and (a.get("score") is None or not a.get("dev_detail")):
            score_dir.mkdir(parents=True, exist_ok=True)
            path = score_dir / f"{hashlib.sha256(a['code'].encode()).hexdigest()[:12]}.py"
            path.write_text(a["code"])
            result = evaluator(path)
            a.update(score=result["score"] if result["valid"] else None, valid=result["valid"],
                     dev_detail=dev_detail(result))
    current = attempts[-1]
    report = _analyze(task, attempts, evaluator, extract)
    diagnosis = diagnose(task, report, overlook)
    stuck = diagnosis["shared_cause"]
    pool = [_lever_v2(lever, overlook)] if lever is not None else lever_pool(diagnosis, report, overlook, families)
    before = current.get("score")
    models = session.cfg["models"]
    implementer = models["implementer"]
    extractor = models.get("extractor") or models.get("labeler") or implementer
    critic = _critic_model(models)
    out = {"version": 4, "lever": pool[0] if pool else None, "levers": pool, "failure_summary": None,
           "anchor_strip": {"constraints": [], "dropped": [], "fallback": False, "model": extractor},
           "sketches": [], "critic": {"model": critic, "scores": [], "rubric": []}, "kept_sketches": [],
           "deepened": [], "rejections": [], "admitted": [], "repairs": [],
           "evolve": {"parents": [], "children": []}, "candidate": None, "candidate_path": None,
           "evaluation": {"before": before, "after": None, "improved": False, "split": "dev"},
           "calls": calls, "diagnosis": diagnosis, "analysis": report, "action": diagnosis["action"], "log": events}
    out["critic.rubric"] = out["critic"]["rubric"]
    if not pool:
        out["failure_summary"] = failure_summary(attempts, diagnosis, evaluator, notes=notes)
        note({"stage": "lever", "event": "no lever available"})
        out["spent_usd"] = session.ledger.spent_this_run
        return out

    impl_llm = _Counted(session.llm, calls, "implementer")
    ext_llm, critic_llm = _Counted(session.llm, calls, "extractor"), _Counted(session.llm, calls, "critic")
    repair_llm = _Counted(session.llm, calls, "repair")
    max_tokens = session.cfg["reality_check"]["max_tokens"]

    # a. anchor strip (as v3)
    summary_v2 = failure_summary(attempts, diagnosis, evaluator, notes=notes)
    try:
        summary, kept_c, dropped_c = anchor_strip(ext_llm, extractor, task, summary_v2, seed=seed)
        out["anchor_strip"].update(constraints=kept_c, dropped=dropped_c, original=summary_v2)
        note({"stage": "anchor_strip", "kept": len(kept_c), "dropped": len(dropped_c)})
    except BudgetExceeded:
        raise
    except Exception as e:
        summary, kept_c = summary_v2, []
        out["anchor_strip"].update(fallback=True, error=sandbox.redact(str(e))[:300])
        note({"stage": "anchor_strip", "event": "fallback to v2 summary", "error": out["anchor_strip"]["error"]})
    out["failure_summary"] = summary
    constraints = _constraint_lines(kept_c, overlook)

    # b. isolated sketches, one lever each (as v3)
    prompts = []
    for i in range(n_sketches):
        lv = pool[i % len(pool)]
        prompt = sketch_prompt_v3(task, summary, constraints, lv)
        prompts.append(prompt)
        try:
            s = ask_json(impl_llm, implementer, prompt, max_tokens=max_tokens, key=f"escape_v4:sketch:{seed}:{i}",
                         temperature=1.0, required=("name", "decision_rule"))
        except BudgetExceeded:
            raise
        except Exception as e:
            reject("sketch", None, e, index=i, lever=lv["name"])
            continue
        out["sketches"].append({k: str(s.get(k) or "") for k in ("name", "state", "decision_rule", "uses_lever")}
                               | {"index": i, "lever": lv["name"]})
    out["sketch_prompts"] = prompts

    # c. rubric critic, hard filters, rank by distinctness + feasibility + constraint_fit
    sketches = out["sketches"]
    ranked = []
    if sketches:
        try:
            reply = ask_json(critic_llm, critic, critic_prompt_v4(task, stuck, sketches), max_tokens=max_tokens,
                             key=f"escape_v4:critic:{seed}", required=("scores",))
            by_pos = {}
            for r in reply.get("scores") or []:
                if isinstance(r, dict) and isinstance(r.get("index"), int) and 0 <= r["index"] < len(sketches):
                    by_pos[r["index"]] = r
            for pos, s in enumerate(sketches):
                r = by_pos.get(pos)
                if r is None:
                    reject("critic", s["name"], "critic gave no score", index=s["index"])
                    continue
                row, why = rubric_row(r)
                row = {"index": s["index"], "sketch": s["name"], **row, "kept": why is None}
                out["critic"]["rubric"].append(row)
                if why:
                    reject("critic", s["name"], why, index=s["index"])
                    continue
                ranked.append((row["total"], -pos, {**s, "critic": row}))
        except BudgetExceeded:
            raise
        except Exception as e:
            out["critic"]["error"] = sandbox.redact(str(e))[:300]
            note({"stage": "critic", "event": "critic failed; sketches kept unranked", "error": out["critic"]["error"]})
            ranked = [(0.0, -pos, {**s, "critic": None}) for pos, s in enumerate(sketches)]
    out["critic"]["scores"] = out["critic"]["rubric"]
    kept = [s for _, _, s in sorted(ranked, key=lambda t: (t[0], t[1]), reverse=True)]
    out["kept_sketches"] = kept

    levers_by_name = {str(lv["name"]): lv for lv in pool}
    code_dir = session.out_dir(task) / "escape_v4"
    code_dir.mkdir(parents=True, exist_ok=True)
    stuck_codes = [a["code"] for a in attempts if a.get("code")]

    def checked(code, label, key):
        """(code, repaired) after the syntax check and at most one repair; code is None if still invalid."""
        problem = syntax_check(code, task.entrypoint) if code is not None else "no code in reply"
        if problem is None:
            return code, False
        rec = {"candidate": label, "error": sandbox.redact(problem)[:300], "fixed": False}
        out["repairs"].append(rec)
        if code is None:
            return None, False
        try:
            entry = repair_llm.chat(implementer, [{"role": "user", "content": REPAIR_TEMPLATE.format(
                entrypoint=task.entrypoint, error=problem, code=code)}], temperature=0.0, max_tokens=max_tokens,
                sample_idx=f"escape_v4:repair:{seed}:{key}")
            fixed = _code_from(entry, task.entrypoint)
        except BudgetExceeded:
            raise
        except Exception as e:
            rec["repair_error"] = sandbox.redact(str(e))[:300]
            return None, False
        again = syntax_check(fixed, task.entrypoint) if fixed is not None else "no code in repair reply"
        if again is not None:
            rec["after_repair"] = sandbox.redact(again)[:300]
            return None, False
        rec["fixed"] = True
        note({"stage": "repair", "candidate": label, "fixed": True})
        return fixed, True

    def admit_and_score(code, lv, base, file_stem):
        """Admission then dev score; returns the admitted record (with code) or None, and the verdict."""
        verdict = _admit(task, code, stuck_codes, lv["name"])
        if not verdict.get("admitted"):
            out["rejections"].append({"stage": "admission", **base, "reasons": list(verdict.get("reasons") or []),
                                      "disagreement": verdict.get("disagreement"),
                                      "sensitivity": verdict.get("sensitivity"),
                                      "perturbation": verdict.get("perturbation")})
            note(out["rejections"][-1])
            return None, verdict
        path = code_dir / f"{file_stem}.py"
        path.write_text(code)
        result = evaluator(path)  # dev split only
        error = (result.get("detail") or {}).get("error")
        rec = {**base, "lever": lv["name"], "path": str(path), "valid": result["valid"],
               "score": result["score"] if result["valid"] else None,
               "disagreement": verdict.get("disagreement"), "sensitivity": verdict.get("sensitivity"),
               "error": sandbox.redact(str(error))[:500] if error else None, "code": code,
               "dev_detail": dev_detail(result) if result["valid"] else None}
        note({"stage": "admitted", **{k: v for k, v in rec.items() if k not in ("code", "dev_detail")}})
        return rec, verdict

    # d. deepen top n_deepen, implement (hybrid allowed), syntax check + one repair, admit, dev score
    impl_prompts = []
    for s in kept[:n_deepen]:
        try:
            lv = levers_by_name[s["lever"]]
            description, _ = _lever_text(lv)
            row = s.get("critic") or {}
            risks = row.get("trap") or "none given"
            d = ask_json(impl_llm, implementer, DEEPEN_TEMPLATE.format(
                task=task.prompt, name=s["name"], state=s["state"] or "none", decision_rule=s["decision_rule"],
                lever=lv["name"], uses_lever=s["uses_lever"], trap=risks, constraints=constraints),
                max_tokens=max_tokens, key=f"escape_v4:deepen:{seed}:{s['index']}",
                required=("mechanism", "main_risk", "first_step"))
        except BudgetExceeded:
            raise
        except Exception as e:
            reject("deepen", s["name"], e, index=s["index"])
            continue
        deep = {"sketch": s["name"], "index": s["index"], "lever": lv["name"],
                **{k: str(d.get(k) or "") for k in ("mechanism", "main_risk", "first_step")}}
        out["deepened"].append(deep)
        note({"stage": "deepened", "sketch": s["name"], "lever": lv["name"]})
        impl_prompt = IMPLEMENT4_TEMPLATE.format(
            task=task.prompt, name=s["name"], state=s["state"] or "none", mechanism=deep["mechanism"],
            main_risk=deep["main_risk"], first_step=deep["first_step"], risks=risks, lever=lv["name"],
            description=description, siblings=_sibling_lines(kept, s), stuck=stuck, constraints=constraints)
        impl_prompts.append(impl_prompt)
        feedback = ""
        for attempt in range(max_resample + 1):
            base = {"sketch": s["name"], "index": s["index"], "attempt": attempt}
            prompt = impl_prompt + feedback
            if feedback:
                impl_prompts.append(prompt)
            try:
                entry = impl_llm.chat(implementer, [{"role": "user", "content": prompt}], temperature=1.0,
                                      max_tokens=max_tokens, sample_idx=f"escape_v4:impl:{seed}:{s['index']}:{attempt}")
                code = _code_from(entry, task.entrypoint)
                if code is None:
                    raise ValueError("no code in reply")
                code, repaired = checked(code, s["name"], f"{s['index']}:{attempt}")
                if code is None:
                    reject("syntax", s["name"], out["repairs"][-1].get("after_repair") or out["repairs"][-1]["error"],
                           index=s["index"], attempt=attempt)
                    continue
                rec, verdict = admit_and_score(code, lv, {**base, "repaired": repaired},
                                               f"{slug(str(lv['name']))}-{seed}-{s['index']}-{attempt}")
            except BudgetExceeded:
                raise
            except Exception as e:
                reject("implement", s["name"], e, index=s["index"], attempt=attempt)
                continue
            if rec is None:
                if feedback_on_reject:
                    feedback = ADMISSION_FEEDBACK.format(reasons="\n".join(
                        f"- {r}" for r in (verdict.get("reasons") or ["not admitted"])), lever=lv["name"])
                continue
            out["admitted"].append(rec)
            break
    out["implementation_prompts"] = impl_prompts

    # e. evolve once: one child per top-n_parents valid candidate (dev only, one generation)
    def valid_of(rows):
        return [c for c in rows if c["valid"] and c["score"] is not None]

    parents = valid_of(out["admitted"])
    lower = getattr(evaluator, "lower_is_better", True)
    parents = sorted(parents, key=lambda c: c["score"] if lower else -c["score"])[:n_parents]
    evolve_prompts = []
    for pi, p in enumerate(parents):
        out["evolve"]["parents"].append({"sketch": p["sketch"], "index": p["index"], "score": p["score"],
                                         "path": p["path"]})
        try:
            lv = levers_by_name[p["lever"]]
            description, _ = _lever_text(lv)
            values = dict(_cases(p.get("dev_detail")))
            worst = sorted(failure_set(p.get("dev_detail"), 3), key=lambda c: (-values.get(c, 0.0), c))
            prompt = EVOLVE_TEMPLATE.format(
                task=task.prompt, lever=lv["name"], description=description,
                direction="lower" if lower else "higher", score=f"{p['score']:.6g}",
                cases=_case_lines(p.get("dev_detail")), worst=", ".join(worst) or "(unknown)", code=p["code"],
                stuck=stuck, constraints=constraints)
            evolve_prompts.append(prompt)
            entry = impl_llm.chat(implementer, [{"role": "user", "content": prompt}], temperature=1.0,
                                  max_tokens=max_tokens, sample_idx=f"escape_v4:evolve:{seed}:{p['index']}")
            code = _code_from(entry, task.entrypoint)
            if code is None:
                raise ValueError("no code in reply")
            label = f"{p['sketch']}/child"
            code, repaired = checked(code, label, f"evolve:{p['index']}")
            if code is None:
                reject("syntax", label, out["repairs"][-1].get("after_repair") or out["repairs"][-1]["error"],
                       index=p["index"], evolve=True)
                continue
            base = {"sketch": label, "index": p["index"], "attempt": "evolve", "parent": p["path"],
                    "repaired": repaired}
            rec, _ = admit_and_score(code, lv, base, f"{slug(str(lv['name']))}-{seed}-{p['index']}-child")
        except BudgetExceeded:
            raise
        except Exception as e:
            reject("evolve", p["sketch"], e, index=p["index"])
            continue
        if rec is not None:
            out["evolve"]["children"].append(rec)
    out["evolve_prompts"] = evolve_prompts

    pool_final = valid_of(out["admitted"]) + valid_of(out["evolve"]["children"])
    if pool_final:
        top = evaluator.best([c["score"] for c in pool_final])
        best = next(c for c in pool_final if c["score"] == top)
        out.update(candidate=best["code"], candidate_path=best["path"], lever=levers_by_name[best["lever"]])
        out["evaluation"] = {"before": before, "after": best["score"], "from": "child" if best["attempt"] == "evolve"
                             else "parent", "improved": before is None or evaluator.beats(best["score"], before),
                             "split": "dev"}
    strip = lambda rows: [{k: v for k, v in c.items() if k not in ("code", "dev_detail")} for c in rows]
    out["admitted"] = strip(out["admitted"])
    out["evolve"]["children"] = strip(out["evolve"]["children"])
    out["spent_usd"] = session.ledger.spent_this_run
    return out


MODES["v4"] = escape_v4
