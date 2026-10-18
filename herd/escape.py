"""ESCAPE: when an agent's attempts are different code but the same strategy, pick ONE lever the
attempts (and most of the herd) leave unused and implement a solution built on it.

Levers, in order of preference: an input the herd overlooks and the attempts never use; an
incidental assumption shared by the herd and by every attempt; a family only one or two sampler
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
                    "why": f"{r['input']} is used by {r.get('used_by')}/{r.get('decided')} herd samples "
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
                    "why": f"held by {a.get('held_by')}/{a.get('checked')} checked herd samples and by all {n} of your "
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
                    "mechanism": f"{fam['family']}: {fam.get('one_line_mechanism') or 'a sourced family the herd rarely uses'}.",
                    "why": f"a sourced family the herd rarely produces ({seen})"})
    return out


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
Evidence: {why}

The current solution, shown only as what to move away from (do not tune it):
```python
{code}
```

Write a complete solution whose decisions materially depend on {lever}: changing it must change
which option is chosen. Keep the task's interface and rules.
Reply with your complete solution in a single Python code block."""


def escape_prompt(task, choice, diagnosis, current_code):
    lever = choice["lever"]
    if lever is None:
        raise ValueError("no lever to pursue")
    quotes = choice.get("evidence", {}).get("hard_constraints") or []
    constraints = "\n".join(f'- "{q}"' for q in quotes) or "- (none recorded beyond the interface and rules above)"
    return ESCAPE_TEMPLATE.format(task=task.prompt, shared_cause=diagnosis["shared_cause"],
                                  failure=diagnosis["failure"], constraints=constraints,
                                  lever_type=lever["type"].replace("_", " "), lever=lever["name"],
                                  mechanism=choice["mechanism"], why=choice["why"],
                                  code=(current_code or "")[:CODE_LIMIT])


def generate(herd, task, choice, diagnosis, current_code, evaluator, k=1, seed=0):
    """k implementations of the escape prompt, each scored on the dev split."""
    prompt = escape_prompt(task, choice, diagnosis, current_code)
    code_dir = herd.out_dir(task) / "escape"
    code_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for i in range(k):
        try:
            entry = herd.llm.chat(herd.cfg["models"]["implementer"], [{"role": "user", "content": prompt}],
                                  temperature=1.0, max_tokens=herd.cfg["reality_check"]["max_tokens"],
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
    """herd.trajectory.analyze, called with whichever of these arguments it takes by name."""
    from . import trajectory  # imported late: written in parallel with this module
    params = inspect.signature(trajectory.analyze).parameters
    if "attempts" not in params:
        return trajectory.analyze(task, attempts)
    offered = {"task": task, "attempts": attempts, "evaluator": evaluator, "extract": extract,
               "lower_is_better": getattr(evaluator, "lower_is_better", True)}
    if "baseline_detail" in params:
        offered["baseline_detail"] = baseline_detail(task, evaluator, attempts)
    return trajectory.analyze(**{k: v for k, v in offered.items() if k in params})


def escape(herd, task, current_solution, previous_attempts, evaluator, overlook=None, families=None,
           extract=None, k=1):
    """Diagnose the attempts and, if they loop on one strategy, implement one escape lever."""
    attempts = [_attempt(a, i) for i, a in enumerate([*previous_attempts, current_solution])]
    score_dir = herd.out_dir(task) / "escape" / "attempts"
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
        generated = generate(herd, task, choice, diagnosis, current["code"], evaluator, k=k)
    best = (generated or {}).get("best")
    before, after = current.get("score"), best["score"] if best else None
    improved = after is not None and (before is None or evaluator.beats(after, before))
    lever = [{k: choice["lever"][k] for k in ("type", "name", "detail")} | {"mechanism": choice["mechanism"],
                                                                            "why": choice["why"]}] if choice["lever"] else []
    return {"loop_detected": report.get("loop_detected"), "strategy_similarity": report.get("strategy_similarity"),
            "code_similarity": report.get("code_similarity"), "shared_strategy": diagnosis["shared_cause"],
            "shared_assumptions": (report.get("shared") or {}).get("assumptions", []),
            "herd_prevalence": diagnosis["herd_prevalence"],
            "overlooked_levers": lever + choice["alternatives"],
            "recommended_switch": {"mechanism": choice["mechanism"], "why": choice["why"]},
            "candidate": best["code"] if best else None, "candidate_path": best["path"] if best else None,
            "candidates": generated["candidates"] if generated else [],
            "evaluation": {"before": before, "after": after, "improved": improved, "split": "dev"},
            "action": diagnosis["action"], "diagnosis": diagnosis, "reasons": diagnosis["reasons"]}


# ================================================================ escape v2
# Research-guided escape (docs/RESEARCH-loop-escape.md: consolidated restart, agreement
# counterexample, idea-level search, admission gate). Fresh contexts only: no attempt code ever
# enters a v2 prompt; the agent sees a failure summary, one lever with its herd evidence and one
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
    """herd.llm with a per-role call counter (ask_json may retry, so count at chat level)."""

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
        share = f"{row['used_by']}/{row['decided']} herd samples" if row and row.get("decided") else "unknown"
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
    """Herd sample ids that use the lever: the overlook input row's evidence, else the lever's own."""
    row = next((r for r in (overlook or {}).get("inputs", []) if r.get("input") == lever["name"]), None)
    evidence = (row or {}).get("evidence") or (lever.get("detail") or {}).get("evidence") or []
    ids = [e.get("sample_id") if isinstance(e, dict) else e for e in evidence]
    return list(dict.fromkeys(i for i in ids if isinstance(i, str)))


def _herd_programs(herd, task, sample_ids, limit=MAX_ALTERNATIVES):
    """{sample_id: code} for up to limit valid herd samples among sample_ids."""
    out_dir = herd.out_dir(task)
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
    alt = f"the alternative (herd sample {cx['alternative_id']}, which uses {lever_name})"
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


def _counterexample_v2(herd, task, attempts, lever, overlook, seed):
    """The best agreement counterexample whose alternative (a herd program using the lever) is
    better, or None; plus a note on what was searched."""
    from . import counterexample as ce
    if task.name not in (ce.BIN, ce.CACHE) or not getattr(task, "evaluator", None):
        return None, "no counterexample: this task has no probe mode"
    alternatives = _herd_programs(herd, task, _lever_samples(lever, overlook))
    if not alternatives:
        return None, f"no counterexample: no valid herd program that uses {lever['name']} was found"
    stuck = {a["id"]: a["code"] for a in attempts if a.get("code")}
    try:
        found = ce.find(task, stuck, alternatives, n=CX_CASES, seed=seed)
    except BudgetExceeded:
        raise
    except Exception as e:  # probes are sandboxed; a probe failure only means no counterexample
        return None, f"no counterexample: probe failed ({sandbox.redact(str(e))[:200]})"
    better = [c for c in found["counterexamples"] if c["alternative_better"]]
    if not better:
        return None, (f"no counterexample: on {found['summary']['cases']} tiny cases no herd program using "
                      f"{lever['name']} beat the attempts where they all agree")
    cx = better[0]
    described = describe_counterexample(task.name, cx, lever["name"], found["summary"]["valid_attempts"])
    return ({"case": cx["case"], "alternative_id": cx["alternative_id"], "attempts_output": cx["attempts_output"],
             "alternative_output": cx["alternative_output"], "alternative_better": True,
             "summary": found["summary"], **described}, None)


def failure_summary(attempts, diagnosis, evaluator, k=2):
    """Dev scores, the best score, the shared failure set and the shared strategy; never code."""
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
    return "\n".join(lines)


def _lever_text(lever):
    detail = lever.get("detail") or {}
    description = detail.get("description") or lever.get("mechanism") or "the lever"
    evidence = [e for e in detail.get("evidence") or [] if isinstance(e, dict) and e.get("evidence")]
    ev = "".join(f"\n- {e.get('model') or e.get('sample_id')}: {e['evidence']}" for e in evidence[:3])
    return description, (f"\nHow herd programs use it:{ev}" if ev else "")


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
    try:
        return admission.admit(task, code, stuck_codes, lever_name)
    except BudgetExceeded:
        raise
    except Exception as e:
        return {"admitted": False, "disagreement": None, "sensitivity": None, "perturbation": None,
                "reasons": [f"admission failed: {sandbox.redact(str(e))[:300]}"], "valid": False}


def escape_v2(herd, task, current_solution, previous_attempts, evaluator, overlook=None, families=None, lever=None,
              k_sketches=3, max_resample=2, seed=0, log=None, extract=None):
    """Escape v2: failure summary + one lever + agreement counterexample -> k mechanism sketches
    (fresh context, no code) -> keep sketches that use the lever and differ from the stuck
    strategy -> implement each, admit (resampling up to max_resample times) -> best on dev."""
    from .llm import ask_json
    events, calls = [], {"implementer": 0, "extractor": 0}

    def note(event):
        events.append(event)
        if log:
            log(event)

    attempts = [_attempt(a, i) for i, a in enumerate([*previous_attempts, current_solution])]
    score_dir = herd.out_dir(task) / "escape_v2" / "attempts"
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
    if lever is None:
        choice = choose(diagnosis, report, overlook, families)
        chosen = choice["lever"] and {**choice["lever"], "mechanism": choice["mechanism"], "why": choice["why"]}
    else:
        chosen = _lever_v2(lever, overlook)
    before = current.get("score")
    out = {"version": 2, "lever": chosen, "failure_summary": None, "counterexample": None, "sketches": [],
           "kept_sketches": [], "rejections": [], "admitted": [], "candidate": None, "candidate_path": None,
           "evaluation": {"before": before, "after": None, "improved": False, "split": "dev"},
           "calls": calls, "diagnosis": diagnosis, "analysis": report, "action": diagnosis["action"], "log": events}
    summary = failure_summary(attempts, diagnosis, evaluator)
    out["failure_summary"] = summary
    if chosen is None:
        note({"stage": "lever", "event": "no lever available"})
        out["spent_usd"] = herd.ledger.spent_this_run
        return out

    cx, no_cx = _counterexample_v2(herd, task, attempts, chosen, overlook, seed)
    out["counterexample"] = cx
    note({"stage": "counterexample", "found": cx is not None, "note": no_cx})

    implementer = herd.cfg["models"]["implementer"]
    extractor = herd.cfg["models"].get("extractor") or herd.cfg["models"].get("labeler") or implementer
    impl_llm, judge_llm = _Counted(herd.llm, calls, "implementer"), _Counted(herd.llm, calls, "extractor")

    prompt = sketch_prompt(task, summary, chosen, cx, overlook, k=k_sketches, no_cx_note=no_cx)
    reply = ask_json(impl_llm, implementer, prompt, max_tokens=herd.cfg["reality_check"]["max_tokens"],
                     key=f"escape_v2:sketch:{seed}", temperature=1.0, required=("sketches",))
    sketches = [s for s in reply.get("sketches") or [] if isinstance(s, dict)][:k_sketches]
    out["sketches"], out["sketch_prompt"] = sketches, prompt

    kept = []
    for i, s in enumerate(sketches):
        if not mentions_lever(s, chosen):
            reason = f"decision rule does not use {chosen['name']}"
            out["rejections"].append({"stage": "sketch_filter", "sketch": s.get("name"), "reasons": [reason]})
            note(out["rejections"][-1])
            continue
        verdict = ask_json(judge_llm, extractor, JUDGE_TEMPLATE.format(
            shared=diagnosis["shared_cause"], name=s.get("name", ""), state=s.get("state", ""),
            decision_rule=s.get("decision_rule", ""), lever=chosen["name"], uses_lever=s.get("uses_lever", "")),
            max_tokens=2000, key=f"escape_v2:same:{seed}:{i}", required=("same_mechanism",))
        if verdict.get("same_mechanism") is not False:
            reason = f"same mechanism as the shared strategy: {verdict.get('reason') or 'judged the same'}"
            out["rejections"].append({"stage": "sketch_filter", "sketch": s.get("name"), "reasons": [reason]})
            note(out["rejections"][-1])
            continue
        kept.append({**s, "index": i, "difference": verdict.get("reason")})
    out["kept_sketches"] = kept

    code_dir = herd.out_dir(task) / "escape_v2"
    code_dir.mkdir(parents=True, exist_ok=True)
    stuck_codes = [a["code"] for a in attempts if a.get("code")]
    prompts = []
    for s in kept:
        impl_prompt = implementation_prompt_v2(task, s, chosen, overlook)
        prompts.append(impl_prompt)
        for attempt in range(max_resample + 1):
            base = {"stage": "admission", "sketch": s.get("name"), "attempt": attempt}
            try:
                entry = impl_llm.chat(implementer, [{"role": "user", "content": impl_prompt}], temperature=1.0,
                                      max_tokens=herd.cfg["reality_check"]["max_tokens"],
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
                continue
            path = code_dir / f"{slug(chosen['name'])}-{seed}-{s['index']}-{attempt}.py"
            path.write_text(code)
            result = evaluator(path)  # dev split only
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
    out["spent_usd"] = herd.ledger.spent_this_run
    return out
