"""ESCAPE: when an agent's attempts are different code but the same strategy, pick ONE lever the
attempts (and most of the herd) leave unused and implement a solution built on it.

Levers, in order of preference: an input the herd overlooks and the attempts never use; an
incidental assumption shared by the herd and by every attempt; a family only one or two sampler
models produced; a blind-spot family. Evidence comes from the overlook study and dev-split
scores only: held-out results never enter a prompt. Candidate code only runs through the
Evaluator (sandboxed)."""
import hashlib
import inspect

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
