"""Admission gate for escape candidates (docs/RESEARCH-loop-escape.md, steps [7] and lever 2).
Runs BEFORE full scoring and rejects a candidate unless

1. it behaves differently from EVERY stuck program: its probe fingerprint (herd/fingerprint.py)
   differs from each stuck program's on at least min_disagreement of the probe states, and
2. it actually depends on its lever's input: perturbing that input in the probe states changes
   at least min_sensitivity of its decisions (the faithfulness check; it catches a lever used
   only in a token way, such as an `if` that never fires).

Everything candidate-written runs in the task sandbox (tasks/_sandbox.py). Only online bin
packing has fingerprint and perturbation modes. Limits: probes are single decisions on best-fit
trajectory states, so this measures local decision changes, not whether the change helps; a
candidate that inspects the evaluator process could still tell runs apart."""
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from herd import fingerprint as fp

# What each perturbation does and why the result still measures dependence on that input.
# Decisions are compared by the original capacity of the chosen option (a new bin = 100), as the
# fingerprint does, so picks between equal bins never count as a change.
PERTURBATIONS = {
    "online_bin_packing": {
        "item": "item size +1 or -1 (deterministic seed), only where every offered bin still fits "
                "it; the offered bins are unchanged. Sound: only the item changes. Note best fit's "
                "choice does not depend on the item once feasibility is fixed, so it scores ~0 here.",
        "capacity_values": "each open bin's remaining capacity +1 or -1 (deterministic seed), kept in "
                           "item..99 so every bin stays feasible; the new bin stays 100 and last. The "
                           "chosen position is mapped back to the original capacity it stands for.",
        "bin_order": "the open bins are reversed; the new bin stays last. Same options, different "
                     "positions; probes where reversal changes nothing (palindromes, one distinct capacity) are skipped.",
        "n_open_bins": "one extra copy of a randomly chosen open bin (deterministic seed) is added "
                       "before the new bin: the list is longer but the set of distinct options is "
                       "the same, so a choice by capacity is unaffected unless the code uses the count.",
        "history": "before each probe seed's block of 50 probes the candidate packs a fixed sequence "
                   "of 50 warm-up items from Weibull(3, 45), versus no warm-up. Probe states are "
                   "unchanged; only module-level state carried across calls can differ.",
        "new_bin_slot": None,  # moving the new-bin entry also shifts every bin's position (it is
        # confounded with bin_order) and breaks the evaluator's contract that the last entry is the new bin
        "item_distribution": None,  # a constant of the task, not an argument: nothing to perturb per call
        "bin_capacity": None,  # likewise a constant (100); changing it changes the problem, not one input
    },
}


def supported(task):
    ev = getattr(task, "evaluator", None)
    return bool(ev) and hasattr(ev, "fingerprint") and hasattr(ev, "fingerprint_perturbed")


def _lever_name(lever):
    if isinstance(lever, dict):
        return lever.get("name")
    return lever


def sensitivity(task, path, lever):
    """{"sensitivity", "noise", "applicable", "perturbation", "valid", "error"} for one input.
    sensitivity = fraction of applicable probes whose chosen capacity changes under the
    perturbation; noise = the same fraction between two unperturbed runs (nondeterminism).
    sensitivity is None when there is no sound perturbation for the input."""
    table = PERTURBATIONS.get(task.name, {})
    text = table.get(lever)
    if text is None:
        return {"sensitivity": None, "noise": None, "applicable": 0, "valid": True, "error": None,
                "perturbation": f"no sound perturbation for {lever}"}
    result = task.evaluator.fingerprint_perturbed(path, lever)
    if not result["valid"]:
        return {"sensitivity": None, "noise": None, "applicable": 0, "valid": False,
                "error": result["detail"].get("error"), "perturbation": text}
    d = result["detail"]
    idx = [i for i, a in enumerate(d["applicable"]) if a]
    changed = sum(d["base"][i] != d["perturbed"][i] for i in idx)
    noise = fp.distance(d["base"], d["repeat"])
    return {"sensitivity": changed / len(idx) if idx else 0.0, "noise": noise, "applicable": len(idx),
            "valid": True, "error": None, "perturbation": text}


def admit(task, candidate_code, stuck_codes, lever, min_disagreement=0.10, min_sensitivity=0.05):
    """Admission check for one escape candidate (code strings, run sandboxed).

    task: a herd.task.Task (or a registered task name); stuck_codes: list or dict of the stuck
    cluster's code; lever: the input name (e.g. "n_open_bins") or a lever dict with "name".
    Returns {"admitted", "valid", "disagreement" (mean over stuck programs),
    "min_disagreement_vs_stuck", "disagreement_vs_stuck" (per stuck program), "sensitivity"
    (None when the lever has no sound perturbation or the candidate is invalid),
    "sensitivity_noise", "probes_perturbed", "perturbation", "lever", "reasons"}.
    Admitted only if valid, min disagreement >= min_disagreement, and (when a sound perturbation
    exists) sensitivity >= min_sensitivity + noise."""
    if isinstance(task, str):
        from herd.task import Task
        task = Task.registered(task)
    if not supported(task):
        raise ValueError(f"task {getattr(task, 'name', task)!r} has no fingerprint/perturbation mode")
    name = _lever_name(lever)
    stuck = stuck_codes if isinstance(stuck_codes, dict) else {f"stuck_{i}": c for i, c in enumerate(stuck_codes)}
    out = {"admitted": False, "valid": False, "disagreement": None, "min_disagreement_vs_stuck": None,
           "disagreement_vs_stuck": {}, "sensitivity": None, "sensitivity_noise": None, "probes_perturbed": 0,
           "perturbation": None, "lever": name, "reasons": []}
    with tempfile.TemporaryDirectory() as d:
        paths = {"__candidate__": os.path.join(d, "candidate.py")}
        for i, key in enumerate(stuck):
            paths[key] = os.path.join(d, f"stuck_{i}.py")
        for key, path in paths.items():
            with open(path, "w") as f:
                f.write(candidate_code if key == "__candidate__" else stuck[key])
        with ThreadPoolExecutor(2) as pool:
            fps_future = pool.submit(fp.fingerprints, task, paths)
            sens_future = pool.submit(sensitivity, task, paths["__candidate__"], name)
            fps, sens = fps_future.result(), sens_future.result()

    out["perturbation"] = sens["perturbation"]
    cand = fps.pop("__candidate__")
    if cand is None or not sens["valid"]:
        where = "probes" if cand is None else "perturbed probes"
        out["reasons"].append(f"invalid: candidate crashed or returned invalid scores on the {where}"
                              + (f" ({sens['error']})" if sens.get("error") else ""))
        return out
    out["valid"] = True

    bad = [k for k, v in fps.items() if v is None]
    if bad:
        out["reasons"].append(f"ignored stuck programs invalid on probes: {', '.join(bad)}")
    per = {k: fp.distance(cand, v) for k, v in fps.items() if v is not None}
    out["disagreement_vs_stuck"] = per
    ok = True
    if not per:
        out["reasons"].append("no valid stuck program to compare against")
        ok = False
    else:
        out["disagreement"] = sum(per.values()) / len(per)
        out["min_disagreement_vs_stuck"] = low = min(per.values())
        if low < min_disagreement:
            closest = min(per, key=per.get)
            out["reasons"].append(f"too close to the stuck cluster: differs from {closest} on {low:.1%} "
                                  f"of probe states (< {min_disagreement:.0%})")
            ok = False

    out["sensitivity"], out["sensitivity_noise"] = sens["sensitivity"], sens["noise"]
    out["probes_perturbed"] = sens["applicable"]
    if sens["sensitivity"] is None:
        out["reasons"].append(f"no sound perturbation for {name}; admission decided on disagreement only")
    else:
        if sens["noise"]:
            out["reasons"].append(f"candidate is nondeterministic: {sens['noise']:.1%} of decisions change "
                                  "between identical runs; required sensitivity raised by that much")
        need = min_sensitivity + (sens["noise"] or 0.0)
        if sens["sensitivity"] < need:
            out["reasons"].append(f"does not depend on {name}: perturbing it changes {sens['sensitivity']:.1%} "
                                  f"of decisions (< {need:.1%})")
            ok = False
    out["admitted"] = ok
    if ok:
        out["reasons"].append("admitted")
    return out
