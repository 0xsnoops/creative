"""Mini rescue test (preliminary): does escape v2 get a naturally stuck agent unstuck better than
simple alternatives at the same budget? online_bin_packing only.

    HERD_TRACE_DIR=results/traces/rescue-20261018 uv run python scripts/rescue_test.py --pilot
    HERD_TRACE_DIR=results/traces/rescue-20261018 uv run python scripts/rescue_test.py

1. Trajectories: the implementer is one cheap sampler model (MODEL). Attempt 1 is a fresh
   sample from the task prompt; attempts 2-6 are revisions, each given the previous code and its
   dev score ("Improve it."), with no instruction to keep or change the approach. --pilot runs 2
   trajectories to check the model actually stalls near best fit; the full run uses 6.
2. Stuck states: a trajectory is stuck when the best dev score of its last 3 attempts is not
   better than the best of its first 3 by more than EPS (relative). Non-stuck ones are dropped.
3. Arms, from each stuck state, same implementer model, at most K dev-scored candidates and the
   same dollar cap per arm:
   - fresh_start: fresh context + a failure summary (scores, shared worst dev instances), no code;
   - try_different: all previous attempts with code and scores + "try a fundamentally different approach";
   - random_lever: the escape v2 pipeline with a random input the herd already uses (control);
   - herd_escape: escape v2 with the lever HERD chooses.
   Selection is on dev only. Held-out is computed once per arm for its dev-best candidate and for
   the stuck best attempt. Rescue = held-out score better than the stuck best's held-out score.
Writes results/rescue/{trajectories,arms,summary}.json.
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from diverge.llm import BudgetExceeded, message_text  # noqa: E402
from diverge.pipeline import Herd, read_json  # noqa: E402
from diverge.reality import Evaluator, dev_detail  # noqa: E402
from diverge.sample import extract_code  # noqa: E402
from diverge.task import Task  # noqa: E402

TASK = "online_bin_packing"
MODEL = "meta-llama/llama-4-maverick"  # cheapest configured sampler; its best herd samples are exactly best fit
N_ATTEMPTS, EPS, K = 6, 0.01, 3
TRAJ_BUDGET, ARM_BUDGET = 0.30, 0.40
OUT = ROOT / "results" / "rescue"

REVISE = """{task}

---
Your previous solution:
```python
{code}
```
{feedback}
Improve it. Reply with your complete solution in a single Python code block."""

FRESH = """{task}

---
{summary}

Write a solution. Reply with your complete solution in a single Python code block."""

DIFFERENT = """{task}

---
Earlier attempts at this task and their scores on the development instances (% excess bins,
lower is better):
{attempts}

Try a fundamentally different approach. Reply with your complete solution in a single Python code block."""


def feedback_for(result, baseline):
    if not result["valid"]:
        return f"It was invalid: {result['detail'].get('error')}"
    return (f"It scored {result['score']:.4f}% excess bins on the development instances (lower is better); "
            f"the best-fit baseline scores {baseline:.4f}%.")


def with_model(herd):
    herd.cfg["models"]["implementer"] = MODEL  # same implementation model in every arm
    return herd


def implement(herd, prompt, tag, evaluator, task):
    entry = herd.llm.chat(MODEL, [{"role": "user", "content": prompt}], temperature=1.0,
                          max_tokens=herd.cfg["reality_check"]["max_tokens"], sample_idx=tag)
    code = extract_code(message_text(entry), task.entrypoint)
    if code is None:
        return None, {"valid": False, "score": None, "detail": {"error": "no code in reply"}}
    path = OUT / "code" / f"{hashlib.sha256(code.encode()).hexdigest()[:12]}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return code, evaluator(path)


def trajectory(r, task, evaluator, base):
    herd = with_model(Herd(budget_usd=TRAJ_BUDGET, progress=lambda m: None))
    attempts = []
    for i in range(N_ATTEMPTS):
        if not attempts:
            prompt = task.prompt + "\n\nReply with your complete solution in a single Python code block."
        else:
            prev = attempts[-1]
            prompt = REVISE.format(task=task.prompt, code=prev["code"], feedback=prev["feedback"])
        try:
            code, result = implement(herd, prompt, f"rescue:traj:{r}:{i}", evaluator, task)
        except BudgetExceeded:
            break
        if code is None:  # keep the previous code as the state; the agent got no code this turn
            if not attempts:
                continue
            code, result = attempts[-1]["code"], {"valid": False, "score": None, "detail": {"error": "no code in reply"}}
        attempts.append({"id": f"attempt-{i + 1}", "code": code, "dev_score": result["score"],
                         "valid": result["valid"], "dev_detail": dev_detail(result),
                         "feedback": feedback_for(result, base)})
        print(f"traj {r} attempt {i + 1}: dev {result['score']}", flush=True)
    return {"run": r, "attempts": attempts, "cost_usd": round(herd.ledger.new_spend, 4)}


def stuck(attempts):
    scores = [a["dev_score"] if a["valid"] else float("inf") for a in attempts]
    if len(scores) < N_ATTEMPTS or min(scores) == float("inf"):
        return False
    early, late = min(scores[:3]), min(scores[3:])
    return not late < early * (1 - EPS)


def failure_summary(attempts):
    from diverge.failures import failure_set
    scored = [a for a in attempts if a["valid"]]
    sets = [failure_set(a["dev_detail"]) for a in scored]
    shared = sorted(frozenset.intersection(*sets)) if sets else []
    lines = ["Earlier attempts at this task scored, on the development instances (% excess bins, lower is better): "
             + ", ".join(f"{a['dev_score']:.4f}" if a["valid"] else "invalid" for a in attempts) + "."]
    lines.append(f"The best scored {min(a['dev_score'] for a in scored):.4f}.")
    if shared:
        lines.append(f"Every valid attempt did worst on the same development instances: {', '.join(shared)}.")
    lines.append("These attempts have stalled.")
    return "\n".join(lines)


def plain_arm(name, prompt, state, task, evaluator):
    herd = with_model(Herd(budget_usd=ARM_BUDGET, progress=lambda m: None))
    cands = []
    for i in range(K):
        try:
            code, result = implement(herd, prompt, f"rescue:{name}:{state['run']}:{i}", evaluator, task)
        except BudgetExceeded:
            break
        cands.append({"code": code, "dev_score": result["score"], "valid": result["valid"]})
    valid = [c for c in cands if c["valid"]]
    best = min(valid, key=lambda c: c["dev_score"]) if valid else None
    return {"arm": name, "candidates": cands, "code": best["code"] if best else None,
            "dev_score": best["dev_score"] if best else None, "cost_usd": round(herd.ledger.new_spend, 4),
            "implementer_calls": len(cands)}


def escape_arm(name, state, task, evaluator, overlook, lever=None):
    from diverge.escape import escape_v2
    herd = with_model(Herd(budget_usd=ARM_BUDGET, progress=lambda m: None))
    attempts = [{"id": a["id"], "code": a["code"], "score": a["dev_score"], "valid": a["valid"],
                 "dev_detail": a["dev_detail"]} for a in state["attempts"]]
    try:
        res = escape_v2(herd, task, attempts[-1], attempts[:-1], evaluator, overlook=overlook, lever=lever,
                        k_sketches=K, max_resample=2, seed=state["run"])
    except BudgetExceeded as e:
        return {"arm": name, "code": None, "dev_score": None, "error": f"budget: {e}",
                "cost_usd": round(herd.ledger.new_spend, 4)}
    keep = {k: res.get(k) for k in ("lever", "failure_summary", "counterexample", "sketches", "kept_sketches",
                                    "rejections", "admitted", "calls")}
    return {"arm": name, "code": res.get("candidate"), "dev_score": (res.get("evaluation") or {}).get("after"),
            "cost_usd": round(herd.ledger.new_spend, 4), **keep}


def heldout(evaluator, code):
    if not code:
        return None
    path = OUT / "code" / f"final_{hashlib.sha256(code.encode()).hexdigest()[:12]}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return evaluator.final(path, None)[0]


def main():
    pilot = "--pilot" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    task = Task.resolve(TASK)
    evaluator = Evaluator(task)
    base = {k: r["score"] for k, r in task.baselines("dev").items()}
    traj_path = OUT / "trajectories.json"
    trajs = read_json(traj_path, {}) or {}
    for r in range(2 if pilot else 6):
        if str(r) not in trajs:
            trajs[str(r)] = trajectory(r, task, evaluator, base["best_fit"])
            traj_path.write_text(json.dumps(trajs, indent=2))
    for t in trajs.values():
        t["stuck"] = stuck(t["attempts"])
        print(f"traj {t['run']}: {[a['dev_score'] and round(a['dev_score'], 3) for a in t['attempts']]} "
              f"stuck={t['stuck']} cost ${t['cost_usd']}")
    traj_path.write_text(json.dumps(trajs, indent=2))
    if pilot:
        return

    overlook = read_json(ROOT / "results" / TASK / "overlook.json")
    used = [r["input"] for r in overlook["inputs"] if (r.get("share") or 0) >= 0.5]
    arms_path = OUT / "arms.json"
    arms = read_json(arms_path, {}) or {}
    for key, t in trajs.items():
        if not t["stuck"] or key in arms:
            continue
        scored = [a for a in t["attempts"] if a["valid"]]
        best = min(scored, key=lambda a: a["dev_score"])
        listing = "\n\n".join(f"Attempt {i + 1} (dev score {a['dev_score']}):\n```python\n{a['code']}\n```"
                              for i, a in enumerate(t["attempts"]))
        rng = random.Random(1000 + t["run"])
        results = [plain_arm("fresh_start", FRESH.format(task=task.prompt, summary=failure_summary(t["attempts"])),
                             t, task, evaluator),
                   plain_arm("try_different", DIFFERENT.format(task=task.prompt, attempts=listing), t, task, evaluator),
                   escape_arm("random_lever", t, task, evaluator, overlook, lever=rng.choice(used)),
                   escape_arm("herd_escape", t, task, evaluator, overlook)]
        stuck_held = heldout(evaluator, best["code"])
        for a in results:
            a["heldout_score"] = heldout(evaluator, a.get("code"))
            a["rescued"] = (a["heldout_score"] is not None and stuck_held is not None
                            and a["heldout_score"] < stuck_held)
        arms[key] = {"run": t["run"], "stuck_best": {"id": best["id"], "dev_score": best["dev_score"],
                                                     "heldout_score": stuck_held}, "arms": results}
        arms_path.write_text(json.dumps(arms, indent=2))
        print(f"state {key}: stuck best held-out {stuck_held}; "
              + "; ".join(f"{a['arm']} {a['heldout_score']} rescued={a['rescued']} ${a['cost_usd']}" for a in results),
              flush=True)

    names = ["fresh_start", "try_different", "random_lever", "herd_escape"]
    summary = {"preliminary": True, "model": MODEL, "n_trajectories": len(trajs),
               "n_stuck": sum(t["stuck"] for t in trajs.values()), "k_per_arm": K, "arm_budget_usd": ARM_BUDGET,
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "per_arm": {n: {"rescued": sum(a["rescued"] for s in arms.values() for a in s["arms"] if a["arm"] == n),
                               "n": len(arms),
                               "cost_usd": round(sum(a["cost_usd"] for s in arms.values() for a in s["arms"]
                                                     if a["arm"] == n), 4)} for n in names},
               "trajectory_cost_usd": round(sum(t["cost_usd"] for t in trajs.values()), 4)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
