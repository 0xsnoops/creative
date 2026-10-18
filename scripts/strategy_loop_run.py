"""One live end-to-end run for the strategy-loop demo (online_bin_packing). Writes
fixtures/trajectories/strategy_loop.json and fixtures/runs/strategy_loop.json.

1. Trajectory: attempt 1 is a real consensus sample from the implementer model in the consensus'
   dominant family; attempts 2-4 are real implementer revisions, each given its previous
   code and dev score and asked to improve it (the usual agent loop).
2. DIVERGE escape on that trajectory (k=2 candidates).
3. fresh_start at the same budget: the implementer gets the task, the attempts and their dev
   scores, and "try a different strategy", with no consensus information (k=2 candidates).
Selection uses dev scores only; held-out scores are computed once at the end for reporting.

    DIVERGE_TRACE_DIR=results/traces/demo-20261018 uv run python scripts/strategy_loop_run.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from diverge.llm import message_text  # noqa: E402
from diverge.pipeline import Session, read_json  # noqa: E402
from diverge.reality import Evaluator  # noqa: E402
from diverge.sample import extract_code  # noqa: E402
from diverge.server import escape_for_agent  # noqa: E402
from diverge.task import Task  # noqa: E402

TASK, REVISIONS, K, BUDGET_EACH = "online_bin_packing", 3, 2, 1.50
# --mode tune (default, fixture "strategy_loop_tune"): each revision is told to keep its approach and tune it,
#   the plateau pattern of an agent refining one idea. --mode open (fixture "strategy_loop_unconstrained"):
#   each revision is just told to improve; recorded first, and it did not get stuck.
MODE = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "tune"
# --mode plateau (fixture "strategy_loop", the demo): continue the tune trajectory 4 more revisions from its
#   last attempt and use those 4 as the agent's current window (agents call DIVERGE on recent attempts).
NAME = {"tune": "strategy_loop_tune", "open": "strategy_loop_unconstrained", "plateau": "strategy_loop"}[MODE]
# Cache tags as recorded (fixtures were renamed afterwards); keeps reruns free.
TAG = {"tune": "strategy_loop", "open": "strategy_loop", "plateau": "strategy_loop_plateau"}[MODE]
KEEP = ("Keep the same approach: tune its parameters and details rather than switching to a different "
        "method.\n") if MODE in ("tune", "plateau") else ""

REVISE = """{task}

---
Your previous solution:
```python
{code}
```
{feedback}
{keep}Improve it. Reply with your complete solution in a single Python code block."""

FRESH = """{task}

---
Earlier attempts at this task and their scores on the development instances (% excess bins,
lower is better):
{attempts}

These attempts have stalled. Try a different strategy. Reply with your complete solution in a
single Python code block."""


def feedback_for(result, baseline):
    if not result["valid"]:
        return f"It was invalid: {result['detail'].get('error')}"
    return (f"It scored {result['score']:.4f}% excess bins on the development instances (lower is better); "
            f"the best-fit baseline scores {baseline:.4f}%.")


def implement(session, prompt, tag, out_dir, evaluator, task):
    entry = session.llm.chat(session.cfg["models"]["implementer"], [{"role": "user", "content": prompt}],
                          temperature=1.0, max_tokens=session.cfg["reality_check"]["max_tokens"], sample_idx=tag)
    code = extract_code(message_text(entry), task.entrypoint)
    if code is None:
        return None, {"valid": False, "score": None, "detail": {"error": "no code in reply"}}
    path = out_dir / f"{tag.replace(':', '_')}.py"
    path.write_text(code)
    return code, evaluator(path)


def main():
    task = Task.resolve(TASK)
    evaluator = Evaluator(task)
    out = ROOT / "results" / TASK / "demo"
    out.mkdir(parents=True, exist_ok=True)
    dev_base = {k: r["score"] for k, r in task.baselines("dev").items()}
    held_base = {k: r["score"] for k, r in task.baselines("heldout").items()}
    consensus_map = read_json(ROOT / "results" / TASK / "herd_map.json")
    samples = [json.loads(l) for l in (ROOT / "results" / TASK / "samples.jsonl").read_text().splitlines()]
    labels = {json.loads(l)["sample_id"]: json.loads(l)["family"]
              for l in (ROOT / "results" / TASK / "labels.jsonl").read_text().splitlines()}
    implementer = Session().cfg["models"]["implementer"]

    # 1. Trajectory.
    seed = next(s for fam in consensus_map["top3"] for s in sorted(samples, key=lambda s: s["idx"])
                if s["model"] == implementer and s["status"] == "valid" and labels.get(s["sample_id"]) == fam)
    traj_session = Session(budget_usd=BUDGET_EACH)
    if MODE == "plateau":
        base = json.loads((ROOT / "fixtures" / "trajectories" / "strategy_loop_tune.json").read_text())["attempts"]
        last = base[-1]
        prior = {"id": last["id"], "code": last["code"], "dev_score": last["dev_score"],
                 "feedback": feedback_for({"valid": True, "score": last["dev_score"], "detail": {}}, dev_base["best_fit"])}
        window = []
        for i in range(4):
            prev = window[-1] if window else prior
            prompt = REVISE.format(task=task.prompt, code=prev["code"], feedback=prev["feedback"], keep=KEEP)
            code, result = implement(traj_session, prompt, f"trajectory:{TAG}:{i}", out, evaluator, task)
            if code is None:
                break
            window.append({"id": f"attempt-{len(base) + i + 1}", "code": code, "dev_score": result["score"],
                           "feedback": feedback_for(result, dev_base["best_fit"]),
                           "source": f"implementer revision of {prev['id']} with dev-score feedback, told to keep the approach"})
            print(f"{window[-1]['id']}: dev {result['score']}", flush=True)
        attempts = window
    else:
        attempts = None
    if attempts is None:
        attempts = [{"id": "attempt-1", "code": (ROOT / "results" / TASK / seed["code_path"]).read_text(),
                     "dev_score": seed["score"], "source": f"consensus sample {seed['sample_id']} ({labels[seed['sample_id']]})"}]
        first = {"valid": True, "score": seed["score"], "detail": {}}  # the map already scored this sample on dev
        attempts[0]["feedback"] = feedback_for(first, dev_base["best_fit"])
        for i in range(REVISIONS):
            prev = attempts[-1]
            prompt = REVISE.format(task=task.prompt, code=prev["code"], feedback=prev["feedback"], keep=KEEP)
            code, result = implement(traj_session, prompt, f"trajectory:{TAG}:{i}", out, evaluator, task)
            if code is None:
                break
            attempts.append({"id": f"attempt-{i + 2}", "code": code, "dev_score": result["score"],
                             "feedback": feedback_for(result, dev_base["best_fit"]),
                             "source": f"implementer revision {i + 1} of attempt-{i + 1} with dev-score feedback"})
            print(f"attempt-{i + 2}: dev {result['score']}", flush=True)
    attempts[-1]["feedback"] = None  # the last attempt is where the agent is now
    fixture = {"name": NAME, "task": TASK, "attempts": attempts, "how_generated": (
        f"attempt-1 is a real consensus-map sample from the implementer model ({implementer}) in the consensus' dominant "
        f"families; attempts 2-{len(attempts)} are real revisions by the same model, each given the previous code "
        f"and its dev score and asked to improve it"
        + (" while keeping the same approach (tuning parameters and details, not switching method). This "
           "instruction was meant to keep the trajectory in one strategy (in the recorded run it still kept improving); an earlier run without it "
           "(fixture strategy_loop_unconstrained) kept improving and was not stuck" if MODE == "tune" else "")
        + f". Generated {time.strftime('%Y-%m-%d %H:%M %Z')}; traj cost "
        f"${traj_session.ledger.new_spend:.4f}.")}
    (ROOT / "fixtures" / "trajectories").mkdir(parents=True, exist_ok=True)
    (ROOT / "fixtures" / "trajectories" / f"{NAME}.json").write_text(json.dumps(fixture, indent=2))

    # 2. DIVERGE escape.
    esc_session = Session(budget_usd=BUDGET_EACH)
    esc = escape_for_agent(esc_session, task, attempts[-1]["code"], [a["code"] for a in attempts[:-1]], evaluator, k=K)
    esc_cost, esc_calls = esc_session.ledger.new_spend, None

    # 3. fresh_start, same implementer, k and caps.
    fresh_session = Session(budget_usd=BUDGET_EACH)
    listing = "\n\n".join(f"Attempt {i + 1} (dev score {a['dev_score']}):\n```python\n{a['code']}\n```"
                          for i, a in enumerate(attempts))
    fresh_prompt = FRESH.format(task=task.prompt, attempts=listing)
    fresh = []
    for i in range(K):
        code, result = implement(fresh_session, fresh_prompt, f"fresh_start:{TAG}:{i}", out, evaluator, task)
        fresh.append({"code": code, "dev_score": result["score"], "valid": result["valid"],
                      "error": result["detail"].get("error")})

    # Held-out, once, for the dev-selected picks.
    def heldout(code, tag):
        if not code:
            return None
        path = out / f"final_{tag}.py"
        path.write_text(code)
        return evaluator.final(path, None)[0]

    scored = [a for a in attempts if a["dev_score"] is not None]
    best_attempt = min(scored, key=lambda a: a["dev_score"])
    valid_fresh = [f for f in fresh if f["valid"]]
    best_fresh = min(valid_fresh, key=lambda f: f["dev_score"]) if valid_fresh else None
    try:
        from diverge import contract
        contract_text = contract.render(esc, read_json(ROOT / "results" / TASK / "overlook.json"),
                                        read_json(ROOT / "results" / TASK / "paraphrase.json"),
                                        read_json(ROOT / "results" / TASK / "blind_spots.json"))
    except ImportError:
        contract_text = None
    overlook = read_json(ROOT / "results" / TASK / "overlook.json")
    para = read_json(ROOT / "results" / TASK / "paraphrase.json")
    run = {"fixture": NAME, "task": TASK, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "preliminary": True, "n_runs": 1, "budget_usd_each": BUDGET_EACH,
           "baselines": {"dev": dev_base, "heldout": held_base},
           "best_attempt": {"id": best_attempt["id"], "dev_score": best_attempt["dev_score"],
                            "heldout_score": heldout(best_attempt["code"], "best_attempt")},
           "escape": {"result": esc, "contract": contract_text, "candidate_code": esc.get("candidate"),
                      "dev_score": esc["evaluation"]["after"], "heldout_score": heldout(esc.get("candidate"), "escape"),
                      "cost_usd": round(esc_cost, 4), "calls": esc_calls},
           "fresh_start": {"prompt": fresh_prompt, "candidates": fresh,
                           "candidate_code": best_fresh["code"] if best_fresh else None,
                           "dev_score": best_fresh["dev_score"] if best_fresh else None,
                           "heldout_score": heldout(best_fresh["code"], "fresh") if best_fresh else None,
                           "cost_usd": round(fresh_session.ledger.new_spend, 4), "calls": K},
           "trajectory_cost_usd": round(traj_session.ledger.new_spend, 4),
           "overlook_excerpt": {"inputs": overlook["inputs"],
                                "assumptions": [a for a in overlook["assumptions"]["assumptions"] if a["shared"]]},
           "paraphrase": {k: para[k] for k in ("verdict", "mean_P", "max_R", "mean_R")} if para else None}
    (ROOT / "fixtures" / "runs").mkdir(parents=True, exist_ok=True)
    (ROOT / "fixtures" / "runs" / f"{NAME}.json").write_text(json.dumps(run, indent=2, default=str))
    print(json.dumps({"attempts": [(a["id"], a["dev_score"]) for a in attempts],
                      "loop_detected": esc.get("loop_detected"), "strategy_similarity": esc.get("strategy_similarity"),
                      "code_similarity": esc.get("code_similarity"), "action": esc.get("action"),
                      "lever": (esc.get("overlooked_levers") or [None])[0],
                      "best_attempt": run["best_attempt"], "escape": {k: run["escape"][k] for k in ("dev_score", "heldout_score", "cost_usd")},
                      "fresh_start": {k: run["fresh_start"][k] for k in ("dev_score", "heldout_score", "cost_usd")},
                      "baselines": run["baselines"], "trajectory_cost": run["trajectory_cost_usd"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
