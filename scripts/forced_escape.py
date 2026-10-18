"""Ablation for a recorded run: implement HERD's chosen lever even though HERD's verdict was
not "escape", at the same k and budget as fresh_start, and record it as forced_escape in
fixtures/runs/<name>.json. This answers "would the lever have helped?"; it is NOT HERD's
recommendation. Dev selection, held-out reporting.

    HERD_TRACE_DIR=results/traces/demo-20261018 uv run python scripts/forced_escape.py strategy_loop_plateau"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from diverge.escape import generate  # noqa: E402
from diverge.pipeline import Herd  # noqa: E402
from diverge.reality import Evaluator  # noqa: E402
from diverge.task import Task  # noqa: E402

name = sys.argv[1]
run_path = ROOT / "fixtures" / "runs" / f"{name}.json"
run = json.loads(run_path.read_text())
fixture = json.loads((ROOT / "fixtures" / "trajectories" / f"{name}.json").read_text())
task, result = Task.resolve(run["task"]), run["escape"]["result"]
evaluator = Evaluator(task)
lever = result["overlooked_levers"][0]
choice = {"lever": {k: lever[k] for k in ("type", "name", "detail")}, "mechanism": lever["mechanism"],
          "why": lever["why"], "evidence": result["diagnosis"].get("evidence", {}), "alternatives": []}
herd = Herd(budget_usd=run["budget_usd_each"])
gen = generate(herd, task, choice, result["diagnosis"], fixture["attempts"][-1]["code"], evaluator,
               k=run["fresh_start"]["calls"], seed=1)
best = gen["best"]
held = None
if best:
    held = evaluator.final(Path(ROOT / best["path"]) if not Path(best["path"]).is_absolute() else Path(best["path"]),
                           best["score"])[0]
run["forced_escape"] = {"note": f"ablation: HERD's verdict was not escape; the lever was implemented anyway",
                        "lever": lever["name"], "candidates": [{k: c.get(k) for k in ("valid", "score", "error")}
                                                               for c in gen["candidates"]],
                        "candidate_code": best["code"] if best else None, "dev_score": best["score"] if best else None,
                        "heldout_score": held, "cost_usd": round(herd.ledger.new_spend, 4),
                        "calls": run["fresh_start"]["calls"], "prompt": gen["prompt"]}
run_path.write_text(json.dumps(run, indent=2, default=str))
print(json.dumps({k: v for k, v in run["forced_escape"].items() if k not in ("candidate_code", "prompt")}, indent=2))
