"""Applies docs/PREREGISTRATION.md step 2 to the actual map spend, checks the stop rules, then
runs both benchmarks. Budgets depend only on map costs, never on benchmark outcomes.

    HERD_TRACE_DIR=results/traces/phase2-20261018 uv run python scripts/run_benches.py

Completed calls are cached, so rerunning resumes where a stopped run left off."""
import json, math, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diverge.pipeline import Herd

CAP, RESERVE, BATCH, LABEL_PER_BATCH, EXPECTED_CANDIDATES = 25.0, 1.00, 8, 0.10, 8
TASKS = {"cache_eviction": (0.65, ["plain", "adhd_frames", "blind_spot", "blind_spot_contrast",
                                   "blind_spot_contrast_random", "blind_spot_feedback", "blind_spot_feedback_sham",
                                   "literature_all", "literature_random"]),
         "online_bin_packing": (0.35, ["blind_spot", "blind_spot_contrast", "blind_spot_feedback", "adhd_frames",
                                       "literature_random"])}
spent = Herd().ledger.total_spent
maps = spent - 0.50  # smoke test excluded
plan = {"spent_after_maps": round(spent, 4), "maps_cost": round(maps, 4)}
if maps > 10:
    plan["stop"] = "maps cost more than $10 (stop rule)"
remaining = CAP - spent - RESERVE
projected = spent
for task, (share, ops) in TASKS.items():
    allowance = LABEL_PER_BATCH * math.ceil(EXPECTED_CANDIDATES * len(ops) * 3 / BATCH)
    per_op = math.floor((share * remaining - allowance) / (len(ops) * 3) * 100) / 100
    plan[task] = {"operators": ops, "budget_per_operator": per_op, "labeling_allowance": allowance}
    projected += per_op * len(ops) * 3 + allowance
plan["projected_total"] = round(projected, 2)
if projected > CAP:
    plan["stop"] = f"projected total ${projected:.2f} exceeds ${CAP}"
print(json.dumps(plan, indent=2), flush=True)
Path("results/phase2").mkdir(parents=True, exist_ok=True)
json.dump(plan, open("results/phase2/bench_plan.json", "w"), indent=2)
if "stop" in plan:
    sys.exit("STOP: " + plan["stop"])
for task in TASKS:
    p = plan[task]
    cmd = [sys.executable, "-m", "diverge", "bench", task, "--budget", str(p["budget_per_operator"]),
           "--repeats", "3", "--n", "20", "--operators", ",".join(p["operators"])]
    print("running:", " ".join(cmd[1:]), flush=True)
    with open(f"results/phase2/bench_{task}.log", "w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"{task} exited {code}; total spend now ${Herd().ledger.total_spent:.4f}", flush=True)
