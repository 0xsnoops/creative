"""Re-score held-out for an existing rescue round, one program at a time, with the task's normal
60 s limit. Used when the original scoring ran on an overloaded machine (load average ~120 on
2026-10-18) and wall-clock timeouts marked programs invalid. Candidates and their dev selection
are untouched: only the held-out number of the same code is recomputed. No model calls.

    uv run python scripts/rescore_rescue.py results/rescue/arms-r2.json [more arms files]
Writes <file>.rescored.json and prints the per-arm rescued counts.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from diverge.reality import Evaluator  # noqa: E402
from diverge.task import Task  # noqa: E402

OUT = ROOT / "results" / "rescue" / "rescore"


def heldout(evaluator, code):
    if not code:
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{hashlib.sha256(code.encode()).hexdigest()[:12]}.py"
    path.write_text(code)
    return evaluator.final(path, None)[0]


def main():
    evaluator = Evaluator(Task.resolve("online_bin_packing"))
    trajs = json.loads((ROOT / "results" / "rescue" / "trajectories.json").read_text())
    counts = {}
    for name in sys.argv[1:]:
        arms = json.loads(Path(name).read_text())
        for key, state in arms.items():
            attempts = [a for a in trajs[key]["attempts"] if a["valid"]]
            best = min(attempts, key=lambda a: a["dev_score"])
            stuck = heldout(evaluator, best["code"])
            state["stuck_best"]["heldout_score_rescored"] = stuck
            for arm in state["arms"]:
                score = heldout(evaluator, arm.get("code"))
                arm["heldout_score_rescored"] = score
                arm["rescued_rescored"] = score is not None and stuck is not None and score < stuck
                c = counts.setdefault(arm["arm"], {"rescued": 0, "n": 0})
                c["rescued"] += arm["rescued_rescored"]
                c["n"] += 1
                print(f"state {key} {arm['arm']:16} held-out {score} vs stuck {stuck} "
                      f"rescued={arm['rescued_rescored']}", flush=True)
        Path(name + ".rescored.json").write_text(json.dumps(arms, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
