"""Offline: re-run loop detection on the 3 recorded live runs with the behavior detector
(herd.trajectory, detector "behavior") and compare with the recorded HERD verdicts.

Detector fix found from run 3, validated by E1. No API calls: extract=None (no LLM captions);
dev scoring and behavior fingerprints run locally in the sandbox (CPU only). Writes
results/redetect/<name>.json and prints a table.

    uv run python scripts/redetect_fixtures.py [name ...]"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from herd import contract, trajectory  # noqa: E402
from herd.diagnose import diagnose  # noqa: E402
from herd.escape import choose  # noqa: E402
from herd.pipeline import read_json  # noqa: E402
from herd.reality import Evaluator  # noqa: E402
from herd.task import Task  # noqa: E402

LABEL = "detector fix found from run 3, validated by E1"
NAMES = ["strategy_loop", "strategy_loop_tune", "strategy_loop_unconstrained"]


def old_verdict(run):
    esc = run.get("escape") or {}
    first = str(esc.get("contract") or "").splitlines()[:1]
    if first and first[0].startswith("VERDICT: "):
        return first[0].removeprefix("VERDICT: ").strip(), (esc.get("result") or {}).get("action")
    action = (esc.get("result") or {}).get("action")
    return {"continue": "keep", "repair": "repair", "switch_strategy": "escape"}.get(action), action


def redetect(name):
    fixture = json.loads((ROOT / "fixtures" / "trajectories" / f"{name}.json").read_text())
    run = json.loads((ROOT / "fixtures" / "runs" / f"{name}.json").read_text())
    task = Task.resolve(fixture["task"])
    evaluator = Evaluator(task)
    attempts = [{"id": a["id"], "code": a["code"], "score": a.get("dev_score"), "dev_detail": None}
                for a in fixture["attempts"]]
    report = trajectory.analyze(task, attempts, extract=None, lower_is_better=evaluator.lower_is_better,
                                evaluator=evaluator)
    out = ROOT / "results" / task.name
    overlook = read_json(out / "overlook.json")
    families = (read_json(out / "blind_spots.json") or []) + (read_json(out / "families.json") or [])
    diagnosis = diagnose(task, report, overlook)
    choice = choose(diagnosis, report, overlook, families)
    lever = [{k: choice["lever"][k] for k in ("type", "name", "detail")}
             | {"mechanism": choice["mechanism"], "why": choice["why"]}] if choice["lever"] else []
    result = {"loop_detected": report["loop_detected"], "strategy_similarity": report["strategy_similarity"],
              "code_similarity": report["code_similarity"], "action": diagnosis["action"], "diagnosis": diagnosis,
              "overlooked_levers": lever + choice["alternatives"],
              "recommended_switch": {"mechanism": choice["mechanism"], "why": choice["why"]},
              "attempts": [{"id": a["id"], "score": a["score"], "valid": a["score"] is not None} for a in attempts],
              "evaluation": {"before": attempts[-1]["score"], "after": None, "improved": False, "split": "dev"},
              "candidates": [], "lower_is_better": evaluator.lower_is_better}
    old, old_action = old_verdict(run)
    b = report["behavior"] or {}
    return {"label": LABEL, "fixture": name, "task": task.name, "detector": report["detector"],
            "old": {"verdict": old, "action": old_action,
                    "loop_detected": (run["escape"].get("result") or {}).get("loop_detected"),
                    "strategy_similarity": (run["escape"].get("result") or {}).get("strategy_similarity"),
                    "code_similarity": (run["escape"].get("result") or {}).get("code_similarity")},
            "new": {"verdict": contract.verdict(result), "action": diagnosis["action"],
                    "loop_detected": report["loop_detected"], "lever": (lever[0]["name"] if lever else None)},
            "conditions": b.get("conditions"),
            "behavior": {k: b.get(k) for k in ("clusters", "cluster_of_recent", "max_probe_disagreement",
                                               "failure_sets", "failure_jaccard", "improvement_last3", "epsilon",
                                               "scores")},
            "similarity": {"strategy": report["strategy_similarity"], "code": report["code_similarity"]},
            "reasons": diagnosis["reasons"], "api_calls": 0}


def main(names):
    dest = ROOT / "results" / "redetect"
    dest.mkdir(parents=True, exist_ok=True)
    print(LABEL)
    print(f"{'fixture':30} {'old':8} {'new':8} {'a':6} {'b':6} {'c':6} {'impr3':>8} {'jacc':>5} {'strat':>5} {'code':>5}")
    for name in names:
        r = redetect(name)
        (dest / f"{name}.json").write_text(json.dumps(r, indent=2, default=str))
        c, b = r["conditions"] or {}, r["behavior"]
        jmin = (b.get("failure_jaccard") or {}).get("min")
        imp = b.get("improvement_last3")
        fmt = lambda x, f: format(x, f) if isinstance(x, (int, float)) else "-"
        print(f"{name:30} {r['old']['verdict'] or '-':8} {r['new']['verdict']:8} {str(c.get('a')):6} "
              f"{str(c.get('b')):6} {str(c.get('c')):6} {fmt(imp, '+.1%'):>8} {fmt(jmin, '.2f'):>5} "
              f"{fmt(r['similarity']['strategy'], '.2f'):>5} {fmt(r['similarity']['code'], '.2f'):>5}")
    print(f"wrote {dest}/<name>.json")


if __name__ == "__main__":
    main(sys.argv[1:] or NAMES)
