"""Offline: (re)render the herd_escape contract stored in fixtures/runs/<name>.json, adding the
attempts' recorded dev scores (as the MCP tool's RecordingEvaluator would). No API calls.

    uv run python scripts/render_run_contract.py strategy_loop"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from herd import contract  # noqa: E402
from herd.pipeline import read_json  # noqa: E402

name = sys.argv[1]
run_path, fix_path = ROOT / "fixtures" / "runs" / f"{name}.json", ROOT / "fixtures" / "trajectories" / f"{name}.json"
run, fixture = json.loads(run_path.read_text()), json.loads(fix_path.read_text())
result = run["escape"]["result"]
result["attempts"] = [{"id": a["id"], "score": a["dev_score"], "valid": a["dev_score"] is not None}
                      for a in fixture["attempts"]]
result["lower_is_better"] = True
out = ROOT / "results" / run["task"]
families = (read_json(out / "blind_spots.json") or []) + (read_json(out / "families.json") or [])
run["escape"]["contract"] = contract.render(result, read_json(out / "overlook.json"), read_json(out / "paraphrase.json"),
                                            families)
run_path.write_text(json.dumps(run, indent=2, default=str))
print(run["escape"]["contract"])
