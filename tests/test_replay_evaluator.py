"""examples/replay_evaluator.py works as an evaluator_cmd: run through DIVERGE's Evaluator (isolated,
empty environment, scratch working directory), with the data path passed as a flag."""
import sys
from pathlib import Path

from diverge.reality import Evaluator
from diverge.task import Task

ROOT = Path(__file__).resolve().parent.parent
CMD = f"{sys.executable} {ROOT / 'examples' / 'replay_evaluator.py'} --data {ROOT / 'examples' / 'replay_sample.jsonl'}"


def score(tmp_path, code):
    path = tmp_path / "candidate.py"
    path.write_text(code)
    return Evaluator(Task.resolve("Predict the expected value from recorded inputs."), CMD)(path)


def test_replay_evaluator_scores_valid_and_invalid_candidates(tmp_path):
    median = score(tmp_path, "def predict(x):\n    return sorted(x)[len(x) // 2]\n")
    mean = score(tmp_path, "def predict(x):\n    return sum(x) / len(x)\n")
    assert median["valid"] and median["score"] == 0.0 and median["detail"]["n_records"] == 4
    assert mean["valid"] and mean["score"] > 0
    broken = score(tmp_path, "def predict(x):\n    return x['nope']\n")
    assert broken["valid"] is False and "TypeError" in broken["detail"]["error"]
    missing = score(tmp_path, "def other(x):\n    return 0\n")
    assert missing["valid"] is False and "predict" in missing["detail"]["error"]
