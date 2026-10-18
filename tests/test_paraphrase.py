"""Offline tests for the paraphrase (consensus-robustness) test."""
import pytest

from diverge.paraphrase import analyze, check, distribution, tvd

PROMPT = "Write X.\n```python\ndef f(a: int) -> int: ...\n```\nItems are 1 to 100, 5,000 per instance.\n"


def test_check_flags_changed_code_and_numbers():
    assert check(PROMPT, PROMPT.replace("Write X.", "Please write X.")) == []
    assert "code blocks differ" in check(PROMPT, PROMPT.replace("a: int", "x: int"))
    assert any("numbers differ" in p for p in check(PROMPT, PROMPT.replace("5,000", "5000")))


def test_tvd_and_emergent_bucketing():
    p = distribution(["A", "A", "B", "emergent:x", "unlabeled"])
    q = distribution(["A", "B", "B", "emergent:y"])
    assert p == {"A": 0.5, "B": 0.25, "emergent": 0.25}
    assert tvd(p, q) == pytest.approx(0.25)
    assert tvd(p, p) == 0


def test_decision_rule():
    same = [["A"] * 7 + ["B"] * 3] * 4
    robust = analyze(same, [["A"] * 7 + ["B"] * 3] * 3)
    assert robust["verdict"] == "robust to wording" and robust["mean_P"] == 0
    shifted = analyze(same, [["C"] * 10] * 3)
    assert shifted["verdict"] == "partly a prompt effect" and shifted["mean_P"] == 1.0
