import pytest

from herd.cache import JsonlCache, request_key
from herd.llm import BudgetExceeded, Ledger
from herd.sample import extract_code


def test_extract_code_picks_block_defining_entrypoint():
    text = ("Here you go:\n```python\nimport math\n```\nand\n```python\n"
            "def priority(item, capacities):\n    return capacities\n```\n")
    assert extract_code(text, "priority").startswith("def priority")
    assert extract_code("```\nclass Policy:\n    pass\n```", "Policy") == "class Policy:\n    pass\n"
    assert extract_code("no code here", "priority") is None
    assert extract_code("```python\ndef priority_helper(): pass\n```", "priority") is None


def test_cache_roundtrip(tmp_path):
    key = request_key({"model": "m", "sample_idx": 1})
    assert key != request_key({"model": "m", "sample_idx": 2})
    JsonlCache(tmp_path / "c.jsonl").put(key, {"cost_usd": 0.5, "response": {"x": 1}})
    reloaded = JsonlCache(tmp_path / "c.jsonl")
    assert reloaded.get(key)["response"] == {"x": 1}
    assert reloaded.total_cost() == 0.5


def test_ledger_refuses_calls_that_could_exceed_cap():
    ledger = Ledger(cap_usd=1.0, spent_usd=0.7)
    ledger.reserve(0.2)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.2)  # 0.7 spent + 0.2 in flight + 0.2 > 1.0
    ledger.settle(0.2, 0.05)
    ledger.reserve(0.2)
    assert ledger.spent == pytest.approx(0.75)
