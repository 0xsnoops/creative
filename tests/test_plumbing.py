import pytest

from diverge.cache import JsonlCache, request_key
from diverge.llm import BudgetExceeded, Ledger
from diverge.sample import extract_code


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
    ledger = Ledger(global_cap_usd=1.0, global_spent_usd=0.7)
    ledger.reserve(0.2)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.2)  # 0.7 spent + 0.2 in flight + 0.2 > 1.0
    ledger.settle(0.2, 0.05)
    ledger.reserve(0.2)
    assert ledger.total_spent == pytest.approx(0.75)


def test_run_budget_is_measured_from_zero_not_from_cache_total():
    # $9 already spent overall must not shrink a $2 run budget under a $15 global cap.
    ledger = Ledger(global_cap_usd=15.0, global_spent_usd=9.0, run_cap_usd=2.0)
    ledger.reserve(1.5)
    ledger.settle(1.5, 1.5)
    with pytest.raises(BudgetExceeded, match="run budget"):
        ledger.reserve(0.6)
    ledger.reserve(0.5)


def test_billed_cache_hits_count_toward_run_but_not_global():
    ledger = Ledger(global_cap_usd=1.0, global_spent_usd=0.95, run_cap_usd=2.0, bill_cached=True)
    ledger.charge_cached(worst_case=0.3, cost=0.3)  # already inside the 0.95: no global check
    assert ledger.spent_this_run == pytest.approx(0.3) and ledger.total_spent == pytest.approx(0.95)
    with pytest.raises(BudgetExceeded, match="run budget"):
        ledger.charge_cached(worst_case=1.8, cost=0.1)


def test_cache_skips_a_damaged_line(tmp_path):
    path = tmp_path / "c.jsonl"
    cache = JsonlCache(path)
    cache.put("a", {"cost_usd": 0.1})
    with path.open("a") as f:
        f.write('{"key": "half-written", "cost_u\n')
    cache.put("b", {"cost_usd": 0.2})
    reloaded = JsonlCache(path)
    assert set(reloaded.entries) == {"a", "b"} and reloaded.skipped_lines == 1


def test_chat_shrinks_max_tokens_to_fit_the_run_cap_and_never_exceeds_it(tmp_path):
    """A call whose full-size worst case no longer fits the run budget runs with fewer max_tokens
    (>= MIN_SHRUNK_TOKENS) instead of ending the run; the reservation stays within the cap."""
    from diverge.llm import LLM, MIN_SHRUNK_TOKENS
    catalog = {"m": {"pricing": {"prompt": "0.000001", "completion": "0.00001"}, "top_provider": {}}}
    ledger = Ledger(global_cap_usd=100.0, global_spent_usd=0.0, run_cap_usd=0.30)
    llm = LLM(JsonlCache(tmp_path / "c.jsonl"), catalog, ledger)
    sent = []

    def post(body, attempts=4):
        sent.append(body["max_tokens"])
        assert ledger.spent_this_run + ledger.reserved <= 0.30 + 1e-9
        # a reply never costs more than its max_tokens allow (here: 8000 tokens, or fewer)
        return {"choices": [{"message": {"content": "x"}}], "usage": {"cost": min(8000, body["max_tokens"]) * 1e-5}}
    llm._post = post
    calls = 0
    with pytest.raises(BudgetExceeded):
        while calls < 10:  # full-size worst case: 16000 * 1e-5 * 1.25 = $0.20
            llm.chat("m", [{"role": "user", "content": "hi"}], max_tokens=16000, sample_idx=calls)
            calls += 1
    assert sent[:2] == [16000, 16000] and all(MIN_SHRUNK_TOKENS <= t < 16000 for t in sent[2:])
    assert len(sent) >= 3  # the old rule stopped after 2 calls ($0.16 spent of $0.30)
    assert ledger.spent_this_run <= 0.30
