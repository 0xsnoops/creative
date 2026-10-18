"""DIVERGE_PROVIDER=anthropic: calls go to the Anthropic Messages API, never OpenRouter."""
import pytest

from diverge import llm
from diverge.cache import JsonlCache
from diverge.llm import LLM, Ledger, anthropic_model_id, anthropic_only_models


def test_model_ids_and_role_mapping():
    assert anthropic_model_id("anthropic/claude-haiku-5.5") == "claude-haiku-5-5"
    assert anthropic_model_id("google/gemini-4-argon") is None
    m = anthropic_only_models({"samplers": ["openai/x", "anthropic/claude-opus-5.5"], "implementer": "google/g",
                               "extractor": "anthropic/claude-haiku-5.5", "labeler": "anthropic/claude-opus-5.5"})
    assert m["samplers"] == ["openai/x", "anthropic/claude-opus-5.5"]  # unchanged: keys the cached maps
    assert m["implementer"].startswith("anthropic/")
    assert all(str(v).startswith("anthropic/") for k, v in m.items() if k != "samplers")


def test_anthropic_provider_posts_to_anthropic_only(monkeypatch, tmp_path):
    monkeypatch.setenv("DIVERGE_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TESTSECRET")
    seen = []

    class R:
        status_code = 200

        def json(self):
            return {"content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1000, "output_tokens": 100}, "model": "claude-haiku-5-5"}

    def post(url, headers, json, timeout):
        seen.append((url, headers["x-api-key"], json))
        return R()
    monkeypatch.setattr(llm.httpx, "post", post)
    catalog = {"anthropic/claude-haiku-5.5": {"pricing": {"prompt": "0.000001", "completion": "0.000005"}, "top_provider": {}},
               "google/g": {"pricing": {"prompt": "0", "completion": "0"}, "top_provider": {}}}
    client = LLM(JsonlCache(tmp_path / "c.jsonl"), catalog, Ledger(10.0, 0.0))
    entry = client.chat("anthropic/claude-haiku-5.5", [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                        max_tokens=50)
    assert llm.message_text(entry) == "hello" and entry["cost_usd"] == pytest.approx(1000e-6 + 100 * 5e-6)
    url, key, body = seen[0]
    assert url == llm.ANTHROPIC and "openrouter" not in url and key == "sk-ant-TESTSECRET"
    assert body["model"] == "claude-haiku-5-5" and body["system"] == "s" and body["messages"] == [{"role": "user", "content": "u"}]
    with pytest.raises(RuntimeError, match="only serves anthropic"):
        client.chat("google/g", [{"role": "user", "content": "u"}], max_tokens=50)
    assert len(seen) == 1  # the non-Anthropic model never reached any network call
