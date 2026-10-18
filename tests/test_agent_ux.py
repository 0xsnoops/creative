"""What a coding agent sees from HERD's MCP server, offline: the operating manual, tool descriptions,
the free diverge_status check, structured verdicts and structured errors. No network, no API, $0."""
import json
import re
import sys

import anyio
import httpx
import pytest
from mcp import Client

import diverge.pipeline as pipeline
from diverge import demo
from diverge.contract import AUTONOMOUS, NEXT_ACTION, VERDICTS, agent_fields, agent_result, parse
from diverge.exa import Exa
from diverge.llm import LLM
from diverge.task import Task
from tests.test_contract import result as contract_result
from tests.test_escape_tool import BEST_FITS
from tests.test_pipeline_offline import FUNSEARCH, fake_world, reply_to  # noqa: F401

REAL_POST = LLM._post  # captured before any fixture replaces it
SECRETS = {"OPENROUTER_API_KEY": "sk-or-TESTSECRET", "EXA_API_KEY": "exa-TESTSECRET"}
BIN = "online_bin_packing"
TOOLS = {"herd_map", "blind_spots", "reality_check", "diverge_check", "diverge_escape", "diverge_status"}
ALIASES = {"herd_check": "diverge_check", "herd_escape": "diverge_escape", "herd_status": "diverge_status"}


def boom(*a, **k):
    raise AssertionError("network, model or search call during an offline test")


@pytest.fixture
def no_network(monkeypatch):
    for name in ("get", "post", "request", "stream"):
        monkeypatch.setattr(httpx, name, boom)
    monkeypatch.setattr(httpx.Client, "send", boom)


@pytest.fixture
def no_calls(no_network, monkeypatch):
    monkeypatch.setattr(LLM, "_post", boom)
    monkeypatch.setattr(Exa, "_post", boom)


def call(name, args=None):
    """(text, structuredContent) of one MCP tool call."""
    from diverge.server import mcp

    async def go():
        async with Client(mcp) as client:
            r = await client.call_tool(name, args or {})
            return r.content[0].text, r.structured_content
    return anyio.run(go)


def tools():
    from diverge.server import mcp

    async def go():
        async with Client(mcp) as client:
            return {t.name: t for t in (await client.list_tools()).tools}
    return anyio.run(go)


# ------------------------------------------------------------------ manual and descriptions
def test_instructions_are_an_operating_manual():
    from diverge.server import mcp
    text = mcp.instructions
    for phrase in ("not for generic debugging", "diverge_status", "diverge_check", "diverge_escape yourself",
                   "do not wait for the user to ask", "HYPOTHESIZED", "lower-level research primitives"):
        assert phrase.lower() in text.lower(), phrase


def test_every_tool_says_when_to_call_prerequisites_and_cost():
    found = tools()
    assert set(found) == TOOLS | set(ALIASES)
    for old, new in ALIASES.items():  # deprecated HERD names point to the DIVERGE tool
        assert found[old].description.startswith(f"Deprecated alias of {new}")
    for name, tool in found.items():
        if name in ALIASES:
            continue
        d = tool.description
        assert re.search(r"\bcall\b", d, re.I), name
        assert "$" in d, name  # a cost statement
        if name == "diverge_status":
            assert "$0" in d and "never its value" in d
            continue
        assert "OPENROUTER_API_KEY" in d, name
        default = tool.input_schema["properties"]["budget_usd"]["default"]
        assert f"Default budget_usd {default}" in d, name  # the stated default is the real one
        assert "Returns" in d, name
    for name in ("herd_map", "blind_spots", "diverge_check"):
        assert "Cold start" in found[name].description
    assert "Not for" in found["diverge_escape"].description and "without waiting to be asked" in found["diverge_escape"].description
    assert "evaluator" in found["diverge_escape"].description and "EXA" not in found["diverge_escape"].description


# ------------------------------------------------------------------ diverge_status
def test_herd_status_is_free_offline_and_never_shows_keys(fake_world, no_calls, monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    text, _ = call("diverge_status")
    for v in SECRETS.values():
        assert v not in text
    status = json.loads(text)
    assert status["spent_usd"] == 0 and fake_world == {"llm": 0, "exa": 0}
    assert status["ready"] and status["live_mode_available"] and status["missing"] == []
    assert status["requirements"]["openrouter"]["status"] == "configured"
    assert status["recommended_primary_tool"] == "diverge_escape" and status["when_to_call"]
    assert "model_catalog" not in status  # the live check is opt-in


def test_herd_status_missing_key_is_actionable(fake_world, no_calls, monkeypatch):
    for k in SECRETS:
        monkeypatch.delenv(k, raising=False)
    status = json.loads(call("diverge_status")[0])
    assert status["ready"] is False and status["missing"] == ["OPENROUTER_API_KEY"]
    assert status["requirements"]["exa"]["status"] == "missing (optional)"
    assert status["offline_demo_available"] and status["offline_demo_command"] == "python -m diverge demo strategy_loop"
    assert "only the user" in status["next_action"] and "never invent" in status["next_action"]
    assert "python -m diverge demo strategy_loop" in status["next_action"]


def test_herd_status_live_catalog_is_optional(fake_world, no_calls, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRETS["OPENROUTER_API_KEY"])
    status = json.loads(call("diverge_status", {"check_model_catalog": True})[0])
    assert status["model_catalog"]["optional"] is True and status["model_catalog"]["status"].startswith("unavailable")
    assert status["ready"]  # a failed optional check does not block


def test_readiness_reports_cached_tasks_and_low_budget(fake_world, no_calls):
    (pipeline.ROOT / "results" / BIN).mkdir(parents=True)
    (pipeline.ROOT / "results" / BIN / "herd_map.json").write_text("{}")
    r = demo.readiness(env={"OPENROUTER_API_KEY": "x"})
    assert r["cached_tasks"] == [BIN] and r["ready"]
    from types import SimpleNamespace

    import yaml
    from diverge.llm import Ledger
    cfg = yaml.safe_load((pipeline.ROOT / "config.yaml").read_text())
    low = demo.readiness(SimpleNamespace(ledger=Ledger(28.0, 27.5), cfg=cfg), env={"OPENROUTER_API_KEY": "x"})
    assert not low["ready"] and low["missing"][0].startswith("budget") and "budget_usd" in low["next_action"]


# ------------------------------------------------------------------ structured verdicts
def test_every_verdict_maps_to_a_next_action():
    assert set(NEXT_ACTION) == set(VERDICTS) == set(AUTONOMOUS)
    assert all(NEXT_ACTION[v] for v in VERDICTS)
    assert AUTONOMOUS["escalate"] is False and all(AUTONOMOUS[v] for v in VERDICTS if v != "escalate")
    cases = {"keep": contract_result(action="continue"), "repair": contract_result(action="repair"),
             "escape": contract_result(), "escalate": contract_result(levers=[]),
             "inconclusive": contract_result(scores=(0.04,))}
    for v, r in cases.items():
        text, s = agent_result(r)
        assert parse(text)[0] == v == s["verdict"]
        assert s["next_action"] and s["complete"] is True and s["can_continue_autonomously"] is AUTONOMOUS[v]
        assert set(s["evidence_summary"]) == {"MEASURED", "SOURCE_BACKED", "HYPOTHESIZED"}
    assert "get 1 more" in agent_result(cases["inconclusive"])[1]["next_action"]
    assert "get 2 more" in agent_fields({}, "inconclusive", {})["next_action"]


def test_herd_escape_returns_structured_verdict_and_next_action(fake_world, monkeypatch):
    pipeline.ensure_map(pipeline.Herd(progress=lambda m: None), Task.resolve(BIN))

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        text = f"```python\n{FUNSEARCH}```" if "Strategy to move away from" in prompt else reply_to(prompt)
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"cost": 0.001}}
    monkeypatch.setattr(LLM, "_post", post)
    text, s = call("diverge_escape", {"task_description": BIN, "current_solution": BEST_FITS[2],
                                   "previous_attempts": BEST_FITS[:2]})
    assert text.splitlines()[0] == "VERDICT: escape"
    v, body = parse(text)
    assert s == {"verdict": v, **body}
    assert s["next_action"] == NEXT_ACTION["escape"] and s["can_continue_autonomously"] is True
    assert s["complete"] is True and s["missing_requirements"] == [] and s["candidate_available"] is True
    assert s["lever"]["mechanism"] and s["evidence_summary"]["MEASURED"] > 0
    assert list(body)[0] == "next_action"  # read right after the VERDICT line


def test_herd_escape_without_evaluator_is_a_structured_error(fake_world):
    text, s = call("diverge_escape", {"task_description": "Design a deduplication service.",
                                   "current_solution": "def f(): pass\n"})
    assert text.startswith("VERDICT: inconclusive\n") and s["verdict"] == "inconclusive"
    assert s["error"] == "evaluator_required" and s["missing_requirements"] == ["evaluator_cmd"]
    assert s["complete"] is False and "evaluator_cmd" in s["next_action"] and s["spent_usd"] == 0


# ------------------------------------------------------------------ structured errors
def test_missing_key_is_a_structured_error_not_a_crash(fake_world, no_network, monkeypatch):
    monkeypatch.setattr(LLM, "_post", REAL_POST)  # the real client: it checks the key before any request
    for k in SECRETS:
        monkeypatch.delenv(k, raising=False)
    r = json.loads(call("herd_map", {"task_description": BIN})[0])
    assert r["error"] == "missing_requirement" and r["missing"] == ["OPENROUTER_API_KEY"]
    assert r["complete"] is False and r["live_calls_available"] is False and r["offline_demo_available"]
    assert "only the user" in r["next_action"] and "never invent" in r["next_action"]
    assert "python -m diverge demo strategy_loop" in r["next_action"] and r["spent_usd"] == 0


def test_budget_exceeded_is_structured_and_resumable(fake_world):
    r = json.loads(call("herd_map", {"task_description": BIN, "budget_usd": 0.0001})[0])
    assert r["error"] == "budget_exceeded" and r["missing"] == ["budget"] and r["complete"] is False
    assert "budget_usd" in r["next_action"] and "ask the user" in r["next_action"]


def test_unknown_family_lists_valid_names(fake_world):
    r = json.loads(call("reality_check", {"task_description": BIN, "family": "Nope", "k": 1})[0])
    assert r["error"] == "unknown_family" and "First Fit" in r["families"] and r["can_continue_autonomously"]


# ------------------------------------------------------------------ CLI: demo and preflight
def test_demo_strategy_loop_makes_no_network_call(no_calls, monkeypatch, capsys):
    monkeypatch.setattr(pipeline.Herd, "__init__", boom)
    from diverge.__main__ import main
    monkeypatch.setattr(sys, "argv", ["herd", "demo", "strategy_loop"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0 and "VERDICT:" in capsys.readouterr().out


@pytest.mark.parametrize("offline", [True, False])
def test_preflight_json(fake_world, no_calls, monkeypatch, capsys, offline):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    from diverge.__main__ import main
    monkeypatch.setattr(sys, "argv", ["herd", "preflight", "--json", *(["--offline"] if offline else [])])
    with pytest.raises(SystemExit) as e:
        main()
    out = capsys.readouterr().out
    assert all(v not in out for v in SECRETS.values())
    r = json.loads(out)
    assert e.value.code == 0 and r["ready"] and r["recommended_primary_tool"] == "diverge_escape"
    assert ("model_catalog" in r) is not offline


def test_herd_escape_missing_key_is_structured(fake_world, no_network, monkeypatch):
    pipeline.ensure_map(pipeline.Herd(progress=lambda m: None), Task.resolve(BIN))
    monkeypatch.setattr(LLM, "_post", REAL_POST)
    for k in SECRETS:
        monkeypatch.delenv(k, raising=False)
    text, s = call("diverge_escape", {"task_description": BIN, "current_solution": BEST_FITS[2],
                                   "previous_attempts": BEST_FITS[:2]})
    assert text.startswith("VERDICT: inconclusive\n") and parse(text)[1]["error"] == "missing_requirement"
    assert s["missing_requirements"] == ["OPENROUTER_API_KEY"] and s["can_continue_autonomously"] is False
    assert s["offline_demo_available"] and "only the user" in s["next_action"]
