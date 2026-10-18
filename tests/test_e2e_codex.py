"""Independent offline acceptance checks (Codex 3 QA, adopted into the main suite).

Reuse v3's fake models/trajectory/admission, but run real dev evaluators. MCP checks
exercise the public tools. CLI checks run actual subprocess entry points with every key
removed and the env file pointed at an absent path. No credentials or network are needed.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

import diverge.pipeline as pipeline
from diverge import llm
from diverge.cache import JsonlCache
from diverge.contract import parse
from diverge.escape import (
    ANCHOR_MARKER, CRITIC_MARKER, DEEPEN_MARKER, DEFAULT_MODE,
    IMPLEMENT3_MARKER, MODES, SKETCH3_MARKER, run_escape,
)
from diverge.llm import LLM, Ledger, anthropic_only_models
from diverge.reality import Evaluator
from diverge.task import Task
from tests.test_agent_ux import call
from tests.test_check_freetext import ALTS, FREE
from tests.test_escape_tool import BEST_FITS
from tests.test_escape_v2 import ATTEMPTS, CURRENT, code_lines, overlook
from tests.test_escape_v3 import v2_world, world  # noqa: F401 (fixture dependency chain)
from tests.test_pipeline_offline import FUNSEARCH, fake_world, reply_to  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
CLAUDE = "anthropic/claude-haiku-5.5"


@pytest.fixture(autouse=True)
def offline_policy(monkeypatch):
    """Fail closed on unfaked HTTP; disable implicit environment-file loading."""
    for name in tuple(os.environ):
        if name.endswith("_KEY") or name in ("DIVERGE_ENV_FILE", "HERD_TRACE_DIR"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DIVERGE_PROVIDER", "openrouter")
    monkeypatch.setattr(pipeline, "load_env", lambda: None)
    original_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda path: False if
                        isinstance(path, (str, bytes, os.PathLike)) and
                        os.path.basename(os.fsdecode(path)).startswith(".env")
                        else original_exists(path))

    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected unfaked HTTP request in an offline acceptance test")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


@pytest.mark.parametrize("fallback,rejections", [(False, 0), (True, 1)])
def test_escape_v3_end_to_end_and_prompt_privacy(world, monkeypatch, fallback, rejections):
    session, task, state = world
    assert DEFAULT_MODE == "v2" and {"v1", "v2", "v3"} <= set(MODES)
    if fallback:
        state["anchor"] = None
    state["reject"] = rejections
    attempts = copy.deepcopy([*ATTEMPTS, CURRENT])
    forbidden = list(code_lines())
    for i, attempt in enumerate(attempts):
        code_marker = f"CODEX_PRIVATE_ATTEMPT_{i}_7df921"
        held_marker = f"CODEX_HELDOUT_RESULT_{i}_b812ea"
        attempt["code"] += f"\n# {code_marker}\n"
        attempt["heldout_score"] = 98765.4321 + i
        attempt["heldout_results"] = {"case": held_marker, "score": attempt["heldout_score"]}
        forbidden.extend((code_marker, held_marker, str(attempt["heldout_score"])))

    # Capture every outbound message, not just the sketch prompts or first message.
    messages = []
    fake_post = LLM._post

    def recording_post(self, body, attempts=4):
        messages.extend(copy.deepcopy(body["messages"]))
        return fake_post(self, body, attempts)

    monkeypatch.setattr(LLM, "_post", recording_post)
    splits = []
    real_evaluate = task.evaluate

    def dev_only(path, split="dev"):
        splits.append(split)
        assert split == "dev", "held-out evaluation during generation/selection"
        return real_evaluate(path, split)

    monkeypatch.setattr(task, "evaluate", dev_only)
    result = run_escape("v3", session, task, attempts[-1], attempts[:-1], Evaluator(task),
                        overlook=overlook(["s-000"]), k=7)
    assert result["version"] == 3
    assert result["diagnosis"]["action"] == "switch_strategy"
    assert result["anchor_strip"]["fallback"] is fallback
    scores = result["critic"]["scores"]
    assert len(scores) == len(result["sketches"]) == 5
    assert all({"viability", "fit", "distance", "total"} <= s.keys() for s in scores)
    assert [s["index"] for s in result["deepened"]] == [3, 2]
    assert 0 < len(result["deepened"]) <= 2
    assert result["evaluation"]["split"] == "dev" and result["evaluation"]["improved"]
    assert result["evaluation"]["after"] < result["evaluation"]["before"]
    assert Path(result["candidate_path"]).read_text() == result["candidate"]
    assert splits and set(splits) == {"dev"}
    prompts = [m["content"] for m in messages]
    for marker in (ANCHOR_MARKER, SKETCH3_MARKER, CRITIC_MARKER, DEEPEN_MARKER, IMPLEMENT3_MARKER):
        assert any(marker in prompt for prompt in prompts), marker
    assert sum(DEEPEN_MARKER in p for p in prompts) == 2
    for prompt in prompts:
        assert not [secret for secret in forbidden if secret in prompt]
    if rejections:
        assert any(r["stage"] == "admission" for r in result["rejections"])


def make_client(tmp_path, *models):
    catalog = {m: {"pricing": {"prompt": "0.000001", "completion": "0.000005"},
                   "top_provider": {}} for m in models}
    return LLM(JsonlCache(tmp_path / "responses.jsonl"), catalog, Ledger(10.0, 0.0))


@pytest.fixture
def anthropic_transport(monkeypatch):
    monkeypatch.setenv("DIVERGE_PROVIDER", "anthropic")
    # A fixed noncredential sentinel exists only inside this fake transport test.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-placeholder")
    seen = []

    def post(url, *, headers, json, timeout):
        seen.append((url, copy.deepcopy(json)))
        return SimpleNamespace(status_code=200, json=lambda: {
            "content": [{"type": "text", "text": "offline response"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 100, "output_tokens": 20},
            "model": json["model"],
        })

    monkeypatch.setattr(httpx, "post", post)
    return seen


def test_anthropic_chat_uses_only_messages_endpoint_without_temperature(tmp_path, anthropic_transport):
    client = make_client(tmp_path, CLAUDE)
    messages = [{"role": "system", "content": "offline system"}, {"role": "user", "content": "offline user"}]
    for temperature in (0.0, 1.0, 2.0):
        entry = client.chat(CLAUDE, messages, temperature=temperature, max_tokens=50)
        assert llm.message_text(entry) == "offline response"
        assert entry["response"]["provider"] == "anthropic"
        assert entry["cost_usd"] == pytest.approx(0.0002)
    assert len(anthropic_transport) == 3
    for url, body in anthropic_transport:
        assert url == "https://api.anthropic.com/v1/messages"
        assert "temperature" not in body
        assert body["model"] == "claude-haiku-5-5"
        assert body["system"] == "offline system" and body["messages"] == messages[1:]


@pytest.mark.parametrize("model", ["google/gemini-test", "openai/gpt-test", "anthropic/not-a-claude-model"])
def test_anthropic_rejects_non_claude_before_network(tmp_path, anthropic_transport, model):
    client = make_client(tmp_path, model)
    error = None
    try:
        client.chat(model, [{"role": "user", "content": "offline"}], max_tokens=50)
    except (RuntimeError, ValueError) as exc:
        error = exc
    assert not anthropic_transport, f"non-Claude model reached HTTP transport: {model}"
    assert error is not None, "non-Claude model must raise"
    assert client.ledger.reserved == 0


def test_anthropic_role_conversion_preserves_sampler_order_and_input():
    models = {"samplers": ["openai/test", CLAUDE, "google/test", "openai/test"],
              "implementer": "google/test", "extractor": CLAUDE, "labeler": "openai/test"}
    original = copy.deepcopy(models)
    converted = anthropic_only_models(models)
    assert models == original
    assert converted["samplers"] == original["samplers"]
    assert converted["extractor"] == CLAUDE
    assert all(converted[role].startswith("anthropic/claude-") for role in ("implementer", "extractor", "labeler"))


def test_anthropic_cold_start_does_not_contact_openrouter(fake_world, monkeypatch):
    monkeypatch.setenv("DIVERGE_PROVIDER", "anthropic")
    catalog_path = pipeline.ROOT / "cache" / "openrouter_models.json"
    catalog = catalog_path.read_text()
    catalog_path.unlink()
    seen = []

    def get(url, **kwargs):
        seen.append(url)
        return httpx.Response(200, text=catalog, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    pipeline.Session(progress=lambda message: None)
    assert not seen, f"Anthropic-only cold initialization contacted another provider: {seen}"


def assert_readable(text, structured, tool):
    head, _, rest = text.partition("\n")
    markdown, separator, tail = rest.partition("\n{")
    assert head == f"VERDICT: {structured['verdict']}"
    assert separator and markdown.startswith(f"## DIVERGE {tool}:")
    assert "**" in markdown and structured["next_action"]
    body = json.loads("{" + tail)
    assert parse(text) == (structured["verdict"], body)
    assert {"verdict": structured["verdict"], **body} == structured


def test_escape_mcp_readable_success(fake_world, monkeypatch):
    pipeline.ensure_map(pipeline.Session(progress=lambda message: None), Task.resolve("online_bin_packing"))

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        answer = f"```python\n{FUNSEARCH}```" if "Strategy to move away from" in prompt else reply_to(prompt)
        return {"choices": [{"message": {"content": answer}, "finish_reason": "stop"}], "usage": {"cost": 0.001}}

    monkeypatch.setattr(LLM, "_post", post)
    text, structured = call("diverge_escape", {"task_description": "online_bin_packing",
                                              "current_solution": BEST_FITS[-1], "previous_attempts": BEST_FITS[:-1]})
    assert_readable(text, structured, "escape")
    assert structured["verdict"] == "escape" and structured["candidate_available"]
    assert structured["next_experiment"]["measured"]["improved"]


def test_check_free_text_readable_and_no_sampling(fake_world, monkeypatch):
    prompts = []

    def post(self, body, attempts=4):
        prompts.append(body["messages"][0]["content"])
        assert prompts[-1].startswith("You are proposing alternative")
        return {"choices": [{"message": {"content": json.dumps({"alternatives": ALTS})}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001}}

    def no_sampling(*args, **kwargs):
        pytest.fail("free-text check attempted sampling or a map build")

    monkeypatch.setattr(LLM, "_post", post)
    monkeypatch.setattr(pipeline, "run_sampling", no_sampling)
    monkeypatch.setattr(pipeline, "ensure_map", no_sampling)
    task = Task.resolve(FREE)
    assert not (pipeline.ROOT / "results" / task.name / "herd_map.json").exists()
    text, structured = call("diverge_check", {"task_description": FREE, "my_approach": "first fit"})
    assert_readable(text, structured, "check")
    assert structured["verdict"] == "untested" and structured["herd_comparison"] is None
    assert len(prompts) == 1 and fake_world["exa"] == 0
    assert len(structured["alternatives"]) == 4
    assert all(a["evidence_class"] == "HYPOTHESIZED" for a in structured["alternatives"])
    assert not (pipeline.ROOT / "results" / task.name / "herd_map.json").exists()
    assert not (pipeline.ROOT / "results" / task.name / "samples").exists()


@pytest.mark.parametrize("length", [300, 10_000])
@pytest.mark.parametrize("unit", ["x", "pack bins ", "容量"])
def test_long_free_text_resolves_verbatim(length, unit):
    description = (unit * length)[:length]
    task = Task.resolve(description)
    assert task.prompt == description.strip() and task.evaluator is None
    assert task.name.startswith("desc-") and len(task.name) == 17
    assert Task.resolve(description).name == task.name
    assert Task.resolve(description + "different").name != task.name


def run_cli_without_keys(tmp_path, *arguments):
    """Run the `diverge` CLI in a subprocess with every *_KEY removed and the private env file pointed
    at an absent path. Keys from the repo's own .env would make preflight "ready", so a small in-process
    shim (no sitecustomize) wraps load_env and drops any *_KEY it added before the command runs."""
    env = {k: v for k, v in os.environ.items() if not k.endswith("_KEY") and k not in
           ("HERD_TRACE_DIR", "DIVERGE_TRACE_DIR", "PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS")}
    env.update(DIVERGE_ENV_FILE=str(tmp_path / "absent.env"), PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(tmp_path))
    shim = ("import os, sys, runpy\n"
            "import diverge.pipeline as p\n"
            "_load = p.load_env\n"
            "def _no_dotenv_keys():\n"
            "    before = {k for k in os.environ if k.endswith('_KEY')}\n"
            "    _load()\n"
            "    for k in [k for k in os.environ if k.endswith('_KEY') and k not in before]:\n"
            "        del os.environ[k]\n"
            "p.load_env = _no_dotenv_keys\n"
            "sys.argv = ['diverge', *sys.argv[1:]]\n"
            "runpy.run_module('diverge', run_name='__main__', alter_sys=True)\n")
    return subprocess.run([sys.executable, "-c", shim, *arguments], cwd=REPO,
                          env=env, capture_output=True, text=True, timeout=60)


def test_cli_demo_offline_without_keys_exits_zero(tmp_path):
    proc = run_cli_without_keys(tmp_path, "demo", "strategy_loop")
    assert "DIVERGE demo: strategy_loop" in proc.stdout and "VERDICT:" in proc.stdout
    assert proc.returncode == 0, f"exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"


def test_cli_preflight_offline_without_keys_reports_not_ready(tmp_path):
    """By design the exit code is live readiness (also with --offline); the JSON says the offline demo
    works and what to do next."""
    proc = run_cli_without_keys(tmp_path, "preflight", "--json", "--offline")
    assert proc.returncode == 1, f"exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    result = json.loads(proc.stdout)
    assert result["ready"] is False
    assert result["offline_demo_available"] is True
    assert result["live_mode_available"] is False
    assert result.get("next_action")
    assert all(r["status"] != "configured" for r in result["requirements"].values())
