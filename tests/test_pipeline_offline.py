"""End-to-end pipeline and MCP server tests with OpenRouter and Exa replaced by fakes.
The evaluators are real."""
import json
import re

import anyio
import pytest
import yaml
from mcp import Client

import herd.pipeline as pipeline
from herd.exa import Exa
from herd.llm import LLM
from herd.reality import Evaluator
from herd.task import Task

FIRST_FIT = "def priority(item, capacities):\n    return [-i for i in range(len(capacities))]\n"
BEST_FIT = "def priority(item, capacities):\n    return [-(c - item) for c in capacities]\n"
CRASHES = "def priority(item, capacities):\n    raise RuntimeError('broken')\n"
MODELS = ["lab-a/model", "lab-b/model"]


def reply_to(prompt):
    """Canned replies keyed on which pipeline prompt is being asked."""
    if prompt.startswith("You are surveying"):
        return json.dumps({"queries": ["bin packing heuristics", "online bin packing survey"]})
    if prompt.startswith("You are building a taxonomy"):
        fam = lambda name, url: {"family": name, "one_line_mechanism": f"{name} mechanism",
                                 "source_summary": f"{name} summary", "source_urls": [url]}
        return json.dumps({"families": [fam("First Fit", "https://a.example"),
                                        fam("Best Fit", "https://b.example"),
                                        fam("Harmonic", "https://c.example"),
                                        fam("Invented", "https://not-in-results.example")]})
    if prompt.startswith("Assign each candidate"):
        ids = re.findall(r'<solution id="([^"]+)">', prompt)
        return json.dumps({"labels": [{"sample_id": i, "justification": "j",
                                       "family": "emergent:odd" if i.endswith("-001") else "First Fit"}
                                      for i in ids]})
    if prompt.startswith("Different labelers"):
        return json.dumps({"families": [{"family": "emergent:odd", "one_line_mechanism": "odd",
                                         "merged_from": ["emergent:odd"]}]})
    if prompt.startswith("Which family"):
        return json.dumps({"family": "First Fit", "justification": "takes the first bin that fits"})
    if "Family: Best Fit" in prompt:
        return f"```python\n{BEST_FIT}```"
    if "Family: Harmonic" in prompt:
        return f"```python\n{CRASHES}```"
    return f"Here is my solution:\n```python\n{FIRST_FIT}```"  # samplers, and the remaining families


@pytest.fixture
def fake_world(tmp_path, monkeypatch):
    cfg = yaml.safe_load((pipeline.ROOT / "config.yaml").read_text())
    cfg["models"] = {"samplers": MODELS, "labeler": MODELS[0], "implementer": MODELS[0]}
    cfg["sampling"]["n_per_model"] = 3
    cfg["reality_check"]["k"] = 2
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "cache").mkdir()
    catalog = {"data": [{"id": m, "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                         "top_provider": {"max_completion_tokens": 64000}} for m in MODELS]}
    (tmp_path / "cache" / "openrouter_models.json").write_text(json.dumps(catalog))
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "_caches", {})

    calls = {"llm": 0, "exa": 0}

    def fake_llm_post(self, body, attempts=4):
        calls["llm"] += 1
        return {"choices": [{"message": {"content": reply_to(body["messages"][0]["content"])},
                             "finish_reason": "stop"}],
                "usage": {"cost": 0.001, "completion_tokens": 10}}

    def fake_exa_post(self, request, attempts=3):
        calls["exa"] += 1
        return {"costDollars": {"total": 0.005},
                "results": [{"url": f"https://{x}.example", "title": x, "highlights": ["..."]} for x in "abc"]}

    monkeypatch.setattr(LLM, "_post", fake_llm_post)
    monkeypatch.setattr(Exa, "_post", fake_exa_post)
    return calls


def test_map_blind_spots_and_check(fake_world):
    herd = pipeline.Herd(progress=lambda m: None)
    task = Task.resolve("online_bin_packing")
    study = pipeline.ensure_map(herd, task)
    m = study["herd_map"]
    assert m["n_labeled"] == 6 and m["top3"][0] == "First Fit"
    assert m["families"][0]["share"] == pytest.approx(4 / 6, abs=1e-4)
    assert {b["family"] for b in study["blind_spots"]} == {"Best Fit", "Harmonic"}
    taxonomy = json.loads((pipeline.ROOT / "results/online_bin_packing/taxonomy.json").read_text())
    assert [r["family"] for r in taxonomy["rejected"]] == ["Invented"]  # cited a URL Exa never returned
    assert (pipeline.ROOT / "results/online_bin_packing/labels_audit.csv").exists()

    calls_before = dict(fake_world)
    assert pipeline.ensure_map(herd, task) == study  # cached: no new API calls
    assert fake_world == calls_before

    verdict = pipeline.check_approach(herd, task, study, "first fit", Evaluator(task), k=2)
    assert verdict["verdict"] == "try_blind_spot"
    assert verdict["blind_spot"]["family"] == "Best Fit"
    scores = json.loads((pipeline.ROOT / "results/online_bin_packing/family_scores.json").read_text())
    harmonic = scores["builtin:online_bin_packing"]["Harmonic"]
    assert harmonic["n_invalid"] == 2 and harmonic["best_score"] is None  # crashes are reported


def test_free_text_task_without_evaluator(fake_world):
    herd = pipeline.Herd(progress=lambda m: None)
    task = Task.resolve("Write a function that packs items into bins of capacity 100.")
    assert task.name.startswith("desc-") and task.evaluator is None
    study = pipeline.ensure_map(herd, task)
    verdict = pipeline.check_approach(herd, task, study, "first fit", None, k=2)
    assert verdict["verdict"] == "untested" and verdict["family"] == "First Fit"


def call_tool(name, args):
    from herd.server import mcp
    progress = []

    async def go():
        async with Client(mcp) as client:
            result = await client.call_tool(name, args, progress_callback=lambda p, t, msg: progress.append(msg))
            return result.content[0].text

    return json.loads(anyio.run(go)), progress


def test_mcp_tools(fake_world):
    herd_map, progress = call_tool("herd_map", {"task_description": "online_bin_packing"})
    assert herd_map["top3"][0] == "First Fit" and herd_map["spent_usd"] > 0
    assert any(m.startswith("sample") for m in progress)

    spots, _ = call_tool("blind_spots", {"task_description": "online_bin_packing"})
    assert {b["family"] for b in spots["blind_spots"]} == {"Best Fit", "Harmonic"}
    assert spots["spent_usd"] == 0  # served from cache

    rc, _ = call_tool("reality_check", {"task_description": "online_bin_packing", "family": "Best Fit", "k": 2})
    assert rc["n_valid"] == 2 and "def priority" in rc["best_code"]
    assert rc["beats_baselines"] == ["first_fit"]  # ties best_fit exactly, so does not beat it

    check, _ = call_tool("herd_check", {"task_description": "online_bin_packing", "my_approach": "first fit", "k": 2})
    assert check["verdict"] == "try_blind_spot" and check["blind_spot"]["family"] == "Best Fit"


def test_mcp_budget_cap(fake_world):
    result, _ = call_tool("herd_map", {"task_description": "online_bin_packing", "budget_usd": 0.0001})
    assert result["error"] == "budget_exceeded"
