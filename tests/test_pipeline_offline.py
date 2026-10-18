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
# FunSearch's published Weibull heuristic: genuinely beats best_fit on the real evaluator.
FUNSEARCH = """import numpy as np
def priority(item, capacities):
    bins = np.asarray(capacities, dtype=float)
    max_bin_cap = max(bins)
    score = (bins - max_bin_cap) ** 2 / item + bins ** 2 / item ** 2 + bins ** 2 / item ** 3
    score[bins > item] = -score[bins > item]
    score[1:] -= score[:-1]
    return score
"""
MODELS = ["lab-a/model", "lab-b/model"]


def reply_to(prompt, temperature=1.0):
    """Canned replies keyed on which pipeline prompt is being asked."""
    if prompt.startswith("You are surveying"):
        return json.dumps({"queries": ["bin packing heuristics", "online bin packing survey"]})
    if prompt.startswith("You are building a taxonomy"):
        fam = lambda name, url: {"family": name, "one_line_mechanism": f"{name} mechanism",
                                 "source_summary": f"{name} summary", "source_urls": [url]}
        return json.dumps({"families": [fam("First Fit", "https://a.example"),
                                        fam("Best Fit", "https://b.example"),
                                        fam("Harmonic", "https://c.example"),
                                        fam("Invented", "https://not-in-results.example")],
                           "inapplicable": [{"family": "First Fit Decreasing", "reason": "sorts all items first",
                                             "source_urls": ["https://a.example"]}]})
    if prompt.startswith("Assign each candidate"):
        def family(sid, code):
            if "max_bin_cap" in code:
                return "emergent:quadratic"
            if "-(c - item)" in code:
                return "Best Fit"
            return "emergent:odd" if sid.endswith("-001") else "First Fit"
        solutions = re.findall(r'<solution id="([^"]+)">\n(.*?)</solution>', prompt, re.S)
        return json.dumps({"labels": [{"sample_id": i, "justification": "j", "family": family(i, code)}
                                      for i, code in solutions]})
    if prompt.startswith("Different labelers"):
        return json.dumps({"families": [{"family": "emergent:odd", "one_line_mechanism": "odd",
                                         "merged_from": ["emergent:odd"]}]})
    if prompt.startswith("Which family"):
        return json.dumps({"family": "First Fit", "justification": "takes the first bin that fits"})
    if "Generate 5 responses" in prompt:
        return "".join(f"<response><name>Idea {i}</name><text>Approach {i}.</text>"
                       f"<probability>0.0{i + 1}</probability></response>" for i in range(5))
    if prompt.startswith("Here is a computational problem"):
        return json.dumps({"problem_class": "online assignment to containers",
                           "queries": [{"field": "biology", "query": "slime mould foraging"}]})
    if prompt.startswith("Problem:"):
        return json.dumps({"mechanisms": [
            {"family": "Slime-mould reinforcement", "field": "biology", "one_line_mechanism": "m",
             "source_summary": "s", "source_urls": ["https://a.example"]},
            {"family": "Unsourced idea", "field": "markets", "one_line_mechanism": "m",
             "source_summary": "s", "source_urls": ["https://nowhere.example"]}]})
    if "Family: Slime-mould reinforcement" in prompt:
        return f"```python\n{FUNSEARCH}```"
    if "Family: Best Fit" in prompt:
        return f"```python\n{BEST_FIT}```"
    if "Family: Harmonic" in prompt:
        return f"```python\n{CRASHES}```"
    if temperature >= 2.0:
        return "zx#q plk ~~ vor"  # too hot: garbage
    return f"Here is my solution:\n```python\n{FIRST_FIT}```"  # samplers, and the remaining families


@pytest.fixture
def fake_world(tmp_path, monkeypatch):
    cfg = yaml.safe_load((pipeline.ROOT / "config.yaml").read_text())
    cfg["models"] = {"samplers": MODELS, "labeler": MODELS[0], "implementer": MODELS[0]}
    cfg["sampling"]["n_per_model"] = 3
    cfg["reality_check"]["k"] = 2
    cfg["sampling"]["concurrency"] = 4
    cfg["bench"]["max_candidates"] = 4
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
        return {"choices": [{"message": {"content": reply_to(body["messages"][0]["content"], body["temperature"])},
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
    rejected = {r["family"]: r["reason"] for r in taxonomy["rejected"]}
    assert rejected["Invented"] == "no source URL from the search results"
    assert "sorts all items first" in rejected["First Fit Decreasing"]  # offline: breaks the task's rules
    assert "First Fit Decreasing" not in {b["family"] for b in study["blind_spots"]}
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


def test_bench_operators_at_equal_budget(fake_world):
    from herd.bench import run_bench
    table, verdict, bench_dir = run_bench("online_bin_packing", budget_usd=0.25, repeats=2,
                                          progress=lambda m: None)
    assert set(table) == {"plain", "hot", "verbalized", "blind_spot", "transplant"}
    data = json.loads((bench_dir / "bench.json").read_text())
    for r in data["per_run"]:
        assert 0 < r["spent_usd"] <= 0.25 and r["n_candidates"] > 0
    hot = [r for r in data["per_run"] if r["operator"] == "hot"]
    assert all(r["notes"]["temperature"] == 1.6 for r in hot)  # 2.0 produced garbage
    assert table["transplant"]["useful_discoveries"]["min"] == 1  # the FunSearch-style mechanism
    assert table["plain"]["useful_discoveries"]["max"] == 0
    run = json.loads((bench_dir / "seed0" / "transplant" / "run.json").read_text())
    assert [m["family"] for m in run["notes"]["rejected_unsourced"]] == ["Unsourced idea"]
    assert "transplant beats verbalized" in verdict
    assert (bench_dir / "bench.png").stat().st_size > 10_000 and "## Verdict" in (bench_dir / "bench.md").read_text()

    calls = dict(fake_world)
    table_again, _, _ = run_bench("online_bin_packing", budget_usd=0.25, repeats=2, progress=lambda m: None)
    assert fake_world == calls  # all cached
    assert table_again == table  # cached reruns are billed the same, so they stop in the same place


def test_map_is_rebuilt_when_a_prompt_changes(fake_world, monkeypatch):
    import herd.literature as literature
    herd = pipeline.Herd(progress=lambda m: None)
    task = Task.resolve("online_bin_packing")
    pipeline.ensure_map(herd, task)
    calls = dict(fake_world)
    monkeypatch.setattr(literature, "TAXONOMY_PROMPT", literature.TAXONOMY_PROMPT + "\nBe thorough.")
    pipeline.ensure_map(herd, task)
    assert fake_world["llm"] > calls["llm"]  # the taxonomy was asked again


def test_scores_take_kind_and_share_from_the_current_map(fake_world):
    herd = pipeline.Herd(progress=lambda m: None)
    task = Task.resolve("online_bin_packing")
    study = pipeline.ensure_map(herd, task)
    families = {f["family"]: f for f in study["families"]}
    scores = pipeline.ensure_family_scores(herd, task, study, [families["Best Fit"]], Evaluator(task), k=1)
    assert scores["Best Fit"]["kind"] == "blind_spot" and scores["Best Fit"]["share"] == 0.0
    # Simulate a rebuilt map in which Best Fit became common.
    study["herd_map"]["top3"] = ["Best Fit"]
    study["herd_map"]["families"] = [{"family": "Best Fit", "share": 0.5, "count": 3, "sourced": True}]
    study["blind_spots"] = []
    scores = pipeline.ensure_family_scores(herd, task, study, [], Evaluator(task), k=1)
    assert scores["Best Fit"]["kind"] == "herd" and scores["Best Fit"]["share"] == 0.5


def test_bench_survives_an_operator_planning_failure(fake_world, monkeypatch):
    from herd.bench import run_bench
    pipeline.ensure_map(pipeline.Herd(progress=lambda m: None), Task.resolve("online_bin_packing"))

    def broken_search(self, request, attempts=3):
        raise RuntimeError("exa: HTTP 500")
    monkeypatch.setattr(Exa, "_post", broken_search)
    table, _, bench_dir = run_bench("online_bin_packing", budget_usd=0.25, repeats=1,
                                    operators=["transplant", "plain"], progress=lambda m: None)
    run = json.loads((bench_dir / "seed0" / "transplant" / "run.json").read_text())
    assert "HTTP 500" in run["notes"]["error"] and run["candidates"] == []
    assert table["plain"]["n_candidates"]["mean"] > 0
    assert "stopped early after an error" in (bench_dir / "bench.md").read_text()


def test_bench_refuses_to_start_if_the_global_cap_cannot_fund_every_operator(fake_world):
    from herd.bench import run_bench
    with pytest.raises(SystemExit, match="only"):
        run_bench("online_bin_packing", budget_usd=10.0, repeats=3, progress=lambda m: None)


def test_external_evaluator_cmd_runs_without_secrets(fake_world, tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    script = tmp_path / "eval.py"
    script.write_text("import json, os\n"
                      "print(json.dumps({'valid': 'OPENROUTER_API_KEY' not in os.environ, 'score': 1.0}))\n")
    task = Task.resolve("Some free-text task")
    result = Evaluator(task, f"{sys.executable} {script}")(tmp_path / "candidate.py")
    assert result["valid"] is True
