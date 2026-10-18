"""diverge_check without an evaluator (free-text mode), offline: no map is built, the consensus comparison
appears only when a map is cached, and every alternative is HYPOTHESIZED with strength/trap/how_to_test."""
import json

import anyio
from mcp import Client

import diverge.pipeline as pipeline
from diverge.contract import parse
from diverge.llm import LLM
from diverge.task import Task
from tests.test_pipeline_offline import fake_world, reply_to  # noqa: F401

FREE = "Write a function that packs items into bins of capacity 100."
ALTS = [{"name": f"Alt {c}", "mechanism": f"mechanism {c}", "strength": f"strength {c}", "trap": f"trap {c}",
         "how_to_test": f"measure {c}"} for c in "ABCD"]


def install(monkeypatch):
    """Fake models: the alternatives prompt gets ALTS; everything else the pipeline's canned replies.
    Returns the list of prompts asked."""
    prompts = []

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        prompts.append((body["model"], prompt))
        text = json.dumps({"alternatives": ALTS}) if prompt.startswith("You are proposing alternative") \
            else reply_to(prompt, body["temperature"])
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001, "completion_tokens": 10}}
    monkeypatch.setattr(LLM, "_post", post)
    return prompts


def call(args):
    from diverge.server import mcp

    async def go():
        async with Client(mcp) as client:
            r = await client.call_tool("diverge_check", args)
            return r.content[0].text, r.structured_content
    return anyio.run(go)


def test_no_evaluator_no_cached_map_returns_alternatives_without_building(fake_world, monkeypatch):
    prompts = install(monkeypatch)
    text, s = call({"task_description": FREE, "my_approach": "first fit"})
    assert text.splitlines()[0] == "VERDICT: untested" and parse(text) == ("untested", s)
    assert s["mode"] == "free_text" and s["complete"] is True and s["herd_comparison"] is None
    assert "No cached consensus map" in s["consensus_note"]
    # exactly one model call (the alternatives), no sampling, no search, no map on disk
    assert len(prompts) == 1 and prompts[0][1].startswith("You are proposing alternative")
    import yaml
    cfg = yaml.safe_load((pipeline.ROOT / "config.yaml").read_text())
    assert prompts[0][0] == (cfg["models"].get("extractor") or cfg["models"]["labeler"])
    assert fake_world["exa"] == 0
    task = Task.resolve(FREE)
    assert not (pipeline.ROOT / "results" / task.name / "herd_map.json").exists()
    assert not (pipeline.ROOT / "results" / task.name / "samples").exists()
    assert 3 <= len(s["alternatives"]) <= 5
    for a in s["alternatives"]:
        assert a["evidence_class"] == "HYPOTHESIZED"
        assert all(a[k] for k in ("name", "mechanism", "strength", "trap", "how_to_test"))
    assert "HYPOTHESIZED" in s["next_action"] and "evaluator_cmd" in s["next_action"]
    assert "untested" in text.split("\n{")[0].lower()  # the markdown says so too
    assert s["spent_usd"] > 0


def test_no_evaluator_with_cached_map_compares_against_the_consensus(fake_world, monkeypatch):
    task = Task.resolve(FREE)
    pipeline.ensure_map(pipeline.Session(progress=lambda m: None), task)  # fake_world builds and caches it
    exa_before = fake_world["exa"]
    prompts = install(monkeypatch)
    text, s = call({"task_description": FREE, "my_approach": "first fit"})
    h = s["herd_comparison"]
    assert h["family"] == "First Fit" and h["in_top_families"] is True and h["rank"] == 1
    assert h["share"] > 0.5 and h["evidence_class"] == "MEASURED"
    assert ["First Fit", h["share"]] in h["top_families"] and "herd_note" not in s
    # no rebuild: only the classifier and the alternatives call
    assert fake_world["exa"] == exa_before
    assert not any(p.startswith("Assign each candidate") or p.startswith("Here is") for _, p in prompts)
    assert sum(p.startswith("You are proposing alternative") for _, p in prompts) == 1
    alt_prompt = next(p for _, p in prompts if p.startswith("You are proposing alternative"))
    assert "First Fit" in alt_prompt  # the consensus top families are named to steer away from
    assert all(a["evidence_class"] == "HYPOTHESIZED" for a in s["alternatives"])
    assert "**Your approach:** First Fit" in text


def test_evaluator_path_unchanged(fake_world):
    text, s = call({"task_description": "online_bin_packing", "my_approach": "first fit", "k": 2})
    assert text.splitlines()[0] == "VERDICT: try_blind_spot"
    assert s["verdict"] == "try_blind_spot" and s["blind_spot"]["family"] == "Best Fit"
    assert "mode" not in s and "alternatives" not in s and s["next_action"] and s["complete"] is True
