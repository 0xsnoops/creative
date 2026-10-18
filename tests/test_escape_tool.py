"""The herd_escape MCP tool end to end, offline: fake models, real sandboxed evaluator and probes."""
import pytest

import herd.pipeline as pipeline
from herd.llm import LLM
from herd.task import Task
from tests.test_pipeline_offline import FUNSEARCH, call_tool, fake_world, reply_to  # noqa: F401

BIN = "online_bin_packing"
BEST_FITS = [
    "def priority(item, capacities):\n    return [-(c - item) for c in capacities]\n",
    "import numpy as np\ndef priority(item, capacities):\n    caps = np.asarray(capacities)\n    return -(caps - item)\n",
    "def priority(item, capacities):\n    out = []\n    for c in capacities:\n        out.append(item - c)\n    return out\n",
]


def test_herd_escape_detects_the_loop_and_tests_an_alternative(fake_world, monkeypatch):
    pipeline.ensure_map(pipeline.Herd(progress=lambda m: None), Task.resolve(BIN))

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        text = f"```python\n{FUNSEARCH}```" if "Strategy to move away from" in prompt else reply_to(prompt)
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"cost": 0.001}}
    monkeypatch.setattr(LLM, "_post", post)
    result, _ = call_tool("herd_escape", {"task_description": BIN, "current_solution": BEST_FITS[2],
                                          "previous_attempts": BEST_FITS[:2]})
    assert result["loop_detected"] is True and result["action"] == "switch_strategy"
    assert result["strategy_similarity"] > result["code_similarity"]
    assert result["evaluation"]["after"] < result["evaluation"]["before"] and result["evaluation"]["improved"]
    assert result["candidate"] and "max_bin_cap" in result["candidate"]
    assert result["counterexample_summary"]["cases"] == 200 and result["counterexamples"]
    assert all(c["alternative_id"] == "escape" for c in result["counterexamples"])
    assert result["spent_usd"] > 0


def test_herd_escape_requires_an_evaluator(fake_world):
    result, _ = call_tool("herd_escape", {"task_description": "Design a deduplication service.",
                                          "current_solution": "def f(): pass\n"})
    assert result["error"] == "evaluator_required"
