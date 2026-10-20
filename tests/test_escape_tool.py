"""The diverge_escape MCP tool end to end, offline: fake models, real sandboxed evaluator and probes."""
import json
from pathlib import Path

import anyio
from mcp import Client

import diverge.pipeline as pipeline
from diverge.contract import parse
from diverge.llm import LLM
from diverge.task import Task
from tests.test_pipeline_offline import FUNSEARCH, fake_world, reply_to  # noqa: F401

BIN = "online_bin_packing"


def call_escape(args):
    """diverge_escape over MCP: (verdict, contract body) parsed from its text output."""
    from diverge.server import mcp

    async def go():
        async with Client(mcp) as client:
            return (await client.call_tool("diverge_escape", args)).content[0].text
    text = anyio.run(go)
    assert text.startswith("VERDICT: ")
    return parse(text)
SKETCHES = [{"name": "Capacity-aware fit", "state": "none",
             "decision_rule": "score bins by item fit relative to the largest open bin capacity",
             "uses_lever": "the decision depends on the lever"}]


def v2_reply(prompt, sketches=SKETCHES, code=None):
    """Fake replies for escape v2's prompts (sketch, uses-lever and same-mechanism judges, implement);
    anything else gets the pipeline's canned reply."""
    from diverge.escape import IMPLEMENT_MARKER, JUDGE_MARKER, SKETCH_MARKER, USES_MARKER
    if SKETCH_MARKER in prompt:
        return json.dumps({"sketches": sketches})
    if prompt.startswith(USES_MARKER):
        return json.dumps({"uses_lever": True, "reason": "r"})
    if prompt.startswith(JUDGE_MARKER):
        return json.dumps({"same_mechanism": False, "reason": "a different decision signal"})
    if IMPLEMENT_MARKER in prompt:
        return f"```python\n{code or FUNSEARCH}```"
    return reply_to(prompt)


BEST_FITS = [
    "def priority(item, capacities):\n    return [-(c - item) for c in capacities]\n",
    "import numpy as np\ndef priority(item, capacities):\n    caps = np.asarray(capacities)\n    return -(caps - item)\n",
    "def priority(item, capacities):\n    out = []\n    for c in capacities:\n        out.append(item - c)\n    return out\n",
]


def test_diverge_escape_detects_the_loop_and_tests_an_alternative(fake_world, monkeypatch):
    pipeline.ensure_map(pipeline.Session(progress=lambda m: None), Task.resolve(BIN))

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        return {"choices": [{"message": {"content": v2_reply(prompt)}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001}}
    monkeypatch.setattr(LLM, "_post", post)
    verdict, body = call_escape({"task_description": BIN, "current_solution": BEST_FITS[2],
                                 "previous_attempts": BEST_FITS[:2]})
    assert verdict == "escape" and body["lever"]["mechanism"]
    claims = {e["claim"]: e["class"] for e in body["evidence"]}
    sim = next(c for c in claims if c.startswith("mean pairwise strategy similarity"))
    assert claims[sim] == "MEASURED"
    strategy, code = (float(x) for x in sim.removeprefix("mean pairwise strategy similarity ").split(" against code similarity "))
    assert strategy > code
    assert any(c.startswith("attempt dev scores: attempt-0=") for c in claims)
    measured = body["next_experiment"]["measured"]
    assert measured["after"] < measured["before"] and measured["improved"] and measured["split"] == "dev"
    assert body["candidate_path"] and "max_bin_cap" in Path(body["candidate_path"]).read_text()
    assert body["counterexamples"] and all(c["alternative_id"] == "escape" for c in body["counterexamples"])
    assert any(e["claim"].startswith("counterexamples:") and e["detail"]["cases"] == 200 for e in body["evidence"])
    assert body["spent_usd"] > 0


def test_diverge_escape_requires_an_evaluator(fake_world):
    verdict, body = call_escape({"task_description": "Design a deduplication service.",
                                 "current_solution": "def f(): pass\n"})
    assert verdict == "inconclusive"
    assert body["error"] == "evaluator_required" and "spent_usd" in body
