"""diverge_escape and diverge_check text: "VERDICT: <v>", then a markdown block, then the JSON object.
Every number in the markdown is copied from the result object."""
import json
import re

from diverge.contract import agent_result, check_result, parse
from tests.test_contract import result as contract_result
from tests.test_agent_ux import call
from tests.test_escape_tool import BEST_FITS
from tests.test_pipeline_offline import FUNSEARCH, fake_world, reply_to  # noqa: F401

import diverge.pipeline as pipeline
from diverge.llm import LLM
from diverge.task import Task

NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def split(text):
    """(verdict line, markdown, json text)."""
    head, _, rest = text.partition("\n")
    md, sep, js = rest.partition("\n{")
    assert sep, "no JSON object after the markdown"
    return head, md, "{" + js


def assert_readable(text, structured):
    head, md, js = split(text)
    assert head == f"VERDICT: {structured['verdict']}"
    assert md.startswith("## DIVERGE ") and "**" in md
    assert not any(line.startswith("{") for line in md.splitlines())
    body = json.loads(js)
    assert parse(text) == (structured["verdict"], body)
    full = json.dumps(structured) + js
    for n in NUMBER.findall(md):
        assert n in full, f"{n} in the markdown is not in the result object"
    return md, body


def test_escape_markdown_for_every_verdict():
    cases = [contract_result(action="continue"), contract_result(action="repair"), contract_result(),
             contract_result(levers=[]), contract_result(scores=(0.04,)),
             {"error": "evaluator_required", "detail": "pass evaluator_cmd", "next_action": "pass it",
              "spent_usd": 0.0}]
    for r in cases:
        text, s = agent_result(r)
        md, body = assert_readable(text, s)
        assert s == {"verdict": s["verdict"], **body}  # structuredContent is the JSON object
        assert "**Next action:**" in md


def test_escape_tool_text_is_readable_and_structured_unchanged(fake_world, monkeypatch):
    pipeline.ensure_map(pipeline.Session(progress=lambda m: None), Task.resolve("online_bin_packing"))

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        text = f"```python\n{FUNSEARCH}```" if "Strategy to move away from" in prompt else reply_to(prompt)
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"cost": 0.001}}
    monkeypatch.setattr(LLM, "_post", post)
    text, s = call("diverge_escape", {"task_description": "online_bin_packing", "current_solution": BEST_FITS[2],
                                      "previous_attempts": BEST_FITS[:2]})
    md, body = assert_readable(text, s)
    assert s == {"verdict": "escape", **body}
    for part in ("**Lever:**", "**Measured (dev):**", "**Traps:**", "**Next test:**"):
        assert part in md, part
    m = s["next_experiment"]["measured"]
    assert json.dumps(m["after"]) in md and json.dumps(m["before"]) in md
    print(text.split("\n{")[0])  # sample for the report (pytest -s)


def test_check_markdown_traceable():
    free = {"verdict": "untested", "mode": "free_text", "complete": True, "spent_usd": 0.0123,
            "herd_comparison": {"family": "First Fit", "share": 0.6667, "rank": 1, "in_top_families": True,
                                "top_families": [["First Fit", 0.6667], ["emergent:odd", 0.1667]]},
            "alternatives": [{"name": "Best Fit", "mechanism": "tightest bin", "strength": "less waste",
                              "trap": "slow", "how_to_test": "compare bins used", "evidence_class": "HYPOTHESIZED"}]}
    measured = {"verdict": "try_blind_spot", "family": "First Fit", "share": 0.6667, "rank": 1,
                "my_family_score": 0.0412, "scored_on": "heldout", "complete": True, "exploratory": True,
                "blind_spot": {"family": "Best Fit", "share": 0.0, "score": 0.0398, "mechanism": "tightest"}}
    for r in (free, measured, {"error": "budget_exceeded", "detail": "x", "next_action": "ask", "complete": False}):
        text, s = check_result(r)
        md, _ = assert_readable(text, s)
        assert s["next_action"] and "complete" in s
    assert "0.0398" in check_result(measured)[0].split("\n{")[0]
