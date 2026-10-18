"""Escape v3 offline: fake models keyed by prompt markers, stubbed trajectory analysis and
admission gate (as in test_escape_v2), the real sandboxed dev evaluator."""
import json
import re

import pytest

from diverge.escape import (ANCHOR_MARKER, CRITIC_MARKER, DEEPEN_MARKER, IMPLEMENT3_MARKER, MODES, SKETCH3_MARKER,
                            escape, escape_v2, escape_v3, run_escape)
from diverge.llm import LLM
from diverge.reality import Evaluator
from tests.test_escape_v2 import ATTEMPTS, CURRENT, code_lines, overlook
from tests.test_escape_v2 import world as v2_world  # noqa: F401
from tests.test_pipeline_offline import FUNSEARCH, MODELS, fake_world, reply_to  # noqa: F401

FAMILIES = [{"family": "Harmonic classes", "one_line_mechanism": "size classes with dedicated bins"},
            {"family": "Reservation", "one_line_mechanism": "reserve room for large items"}]
LEVER_RE = re.compile(r"Build your mechanism on this one starting point: (\S+)")


def default_critic(n):
    # index 0 is a trap, 1 restates the stuck strategy; ranking of the rest: 3 (27) > 2 (20) > 4 (15)
    totals = {2: (8, 6, 6), 3: (9, 9, 9), 4: (5, 5, 5)}
    out = []
    for i in range(n):
        v, f, d = totals.get(i, (7, 7, 7))
        out.append({"index": i, "viability": v, "fit": f, "distance": d, "strength": f"strength {i}",
                    "trap": f"trap {i}", "is_trap": i == 0, "restates_stuck": i == 1})
    return {"scores": out}


@pytest.fixture
def world(v2_world, monkeypatch):
    h, task, state = v2_world
    words = task.prompt.split()
    verbatim = "  ".join(words[5:14])  # verbatim fragment, whitespace changed
    state.update(prompts=[], models=[], sketch_n=0, garbage_levers=set(), raise_deepen=set(),
                 anchor={"summary": "Attempts score about the same; hardest dev instances shared. STRIPPED",
                         "constraints": [{"constraint": "kept", "quote": verbatim},
                                         {"constraint": "invented", "quote": "bins may be reopened at any later time"}]},
                 critic=default_critic, verbatim=verbatim)

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        state["prompts"].append(prompt)
        state["models"].append((body["model"], prompt))
        if prompt.startswith(ANCHOR_MARKER):
            if state["anchor"] is None:
                raise RuntimeError("anchor model down")
            text = json.dumps(state["anchor"])
        elif SKETCH3_MARKER in prompt:
            lever = LEVER_RE.search(prompt).group(1)
            if lever in state["garbage_levers"]:
                text = "no json here"
            else:
                n = state["sketch_n"]
                state["sketch_n"] += 1
                text = json.dumps({"name": f"Sketch{n}-{lever}", "state": f"state-token-{n}",
                                   "decision_rule": f"rule-token-{n} using {lever}", "uses_lever": f"via {lever}"})
        elif prompt.startswith(CRITIC_MARKER):
            n = len(re.findall(r"^\[\d+\] ", prompt, re.M))
            text = json.dumps(state["critic"](n))
        elif DEEPEN_MARKER in prompt:
            name = re.search(r"Name: (\S+)", prompt).group(1)
            if name in state["raise_deepen"]:
                raise RuntimeError("deepen branch exploded")
            text = json.dumps({"mechanism": f"mechanism of {name}", "main_risk": "risk", "first_step": "step"})
        elif IMPLEMENT3_MARKER in prompt:
            text = f"```python\n{FUNSEARCH}```"
        else:
            text = reply_to(prompt, body["temperature"])
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001, "completion_tokens": 10}}
    monkeypatch.setattr(LLM, "_post", post)
    return h, task, state


def run(h, task, **kw):
    return escape_v3(h, task, CURRENT, ATTEMPTS, Evaluator(task), overlook=overlook(["s-000"]), **kw)


def sketch_prompts(state):
    return [p for p in state["prompts"] if SKETCH3_MARKER in p]


def test_anchor_strip_keeps_verbatim_drops_invented(world):
    h, task, state = world
    result = run(h, task)
    a = result["anchor_strip"]
    assert not a["fallback"] and [c["constraint"] for c in a["constraints"]] == ["kept"]
    assert [c["constraint"] for c in a["dropped"]] == ["invented"]
    assert result["failure_summary"].endswith("STRIPPED")
    p = sketch_prompts(state)[0]
    assert " ".join(state["verbatim"].split()) in p and "reopened at any later time" not in p
    assert "STRIPPED" in p and "a0=" not in p  # stripped summary replaces the v2 one


def test_anchor_strip_failure_falls_back_to_v2_summary(world):
    h, task, state = world
    state["anchor"] = None
    events = []
    result = run(h, task, log=events.append)
    assert result["anchor_strip"]["fallback"] and "anchor model down" in result["anchor_strip"]["error"]
    assert "a0=" in result["failure_summary"] and "Shared strategy" in result["failure_summary"]
    assert any(e.get("stage") == "anchor_strip" and "fallback" in e.get("event", "") for e in events)
    assert result["candidate"]


def test_sketches_isolated_no_code_and_distinct_levers(world):
    h, task, state = world
    result = run(h, task, families=FAMILIES)
    prompts = sketch_prompts(state)
    assert len(prompts) == 5 and len(result["sketches"]) == 5
    levers = [s["lever"] for s in result["sketches"]]
    assert len(set(levers)) == 5 and levers[0] == "history"
    tokens = [t for s in result["sketches"] for t in (s["name"], s["state"], s["decision_rule"])]
    for p in prompts:
        assert not [t for t in tokens if t in p]  # no sketch (own or other) visible
        assert not [line for line in code_lines() if line in p]
        assert "viability" not in p and "held-out" not in p.lower()
    for p in [p for p in state["prompts"] if DEEPEN_MARKER in p or IMPLEMENT3_MARKER in p or p.startswith(CRITIC_MARKER)]:
        assert not [line for line in code_lines() if line in p]


def test_levers_cycle_when_fewer_than_sketches(world):
    h, task, _ = world
    result = run(h, task)  # pool: history, bin_order, new_slot
    assert [s["lever"] for s in result["sketches"]] == ["history", "bin_order", "new_slot", "history", "bin_order"]


def test_critic_drops_traps_ranks_and_deepens_top_two(world):
    h, task, state = world
    h.cfg["models"]["labeler"] = MODELS[1]  # critic should differ from the implementer
    result = run(h, task)
    assert result["critic"]["model"] == MODELS[1] != h.cfg["models"]["implementer"]
    critic_models = {m for m, p in state["models"] if p.startswith(CRITIC_MARKER)}
    assert critic_models == {MODELS[1]}
    assert [s["index"] for s in result["kept_sketches"]] == [3, 2, 4]
    dropped = {r["index"]: r["reasons"][0] for r in result["rejections"] if r["stage"] == "critic"}
    assert "trap" in dropped[0] and "restates" in dropped[1]
    assert [d["index"] for d in result["deepened"]] == [3, 2]
    assert len([p for p in state["prompts"] if IMPLEMENT3_MARKER in p]) == 2
    assert len(state["admit"]) == 2 and {a["sketch"] for a in result["admitted"]} == {"Sketch3-history", "Sketch2-new_slot"}
    assert result["calls"] == {"implementer": 5 + 2 + 2, "extractor": 1, "critic": 1}
    assert {c["sketch"]: c["total"] for c in result["critic"]["scores"]}["Sketch3-history"] == 27


def test_failing_branches_do_not_kill_the_run(world):
    h, task, state = world
    state["garbage_levers"] = {"bin_order"}  # sketches 1 and 4 return garbage
    state["raise_deepen"] = {"Sketch0-history"}  # top-ranked kept sketch raises when deepened
    state["critic"] = lambda n: {"scores": [{"index": i, "viability": 10 - i, "fit": 5, "distance": 5,
                                             "strength": "s", "trap": "t", "is_trap": False, "restates_stuck": False}
                                            for i in range(n)]}
    result = run(h, task)
    stages = [r["stage"] for r in result["rejections"]]
    assert stages.count("sketch") == 2 and stages.count("deepen") == 1
    assert len(result["sketches"]) == 3
    assert [d["sketch"] for d in result["deepened"]] == ["Sketch1-new_slot"]
    assert result["candidate"] and result["evaluation"]["after"] is not None


def test_lever_override_applies_to_all_sketches(world):
    h, task, state = world
    result = run(h, task, lever="item", families=FAMILIES)
    assert {s["lever"] for s in result["sketches"]} == {"item"}
    assert all("starting point: item" in p for p in sketch_prompts(state))
    assert {a["lever"] for a in state["admit"]} == {"item"} and result["lever"]["name"] == "item"


def test_end_to_end_candidate_and_modes(world):
    h, task, state = world
    events = []
    result = run_escape("v3", h, task, CURRENT, ATTEMPTS, Evaluator(task), overlook=overlook(["s-000"]),
                        log=events.append, k=7)
    ev = result["evaluation"]
    assert result["version"] == 3 and ev["split"] == "dev" and ev["after"] is not None
    assert "max_bin_cap" in result["candidate"] and ev["after"] < ev["before"] and ev["improved"]
    assert result["candidate_path"] and all("code" not in a for a in result["admitted"])
    assert any(e["stage"] == "admitted" for e in events) and result["spent_usd"] >= 0
    assert {k: MODES[k] for k in ("v1", "v2", "v3")} == {"v1": escape, "v2": escape_v2, "v3": escape_v3}


def test_resample_prompt_carries_the_admission_reasons():
    """P2: after an admission rejection, the resample prompt states why (not a blind resample)."""
    from diverge.escape import ADMISSION_FEEDBACK
    text = ADMISSION_FEEDBACK.format(reasons="- differs from stuck_2 on only 4.0% of probes", lever="n_open_bins")
    assert "4.0% of probes" in text and "n_open_bins" in text and "rejected before scoring" in text
