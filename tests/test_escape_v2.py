"""Escape v2 offline: fake models, a stubbed trajectory analysis and admission gate, the real
sandboxed evaluator and counterexample probes."""
import json
import sys
import types

import pytest

import diverge as pkg
import diverge.pipeline as pipeline
from diverge import counterexample as ce
from diverge.escape import (IMPLEMENT_MARKER, JUDGE_MARKER, USES_MARKER, NO_FUTURE, SKETCH_MARKER, describe_counterexample,
                         escape_v2, mentions_lever)
from diverge.llm import LLM
from diverge.reality import Evaluator
from diverge.task import Task
from tests.test_escape import report
from tests.test_pipeline_offline import BEST_FIT, FUNSEARCH, fake_world, reply_to  # noqa: F401

BIN = "online_bin_packing"
RESCALED = "def priority(item, capacities):\n    return [-1000.0 * (c - item) + 7 for c in capacities]\n"
PLUS_ZERO = BEST_FIT.replace("item)", "item) + 0.0")
ATTEMPTS = [{"id": "a0", "code": RESCALED}, {"id": "a1", "code": BEST_FIT}]
CURRENT = {"id": "a2", "code": PLUS_ZERO}

SKETCHES = [
    {"name": "History-reserved bins", "state": "counts of recent item sizes",
     "decision_rule": "use the history of item sizes to reserve bins for likely complements",
     "uses_lever": "how the decision depends on history"},
    {"name": "Rescaled tight fit", "state": "none",
     "decision_rule": "tightest fit, weighted by history",
     "uses_lever": "re-weights the tight fit"},
    {"name": "Plain worst fit", "state": "none", "decision_rule": "put the item in the emptiest bin",
     "uses_lever": "it does not"},
]


def overlook(sample_ids):
    row = lambda name, share, ev: {"input": name, "description": f"{name} desc", "share": share,
                                   "used_by": round(share * 95), "decided": 95, "overlooked": share <= 0.10,
                                   "used_by_models": {}, "evidence": ev}
    ev = [{"sample_id": s, "model": "lab-a/model", "evidence": "history enters a score at line 3"} for s in sample_ids]
    return {"inputs": [row("history", 0.03, ev), row("item", 1.0, []), row("capacity_values", 0.98, [])],
            "assumptions": {"assumptions": [
                {"assumption": "Items must be placed immediately", "kind": "hard", "shared": True, "held_by": 12,
                 "checked": 12, "prompt_quote": "must each be placed in a bin immediately", "evidence": []}]},
            "minority": {"families": []}, "herd": {"samples": 100, "valid": 95}, "paraphrase_test": None}


@pytest.fixture
def world(fake_world, monkeypatch):
    """Session with two consensus samples (FunSearch valid, one invalid), stubbed analysis and admission,
    and a fake model that records every prompt."""
    stub = types.ModuleType("diverge.trajectory")
    stub.analyze = lambda task, attempts: report()
    monkeypatch.setitem(sys.modules, "diverge.trajectory", stub)
    monkeypatch.setattr(pkg, "trajectory", stub, raising=False)

    state = {"prompts": [], "admit": [], "reject": 0, "sketches": SKETCHES}

    def admit(task, code, stuck_codes, lever, min_disagreement=0.10, min_sensitivity=0.05):
        state["admit"].append({"code": code, "stuck": stuck_codes, "lever": lever})
        ok = len(state["admit"]) > state["reject"]
        return {"admitted": ok, "disagreement": 0.4 if ok else 0.01, "sensitivity": 0.2 if ok else 0.0,
                "perturbation": "history shuffled", "reasons": [] if ok else ["agrees with the stuck attempts"],
                "valid": True}
    adm = types.ModuleType("diverge.admission")
    adm.admit = admit
    monkeypatch.setitem(sys.modules, "diverge.admission", adm)
    monkeypatch.setattr(pkg, "admission", adm, raising=False)

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        state["prompts"].append(prompt)
        if SKETCH_MARKER in prompt:
            text = json.dumps({"sketches": state["sketches"]})
        elif prompt.startswith(USES_MARKER):  # semantic fallback: only explicit paraphrases count
            text = json.dumps({"uses_lever": "PARAPHRASED_LEVER" in prompt, "reason": "r"})
        elif prompt.startswith(JUDGE_MARKER):
            text = json.dumps({"same_mechanism": "Rescaled" in prompt, "reason": "r"})
        elif IMPLEMENT_MARKER in prompt:
            text = f"```python\n{FUNSEARCH}```"
        else:
            text = reply_to(prompt, body["temperature"])
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001, "completion_tokens": 10}}
    monkeypatch.setattr(LLM, "_post", post)

    h = pipeline.Session(progress=lambda m: None)
    task = Task.resolve(BIN)
    out = h.out_dir(task)
    (out / "samples").mkdir(exist_ok=True)
    (out / "samples" / "s-000.py").write_text(FUNSEARCH)
    rows = [{"sample_id": "s-000", "status": "valid", "code_path": "samples/s-000.py"},
            {"sample_id": "s-001", "status": "invalid", "code_path": "samples/s-001.py"}]
    (out / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return h, task, state


def code_lines():
    task_prompt = Task.resolve(BIN).prompt
    return {line.strip() for c in (RESCALED, BEST_FIT, PLUS_ZERO, FUNSEARCH) for line in c.splitlines()
            if line.strip() and line.strip() not in task_prompt}


def test_escape_v2_end_to_end(world):
    h, task, state = world
    events = []
    result = escape_v2(h, task, CURRENT, ATTEMPTS, Evaluator(task), overlook=overlook(["s-001", "s-000"]),
                       log=events.append)
    assert result["version"] == 2 and result["lever"]["name"] == "history"
    assert "seed:" in result["failure_summary"] and "a0=" in result["failure_summary"]
    assert result["failure_summary"].count("Shared strategy") == 1

    cx = result["counterexample"]
    assert cx and cx["alternative_id"] == "s-000" and cx["alternative_better"]
    assert NO_FUTURE in cx["text"] and cx["text"] in result["sketch_prompt"]
    assert cx["decision"]["placed_so_far"] == cx["case"]["items"][:cx["step"] - 1]

    assert len(result["sketches"]) == 3
    assert [s["name"] for s in result["kept_sketches"]] == ["History-reserved bins"]
    dropped = {r["sketch"]: r["reasons"][0] for r in result["rejections"] if r["stage"] == "sketch_filter"}
    assert "does not use history" in dropped["Plain worst fit"]  # no lever in the decision rule
    assert "same mechanism" in dropped["Rescaled tight fit"]  # extractor judged it the same
    # extractor: one "uses the lever?" fallback for the sketch without the lever name, two "same mechanism?" judgments
    assert result["calls"] == {"implementer": 2, "extractor": 3}

    ev = result["evaluation"]
    assert "max_bin_cap" in result["candidate"] and ev["split"] == "dev"
    assert ev["after"] < ev["before"] and ev["improved"]
    assert result["admitted"][0]["valid"] and "code" not in result["admitted"][0]
    assert any(e["stage"] == "admitted" for e in events)
    assert state["admit"][0]["lever"] == "history" and set(state["admit"][0]["stuck"]) == {RESCALED, BEST_FIT, PLUS_ZERO}

    # No attempt (or consensus sample) code in any sketch / implementation / judge prompt.
    v2_prompts = [p for p in state["prompts"] if SKETCH_MARKER in p or IMPLEMENT_MARKER in p or p.startswith(JUDGE_MARKER) or p.startswith(USES_MARKER)]
    assert len(v2_prompts) == 5  # sketch, uses-lever fallback, 2 judgments, implementation
    for p in v2_prompts:
        assert not [line for line in code_lines() if line in p]
        assert "held-out" not in p.lower() and "heldout" not in p.lower()


def test_counterexample_decision_ignores_future_items(world):
    h, task, _ = world
    found = ce.find(task, {"a": BEST_FIT, "b": RESCALED}, {"alt": FUNSEARCH}, n=1000)
    cx = next(c for c in found["counterexamples"] if c["alternative_better"])
    d = describe_counterexample(BIN, cx, "history", 2)
    step = d["step"]
    assert NO_FUTURE in d["decision_text"] and d["decision"]["item"] == cx["case"]["items"][step - 1]
    # Changing the items after the divergence step leaves the decision description unchanged.
    changed = {**cx, "case": {"items": cx["case"]["items"][:step] + [99, 98, 97]}}
    assert describe_counterexample(BIN, changed, "history", 2)["decision_text"] == d["decision_text"]
    later = cx["case"]["items"][step:]
    assert d["text"].index("Outcome") > d["text"].index("Decision at step")
    if later:
        assert ", ".join(map(str, later)) in d["outcome_text"]


def test_admission_rejection_resamples_and_is_logged(world):
    h, task, state = world
    state["reject"] = 1
    events = []
    result = escape_v2(h, task, CURRENT, ATTEMPTS, Evaluator(task), overlook=overlook(["s-000"]), log=events.append)
    rejected = [r for r in result["rejections"] if r["stage"] == "admission"]
    assert len(rejected) == 1 and rejected[0]["attempt"] == 0
    assert rejected[0]["reasons"] == ["agrees with the stuck attempts"]
    assert rejected[0]["disagreement"] == 0.01 and rejected[0]["sensitivity"] == 0.0
    assert rejected[0] in events and rejected[0] in result["log"]
    assert result["admitted"][0]["attempt"] == 1 and result["candidate"]
    assert result["calls"]["implementer"] == 3  # sketches + two implementations


def test_max_resample_is_respected(world):
    h, task, state = world
    state["reject"] = 100
    result = escape_v2(h, task, CURRENT, ATTEMPTS, Evaluator(task), overlook=overlook(["s-000"]), max_resample=1)
    assert len(state["admit"]) == 2  # one kept sketch x (1 + max_resample)
    assert len([r for r in result["rejections"] if r["stage"] == "admission"]) == 2
    assert result["candidate"] is None and result["evaluation"]["after"] is None
    assert not result["evaluation"]["improved"] and result["evaluation"]["before"] is not None


def test_lever_override_control_arm(world):
    h, task, state = world
    state["sketches"] = [{"name": "Item-size classes", "state": "none",
                          "decision_rule": "classify the item by size and open class bins",
                          "uses_lever": "how the decision depends on item"}]
    result = escape_v2(h, task, CURRENT, ATTEMPTS, Evaluator(task), overlook=overlook(["s-000"]), lever="item")
    assert result["lever"]["name"] == "item" and result["lever"]["type"] == "input"
    assert result["counterexample"] is None  # no consensus program listed as evidence for item
    assert "Lever to build on: item" in result["sketch_prompt"]
    assert "no valid consensus program that uses item" in result["sketch_prompt"]
    assert state["admit"][0]["lever"] == "item" and result["candidate"]


def test_sketch_filter_requires_the_lever():
    lever = {"name": "n_open_bins", "detail": {"description": "how many bins are offered (len(capacities))"}}
    assert mentions_lever({"decision_rule": "when n open bins is large, prefer..."}, lever)
    assert mentions_lever({"decision_rule": "if len(capacities) > 3 then ..."}, lever)
    assert not mentions_lever({"decision_rule": "tightest fit", "uses_lever": "n_open_bins"}, lever)


def test_sketch_filter_fallback_prompt_for_paraphrased_or_family_levers():
    """The literal lever-name check is a pre-filter: a paraphrase ("a fresh bin") or a family lever
    never matches it, so those go to the extractor with this prompt instead of being dropped."""
    import diverge.escape as esc
    lever = {"name": "new_bin_slot", "detail": {"description": "special-casing the new bin"}}
    para = {"name": "a", "decision_rule": "opens a fresh bin when the tightest fit wastes too much"}
    assert not esc.mentions_lever(para, lever)
    prompt = esc.USES_TEMPLATE.format(lever="new_bin_slot", description=" (special-casing the new bin)", name="a",
                                      state="none", decision_rule=para["decision_rule"], uses_lever="")
    assert prompt.startswith(esc.USES_MARKER) and para["decision_rule"] in prompt and '"uses_lever"' in prompt
