"""diverge_escape after the review panel, offline: lever_policy (chosen, random, mixed), levers for a task
with no consensus data (the attempts' unused inputs, else one model-proposed alternative), admission
passing through on tasks without probes, and a new evaluator_cmd task escaping end to end."""
import json
import sys
from pathlib import Path

import pytest

import diverge.pipeline as pipeline
from diverge import escape as E
from diverge.llm import LLM
from diverge.task import Task
from tests.test_escape import report
from tests.test_escape_tool import call_escape, v2_reply
from tests.test_escape_v2 import ATTEMPTS, CURRENT, overlook, world  # noqa: F401
from tests.test_pipeline_offline import fake_world  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
BIN = "online_bin_packing"


# ------------------------------------------------------------------ lever choice units
def test_random_lever_draws_like_the_control_and_is_seeded_by_the_attempts():
    ov = overlook([])
    attempts = [{"code": "a"}, {"code": "b"}]
    first = E.random_lever(report(), ov, [], attempts)
    assert first["type"] == "random_input" and first["name"] in ("item", "capacity_values")  # share >= 0.5
    assert "drawn at random" in first["why"] and "rescue test" in first["why"]
    assert E.random_lever(report(), ov, [], attempts) == first  # same attempts, same draw
    draws = {E.random_lever(report(), ov, [], [{"code": str(i)}])["name"] for i in range(20)}
    assert draws == {"item", "capacity_values"}
    found = [{"type": "blind_spot", "name": "Harmonic", "mechanism": "m", "why": "w"}]
    assert E.random_lever(report(), None, found, attempts)["name"] == "Harmonic"
    assert E.random_lever(report(), None, [], attempts) is None


def test_attempt_levers_are_the_inputs_no_attempt_uses():
    found = E.attempt_levers(report())
    assert [f["name"] for f in found] == report()["shared"]["inputs_unused"] and found
    assert all(f["type"] == "attempt_unused_input" and "no consensus data" in f["why"] for f in found)
    assert E.attempt_levers({"n": 2, "shared": {"inputs_unused": []}}) == []


def test_admission_passes_through_on_tasks_without_probes():
    verdict = E._admit(Task.resolve("cache_eviction"), "class Policy: pass\n", ["x = 1\n"], "history")
    assert verdict["admitted"] is True and "not checked" in verdict["reasons"][0]


# ------------------------------------------------------------------ escape_v2 lever sources
def test_v2_without_consensus_data_uses_the_attempts_then_the_fallback(world):
    session = pipeline.Session(progress=lambda m: None)
    task, ev = Task.resolve(BIN), _evaluator()
    out = E.escape_v2(session, task, CURRENT, ATTEMPTS, ev, overlook=None, k_sketches=1, max_resample=0)
    assert out["lever_source"] == "attempts" and out["lever"]["type"] == "attempt_unused_input"
    calls = []

    def fallback(diagnosis, rep):
        calls.append(diagnosis["action"])
        return {"type": "agent_alternative", "name": "Harmonic classes", "detail": {"description": "size classes"},
                "mechanism": "pack items by size class", "why": "HYPOTHESIZED"}
    import diverge.trajectory as stub  # the world fixture's stub
    stub.analyze = lambda task, attempts: {**report(), "shared": {**report()["shared"], "inputs_unused": []}}
    out = E.escape_v2(pipeline.Session(progress=lambda m: None), task, CURRENT, ATTEMPTS, ev, overlook=None,
                      k_sketches=1, max_resample=0, lever_fallback=fallback)
    assert calls == ["switch_strategy"] and out["lever_source"] == "fallback"
    assert out["lever"]["name"] == "Harmonic classes"


def test_v2_generates_nothing_unless_switching_when_asked(world):
    import diverge.trajectory as stub
    stub.analyze = lambda task, attempts: report(loop=False, improvement=0.2)
    out = E.escape_v2(pipeline.Session(progress=lambda m: None), Task.resolve(BIN), CURRENT, ATTEMPTS, _evaluator(),
                      overlook=overlook([]), only_on_switch=True)
    assert out["action"] == "continue" and out["sketches"] == [] and out["candidate"] is None
    assert any("diagnosis is continue" in str(e.get("event")) for e in out["log"])


def test_unknown_lever_policy_is_refused(world):
    with pytest.raises(ValueError, match="lever_policy"):
        E.escape_v2(pipeline.Session(progress=lambda m: None), Task.resolve(BIN), CURRENT, ATTEMPTS, _evaluator(),
                    lever_policy="best")


def _evaluator():
    from diverge.reality import Evaluator
    return Evaluator(Task.resolve(BIN))


# ------------------------------------------------------------------ mixed policy
def _raw(policy, after, improved, lever):
    return {"diagnosis": {"action": "switch_strategy", "shared_cause": "c", "herd_prevalence": None, "reasons": [],
                          "evidence": {}},
            "analysis": {}, "lever": {"type": "input", "name": lever, "mechanism": "m", "why": "w", "detail": {}},
            "lever_policy": policy, "lever_source": policy, "lever_alternatives": [], "admitted": [],
            "evaluation": {"before": 4.0, "after": after, "improved": improved, "split": "dev"},
            "action": "switch_strategy", "candidate": None, "candidate_path": None}


def test_mixed_tries_a_random_lever_only_when_the_chosen_one_did_not_improve(fake_world, monkeypatch):
    from diverge import server
    runs = []

    def fake_v2(session, task, cur, prev, evaluator, **kw):
        runs.append(kw["lever_policy"])
        return _raw(kw["lever_policy"], *({"chosen": (4.5, False, "history"), "random": (3.0, True, "item")}
                                          [kw["lever_policy"]]))
    monkeypatch.setattr(E, "escape_v2", fake_v2)
    session, ev = pipeline.Session(progress=lambda m: None), _evaluator()
    out = server.escape_for_agent(session, Task.resolve(BIN), "x", [], ev, lever_policy="mixed", counterexamples=False)
    assert runs == ["chosen", "random"] and out["evaluation"]["after"] == 3.0
    assert out["overlooked_levers"][0]["name"] == "item"
    assert [a["policy"] for a in out["escape"]["lever_attempts"]] == ["chosen", "random"]
    runs.clear()
    monkeypatch.setattr(E, "escape_v2", lambda *a, **kw: (runs.append(kw["lever_policy"]),
                                                         _raw("chosen", 3.5, True, "history"))[1])
    out = server.escape_for_agent(session, Task.resolve(BIN), "x", [], ev, lever_policy="mixed", counterexamples=False)
    assert runs == ["chosen"] and out["evaluation"]["after"] == 3.5


# ------------------------------------------------------------------ a new task, end to end
def predictor(expr):
    return f"def predict(x):\n    return {expr}\n"


def test_new_evaluator_cmd_task_escapes_with_a_model_proposed_lever(fake_world, monkeypatch, tmp_path):
    data = tmp_path / "recorded.jsonl"
    data.write_text("".join(json.dumps({"id": f"r{i}", "input": x, "expected": sorted(x)[1]}) + "\n"
                            for i, x in enumerate([[1, 2, 30], [5, 6, 7], [2, 40, 41], [9, 10, 90]])))
    cmd = f"{sys.executable} {ROOT / 'examples' / 'replay_evaluator.py'} --data {data}"
    median = predictor("sorted(x)[len(x) // 2]")
    prompts = []

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        prompts.append(prompt)
        if prompt.startswith("You are proposing alternative"):
            text = json.dumps({"alternatives": [{"name": "Median of inputs", "mechanism": "predict the middle value",
                                                 "strength": "s", "trap": "t", "how_to_test": "h",
                                                 "test_cost": "minutes", "decisive": True,
                                                 "kill_result": "k"}], "relax_hint": ""})
        else:
            text = v2_reply(prompt, sketches=[{"name": "Middle value", "state": "none",
                                               "decision_rule": "return the median of the inputs",
                                               "uses_lever": "it is the median"}], code=median)
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"cost": 0.001}}
    monkeypatch.setattr(LLM, "_post", post)
    # three means in different code: same worst records, flat scores
    attempts = [predictor("sum(x) / len(x)"), predictor("sum(v for v in x) / len(x)"), predictor("(x[0] + x[1] + x[2]) / 3")]
    verdict, body = call_escape({"task_description": "Predict the expected value of each recorded input.",
                                 "evaluator_cmd": cmd, "current_solution": attempts[-1],
                                 "previous_attempts": attempts[:-1]})
    assert verdict == "escape", body
    assert body["escape"]["lever_source"] == "fallback" and body["lever"]["type"] == "agent_alternative"
    assert body["next_experiment"]["measured"]["improved"] is True
    assert body["next_experiment"]["measured"]["after"] == 0.0
    assert any("failure loop" in e["claim"] for e in body["evidence"])
    assert any(p.startswith("You are proposing alternative") for p in prompts)


def test_new_task_without_per_case_results_is_inconclusive_with_a_way_forward(fake_world, monkeypatch, tmp_path):
    script = tmp_path / "eval.py"
    script.write_text("import json, sys\nprint(json.dumps({'valid': True, 'score': 0.5}))\n")
    monkeypatch.setattr(LLM, "_post", lambda self, body, attempts=4: {
        "choices": [{"message": {"content": v2_reply(body["messages"][0]["content"])}, "finish_reason": "stop"}],
        "usage": {"cost": 0.001}})
    verdict, body = call_escape({"task_description": "Predict the expected value of each recorded input.",
                                 "evaluator_cmd": f"{sys.executable} {script}",
                                 "current_solution": predictor("1"), "previous_attempts": [predictor("2")]})
    assert verdict == "inconclusive" and '"cases"' in body["next_action"]
    assert body.get("candidate_path") is None


# ------------------------------------------------------------------ diverge_loop_check ($0)
def test_loop_check_is_free_and_uses_the_escape_rules(fake_world, monkeypatch):
    from tests.test_agent_ux import call

    def boom(*a, **k):
        raise AssertionError("no model call")
    monkeypatch.setattr(LLM, "_post", boom)
    check = lambda **a: json.loads(call("diverge_loop_check", a)[0])
    assert check(scores=[0.5])["decision"] == "need_more_attempts"
    assert check(scores=[5.0, 4.0, 3.0])["decision"] == "keep_going"
    loop = check(scores=[4.1, 4.1, 4.09], failing_cases=[["s1", "s3"], ["s1", "s3"], ["s3", "s1"]])
    assert loop["decision"] == "call_diverge_escape" and loop["failure_jaccard_min"] == 1.0
    assert loop["spent_usd"] == 0.0 and loop["stalled"] is True
    assert check(scores=[4.1, 4.1, 4.1], failing_cases=[["s1", "s3"], ["s2", "s4"], ["s1", "s3"]])["decision"] == "repair"
    assert check(scores=[0.3, 0.3, None, 0.3], lower_is_better=False)["decision"] == "stalled_unknown"


# ------------------------------------------------------------------ patch operator (research arm)
def test_patch_mutates_the_best_attempt_and_never_gates_on_admission(world, monkeypatch):
    from tests.test_pipeline_offline import FUNSEARCH
    prompts = []
    real = LLM._post

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        if E.PATCH_MARKER in prompt:
            prompts.append(prompt)
            return {"choices": [{"message": {"content": f"```python\n{FUNSEARCH}```"}, "finish_reason": "stop"}],
                    "usage": {"cost": 0.001}}
        return real(self, body, attempts)
    monkeypatch.setattr(LLM, "_post", post)
    world[2]["reject"] = 99  # the stub admission rejects everything: patch must still score its candidates
    session = pipeline.Session(progress=lambda m: None)
    out = E.escape_patch(session, Task.resolve(BIN), CURRENT, ATTEMPTS, _evaluator(), overlook=overlook([]),
                         lever="history", k=2)
    parent = out["parent"]
    assert parent["id"] in ("a0", "a1", "a2") and len(prompts) == 2
    codes = {a["id"]: a["code"].strip() for a in [*ATTEMPTS, CURRENT]}
    assert codes[parent["id"]] in prompts[0]  # only the parent's code is shown
    assert not any(c in prompts[0] for i, c in codes.items() if i != parent["id"])
    assert "depend on history" in prompts[0] and "Attempt dev scores" in prompts[0]
    assert len(out["admitted"]) == 2 and all(c["would_be_admitted"] is False for c in out["admitted"])
    assert out["candidate"] and out["evaluation"]["after"] is not None and out["evaluation"]["improved"] is True
    assert E.MODES["patch"] is E.escape_patch


def test_fallback_screens_out_the_stuck_approach_and_random_draws(fake_world, monkeypatch):
    from diverge import alternatives as A
    from diverge import server
    asked = []

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        asked.append(prompt)
        if prompt.startswith("You are proposing alternative"):
            alts = [{"name": n, "mechanism": f"{n} mechanism", "strength": "s", "trap": "t", "how_to_test": "h",
                     "test_cost": "minutes", "decisive": True, "kill_result": "k"}
                    for n in ("LinearModel", "Per-link offsets", "Queue buckets")]
            text = json.dumps({"alternatives": alts, "relax_hint": ""})
        elif prompt.startswith(A.SCREEN_MARKER):
            text = json.dumps({"screen": [{"index": 0, "duplicate_of": 0, "violates": None, "reason": "same fit"}]})
        else:
            text = "{}"
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"cost": 0.001}}
    monkeypatch.setattr(LLM, "_post", post)
    task = Task.resolve("Predict the delay of each recorded message.")
    diagnosis = {"shared_cause": "all 3 attempts implement a linear fit", "failure": "no progress after 3 attempts"}
    report = {"fingerprints": {"a0": {"mechanism_family": "Linear Regression", "decision_rule": {"primary_signal": "queue_len"}},
                               "a1": {"mechanism_family": "linear model", "decision_rule": {}},
                               "a2": {"mechanism_family": "Linear Regression", "decision_rule": {"primary_signal": "queue_len"}}}}
    lever = server.fallback_lever(pipeline.Session(progress=lambda m: None), task, [], [], "")(diagnosis, report)
    gen = next(p for p in asked if p.startswith("You are proposing alternative"))
    assert "all 3 attempts implement a linear fit: stalled: no progress after 3 attempts" in gen
    # each attempt's own reading is listed once, so a screen can match "LinearModel" against them
    assert gen.count("an attempt's strategy: Linear Regression, decides by queue_len") == 1
    assert "an attempt's strategy: linear model:" in gen
    assert lever["type"] == "agent_alternative" and lever["name"] == "Per-link offsets"  # LinearModel screened out
    draws = {server.fallback_lever(pipeline.Session(progress=lambda m: None), task, [], [], "", "random", seed)(
        diagnosis, {})["name"] for seed in range(12)}
    assert draws == {"Per-link offsets", "Queue buckets"}
