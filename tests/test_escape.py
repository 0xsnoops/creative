"""Diagnosis (repair vs escape), consensus anchors and the escape step, offline. The evaluator in the
end-to-end test is the real one; the implementer and diverge.trajectory.analyze are fakes."""
import sys
import types

import pytest

import diverge.pipeline as pipeline
from diverge.diagnose import anchors, diagnose
from diverge.escape import ESCAPE_MARKER, choose, escape, escape_prompt
from diverge.llm import LLM
from diverge.reality import Evaluator
from diverge.task import Task
from tests.test_pipeline_offline import BEST_FIT, FIRST_FIT, FUNSEARCH, fake_world, reply_to  # noqa: F401

INPUTS = ["item", "capacity_values", "bin_order", "new_slot", "history"]


def fingerprint(used):
    return {"inputs_used": used, "inputs_unused": [i for i in INPUTS if i not in used], "inputs_ambiguous": [],
            "state": {"persistent": False, "features": []},
            "decision_rule": {"primary_signal": "capacity_values", "secondary_signal": None},
            "assumptions": ["only the current item and capacities matter"], "mechanism_family": "Best Fit",
            "behavior_signature": "tightest fit", "failure_signature": {"weak_regimes": ["many mid-size items"]}}


def report(loop=True, improvement=0.0, weak=("many mid-size items",), n=3):
    used = ["item", "capacity_values"]
    return {"n": n, "fingerprints": {f"a{i}": fingerprint(used) for i in range(n)},
            "code_similarity": 0.4, "strategy_similarity": 0.95 if loop else 0.3,
            "shared": {"inputs_used": used, "inputs_unused": [i for i in INPUTS if i not in used],
                       "state_persistent": False, "primary_signal": "capacity_values", "mechanism_family": "Best Fit",
                       "assumptions": ["Only the current item and capacities matter"], "weak_regimes": list(weak)},
            "progress": {"first": 4.0, "best": 4.0 * (1 - improvement), "relative_improvement": improvement},
            "loop_score": 0.9 if loop else 0.2, "loop_detected": loop, "reasons": ["same strategy"] if loop else []}


def overlook_dict(history_share=0.03, assumptions=True, minority=True):
    row = lambda name, share, used: {"input": name, "description": f"{name} desc", "share": share, "used_by": used,
                                     "decided": 95, "overlooked": share <= 0.10, "used_by_models": {}, "evidence": []}
    return {"inputs": [row("history", history_share, round(history_share * 95)), row("bin_order", 0.2, 19),
                       row("item", 1.0, 95), row("capacity_values", 0.98, 93)],
            "assumptions": {"assumptions": [
                {"assumption": "Only the current item and the capacities matter", "kind": "incidental", "shared": True,
                 "held_by": 12, "checked": 12, "prompt_quote": None, "evidence": []},
                {"assumption": "Items must be placed immediately", "kind": "hard", "shared": True, "held_by": 12,
                 "checked": 12, "prompt_quote": "must each be placed in a bin immediately, without knowing future items",
                 "evidence": []}] if assumptions else []},
            "minority": {"families": [{"item": "Best Fit", "models": ["m1"], "absent_from": ["m2", "m3"], "samples": {},
                                       "count": 1},
                                      {"item": "Harmonic", "models": ["m2"], "absent_from": ["m1", "m3"], "samples": {},
                                       "count": 2}] if minority else []},
            "herd": {"samples": 100, "valid": 95, "top3": ["First Fit", "Best Fit"], "concentration_top3": 0.9},
            "paraphrase_test": None}


def test_diagnose_loop_switches_strategy():
    d = diagnose(None, report(), overlook_dict())
    assert d["action"] == "switch_strategy" and d["repairability"] == "low"
    assert "use an input the attempts never use: history" == d["target_properties"][0]  # rarest in the consensus first
    assert "a primary signal other than capacity_values" in d["target_properties"]
    assert "keep persistent state across calls" in d["target_properties"]
    assert d["shared_cause"] == ("all 3 attempts decide using only item and capacity_values, keep no state, "
                                 "primary signal capacity_values, mechanism family Best Fit")
    assert "many mid-size items" in d["failure"]
    # min over marginals: uses item 1.0, uses capacity_values 0.98, not history 0.97, not bin_order 0.8
    assert d["herd_prevalence"] == pytest.approx(0.8)
    assert diagnose(None, report(weak=()))["repairability"] == "medium"
    assert diagnose(None, report(weak=()))["failure"] == "no progress after 3 attempts"


def test_diagnose_continue_and_repair():
    assert diagnose(None, report(improvement=0.05))["action"] == "continue"
    assert diagnose(None, report(loop=False, improvement=0.001))["action"] == "repair"
    polished = diagnose(None, report(), phrase=lambda t: t.upper())
    assert polished["shared_cause"].startswith("ALL 3") and polished["action"] == "switch_strategy"


def test_anchors_separate_required_details_from_habits():
    task = "Build a URL shortener service. Store links in Redis and hash with SHA-256. Keep it simple."
    texts = [("We use Redis. " if i < 9 else "We use Postgres. ") + ("Hash: SHA-256." if i < 2 else "") for i in range(10)]
    rows = {r["anchor"]: r for r in anchors(task, texts, required_text="Links must use SHA-256 hashes.")}
    assert rows["Redis"]["retained_by"] == 9 and rows["Redis"]["suspected_anchor"]
    assert rows["SHA-256"]["retained_by"] == 2 and rows["SHA-256"]["required"]
    assert not rows["SHA-256"]["suspected_anchor"]
    assert "Build" not in rows and "Keep" not in rows  # sentence-initial words are not names
    assert [r["anchor"] for r in anchors(task, texts, extract=lambda t: ["redis"])] == ["redis"]


def test_choose_prefers_an_overlooked_unused_input_then_falls_back():
    tr = report()
    c = choose(diagnose(None, tr, overlook_dict()), tr, overlook_dict())
    assert c["lever"]["type"] == "unused_signal" and c["lever"]["name"] == "history"
    assert "3/95 consensus samples" in c["why"] and "none of your 3 attempts" in c["why"]
    assert [a["type"] for a in c["alternatives"]] == ["assumption_break", "minority_mechanism"]

    common = overlook_dict(history_share=0.5)
    c = choose(None, tr, common)
    assert c["lever"]["type"] == "assumption_break" and "12/12" in c["why"]

    c = choose(None, tr, overlook_dict(history_share=0.5, assumptions=False))
    assert c["lever"] == {"type": "minority_mechanism", "name": "Harmonic", "detail": c["lever"]["detail"]}  # not Best Fit
    assert "1/3 sampler models" in c["why"]

    bare = overlook_dict(history_share=0.5, assumptions=False, minority=False)
    c = choose(None, tr, bare, families=[{"family": "Best Fit"}, {"family": "Sum of squares", "share": 0.0,
                                                                   "observed": "0 of 95 labeled samples"}])
    assert c["lever"]["type"] == "blind_spot" and c["lever"]["name"] == "Sum of squares"
    assert "0 of 95" in c["why"]
    assert choose(None, tr, bare)["lever"] is None


def test_escape_prompt_has_lever_constraints_and_cause():
    task = Task.resolve("online_bin_packing")
    tr = report()
    d = diagnose(task, tr, overlook_dict())
    c = choose(d, tr, overlook_dict())
    prompt = escape_prompt(task, c, d, BEST_FIT * 400)
    assert prompt.startswith(task.prompt) and ESCAPE_MARKER in prompt
    assert d["shared_cause"] in prompt and "history" in prompt and "3/95" in prompt
    assert '"must each be placed in a bin immediately, without knowing future items"' in prompt
    assert prompt.count("def priority") <= 4000 // len(BEST_FIT) + 2  # current code truncated
    assert "held-out" not in prompt.lower() and "heldout" not in prompt.lower()


def test_escape_end_to_end(fake_world, monkeypatch):
    def fake_post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        text = f"```python\n{FUNSEARCH}```" if ESCAPE_MARKER in prompt else reply_to(prompt, body["temperature"])
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001, "completion_tokens": 10}}
    monkeypatch.setattr(LLM, "_post", fake_post)
    seen = {}

    def analyze(task, attempts):
        seen["attempts"] = attempts
        return report()
    stub = types.ModuleType("diverge.trajectory")
    stub.analyze = analyze
    monkeypatch.setitem(sys.modules, "diverge.trajectory", stub)
    import diverge as pkg
    monkeypatch.setattr(pkg, "trajectory", stub, raising=False)

    h = pipeline.Session(progress=lambda m: None)
    task = Task.resolve("online_bin_packing")
    best_fit_dev = task.baselines()["best_fit"]["score"]
    rescaled = "def priority(item, capacities):\n    return [-1000.0 * (c - item) + 7 for c in capacities]\n"
    previous = [{"id": "a0", "code": rescaled, "score": best_fit_dev}, {"id": "a1", "code": BEST_FIT, "score": best_fit_dev}]
    result = escape(h, task, {"id": "a2", "code": BEST_FIT.replace("item)", "item) + 0.0")}, previous,
                    Evaluator(task), overlook=overlook_dict())
    assert [a["id"] for a in seen["attempts"]] == ["a0", "a1", "a2"]
    assert seen["attempts"][2]["score"] == pytest.approx(best_fit_dev)  # current scored on dev
    assert result["action"] == "switch_strategy" and result["loop_detected"]
    assert result["overlooked_levers"][0]["name"] == "history"
    assert "max_bin_cap" in result["candidate"] and result["candidates"][0]["valid"]
    ev = result["evaluation"]
    assert ev["after"] < ev["before"] and ev["improved"]

    # Not a loop: diagnosis only, no candidate.
    stub.analyze = lambda task, attempts: report(loop=False)
    result = escape(h, task, BEST_FIT, [FIRST_FIT], Evaluator(task), overlook=overlook_dict())
    assert result["action"] == "repair" and result["candidate"] is None and result["evaluation"]["after"] is None


def test_escape_with_the_real_trajectory_module(fake_world, monkeypatch):
    pytest.importorskip("diverge.trajectory")

    def fake_post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        text = f"```python\n{FUNSEARCH}```" if ESCAPE_MARKER in prompt else reply_to(prompt, body["temperature"])
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001, "completion_tokens": 10}}
    monkeypatch.setattr(LLM, "_post", fake_post)
    task = Task.resolve("online_bin_packing")
    rescaled = "def priority(item, capacities):\n    return [-1000.0 * (c - item) + 7 for c in capacities]\n"
    result = escape(pipeline.Session(progress=lambda m: None), task, BEST_FIT, [rescaled, BEST_FIT.replace("item)", "item) + 0.0")],
                    Evaluator(task), overlook=overlook_dict())
    assert result["action"] in {"switch_strategy", "repair", "continue"}
    assert result["evaluation"]["before"] == pytest.approx(task.baselines()["best_fit"]["score"])
    if result["action"] == "switch_strategy":
        assert result["evaluation"]["improved"]
