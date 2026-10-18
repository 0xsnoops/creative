"""Offline tests for the operators and measurements added after the model review: control flow,
the exact prompt contents, accounting, fingerprints, the MCP contrast flag and traces."""
import json
import random

import pytest

import diverge.pipeline as pipeline
from diverge import fingerprint
from diverge.operators import run_operator
from diverge.operators.adhd_frames import FRAMES, select_frames
from diverge.task import Task
from tests.test_pipeline_offline import BEST_FIT, FIRST_FIT, call_tool, fake_world  # noqa: F401

BIN = "online_bin_packing"


def study_and_task():
    task = Task.resolve(BIN)
    return pipeline.ensure_map(pipeline.Herd(progress=lambda m: None), task), task


def run(name, budget=0.3, seed=0):
    study, task = study_and_task()
    return run_operator(name, task, budget, seed=seed, study=study, progress=lambda m: None), study


def test_contrast_lists_the_herd_and_the_control_lists_random_others(fake_world):
    herd_run, study = run("blind_spot_contrast")
    top3, blind = study["herd_map"]["top3"], {b["family"] for b in study["blind_spots"]}
    c = herd_run["candidates"][0]
    assert c["contrast_kind"] == "herd" and set(c["contrast_families"]) == set(top3) - {c["intended_family"]}
    assert "Mechanisms your solution should differ from" in c["prompt"]
    for name in c["contrast_families"]:
        assert f"- {name}:" in c["prompt"]
    rnd, _ = run("blind_spot_contrast_random")
    r = rnd["candidates"][0]
    assert r["contrast_kind"] == "random" and r["contrast_families"]
    assert not set(r["contrast_families"]) & set(top3) and r["intended_family"] not in r["contrast_families"]
    assert "Mechanisms your solution should differ from" in r["prompt"]  # identical wording
    plain, _ = run("blind_spot")
    assert "should differ from" not in plain["candidates"][0]["prompt"]


def test_feedback_uses_the_familys_own_dev_outcome_and_never_heldout(fake_world):
    result, study = run("blind_spot_feedback", budget=0.5)
    by_round = {}
    for c in result["candidates"]:
        by_round.setdefault(c["feedback_round"], []).append(c)
    assert all("Previous attempt" not in c["prompt"] for c in by_round[0])
    later = by_round[1]
    harmonic = next(c for c in later if c["intended_family"] == "Harmonic")
    best_fit = next(c for c in later if c["intended_family"] == "Best Fit")
    assert harmonic["feedback_from"] == "Harmonic" and "It was invalid. Error:" in harmonic["prompt"]
    assert "broken" in harmonic["prompt"]  # the crashing implementation's own error
    assert best_fit["feedback_from"] == "Best Fit" and "weakest development case: instance seed" in best_fit["prompt"]
    assert all("held" not in c["prompt"].lower() for c in result["candidates"])
    assert result["spent_usd"] <= 0.5


def test_sham_feedback_comes_from_another_family(fake_world):
    result, _ = run("blind_spot_feedback_sham", budget=0.5)
    later = [c for c in result["candidates"] if c["feedback_round"] == 1]
    assert later and all(c["feedback_from"] != c["intended_family"] for c in later)


def test_adhd_frames_follows_adhds_selection_and_bills_planning(fake_world):
    frames = select_frames(random.Random(1))
    tags = {f[0]: f[3] for f in FRAMES}
    assert len(frames) == 5 and len({f[0] for f in frames}) == 5 and sum("wild" in tags[f[0]] for f in frames) >= 1
    assert sum(("code" in tags[f[0]] or "design" in tags[f[0]]) for f in frames) >= 4
    calls = fake_world["llm"]
    result, _ = run("adhd_frames", budget=0.3)
    first = result["notes"]["rounds"][0]
    assert first["frames"] == [f[0] for f in select_frames(random.Random("adhd:0:0"))]
    assert result["candidates"] and all(c["frame"] in first["frames"] for c in result["candidates"])
    assert "idea 0" in result["candidates"][0]["prompt"]
    assert "MIT" in result["notes"]["attribution"]
    diverge_calls = len(first["ideas"])
    assert fake_world["llm"] - calls >= diverge_calls + len(result["candidates"])  # planning calls happened
    assert 0 < result["spent_usd"] <= 0.3


def test_fingerprint_gate_and_clusters(fake_world, tmp_path):
    task = Task.resolve(BIN)
    gate = fingerprint.validation_gate(task)
    assert gate["passed"] and gate["max_copy_distance"] == 0.0
    paths = {}
    for name, code in {"ff": FIRST_FIT, "bf": BEST_FIT, "bf2": BEST_FIT.replace("-(c - item)", "-2 * (c - item)")}.items():
        paths[name] = tmp_path / f"{name}.py"
        paths[name].write_text(code)
    groups = fingerprint.clusters(fingerprint.fingerprints(task, paths))
    assert sorted(map(sorted, groups)) == [["bf", "bf2"], ["ff"]]
    agreement = fingerprint.label_agreement(groups, {"ff": "First Fit", "bf": "Best Fit", "bf2": "Other"})
    assert agreement["clusters_with_several_labels"] == 1


def test_bench_reports_behavior_clusters_and_candidate_traces(fake_world, tmp_path, monkeypatch):
    from diverge.bench import run_bench
    monkeypatch.setenv("HERD_TRACE_DIR", str(tmp_path / "trace"))
    table, _, bench_dir = run_bench(BIN, budget_usd=0.25, repeats=1, operators=["blind_spot", "blind_spot_contrast"],
                                    progress=lambda m: None, allow_small_map=True)
    data = json.loads((bench_dir / "bench.json").read_text())
    assert data["behavior"]["gate"]["passed"] and table["blind_spot"]["behavior_clusters"]["mean"] >= 1
    rows = [json.loads(l) for l in (tmp_path / "trace" / "candidates.jsonl").read_text().splitlines()]
    assert rows and all(r["prompt"] and "operator" in r for r in rows)
    assert any(r["selected"] for r in rows) and any(r["heldout_score"] is not None for r in rows)
    calls = [json.loads(l) for l in (tmp_path / "trace" / "calls.jsonl").read_text().splitlines()]
    assert calls and all(c["phase"] == "experiment" and "messages" in c for c in calls if c["kind"] == "llm")


def test_mcp_contrast_flag(fake_world):
    check, _ = call_tool("diverge_check", {"task_description": BIN, "my_approach": "first fit", "k": 1,
                                        "contrast": True})
    assert check["contrast"] is True and check["verdict"] in ("keep", "try_blind_spot", "try_other_family")
    study, task = study_and_task()
    from diverge.reality import Evaluator
    saved = json.loads((pipeline.ROOT / "results" / BIN / "family_scores.json").read_text())
    contrast_scores = saved[Evaluator(task).key + "|contrast"]
    assert contrast_scores["Best Fit"]["contrast"] is True and contrast_scores["First Fit"]["contrast"] is False
