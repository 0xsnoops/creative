"""Regression tests for the confirmed issues from the outline-only model review
(results/model_review/review-20261018-7e4cb73/report.md, section 6). Each test names its issue."""
import json

import pytest

import herd.pipeline as pipeline
from herd.llm import LLM, BudgetExceeded, Ledger, ask_json
from herd.reality import Evaluator, slug
from herd.task import Task
from tests.test_pipeline_offline import MODELS, call_tool, fake_world  # noqa: F401  (fixture)

BIN = "online_bin_packing"


def herd():
    return pipeline.Herd(progress=lambda m: None)


# I2: the thesis benchmark needs a big enough map, and a count-matched literature control.
def test_I2_bench_refuses_a_small_map_without_override(fake_world):
    from herd.bench import run_bench
    with pytest.raises(SystemExit, match="samples per model"):
        run_bench(BIN, budget_usd=0.25, repeats=1, operators=["blind_spot"], progress=lambda m: None)


def test_I2_literature_random_matches_the_blind_spot_count(fake_world):
    from herd.operators import run_operator
    task = Task.resolve(BIN)
    study = pipeline.ensure_map(herd(), task)
    run = run_operator("literature_random", task, 0.25, seed=0, study=study, progress=lambda m: None)
    sourced = {f["family"] for f in study["families"] if f["source_urls"]}
    assert len(run["notes"]["families"]) == len(study["blind_spots"])
    assert set(run["notes"]["families"]) <= sourced
    again = run_operator("literature_random", task, 0.25, seed=0, study=study, progress=lambda m: None)
    assert again["notes"]["families"] == run["notes"]["families"]  # seeded


# I3: dev evaluator detail is kept with each candidate and implementation.
def test_I3_dev_detail_is_kept(fake_world):
    task = Task.resolve(BIN)
    study = pipeline.ensure_map(herd(), task)
    best_fit = next(f for f in study["families"] if f["family"] == "Best Fit")
    s = pipeline.ensure_family_scores(herd(), task, study, [best_fit], Evaluator(task), k=1)["Best Fit"]
    detail = s["implementations"][0]["dev_detail"]
    assert detail["split"] == "dev" and len(detail["instances"]) == 5


# I4: saved scores are keyed by score direction and by the inputs that produced them.
def test_I4_evaluator_key_includes_direction_and_inputs():
    task = Task.resolve("some free-text task")
    assert Evaluator(task, "python e.py", True).key != Evaluator(task, "python e.py", False).key
    other = Task.resolve("another free-text task")
    assert Evaluator(task, "python e.py").key != Evaluator(other, "python e.py").key


def test_I4_changed_family_text_is_rescored(fake_world):
    task = Task.resolve(BIN)
    study = pipeline.ensure_map(herd(), task)
    fam = dict(next(f for f in study["families"] if f["family"] == "Best Fit"))
    pipeline.ensure_family_scores(herd(), task, study, [fam], Evaluator(task), k=1)
    calls = fake_world["llm"]
    fam["one_line_mechanism"] = "a different mechanism text"
    pipeline.ensure_family_scores(herd(), task, study, [fam], Evaluator(task), k=1)
    assert fake_world["llm"] > calls


# I5: labeling cost is in the preflight, and a budget stop during labeling keeps the results.
def test_I5_labeling_budget_stop_keeps_operator_results(fake_world, monkeypatch):
    import herd.bench as bench
    def broke(*a, **k):
        raise BudgetExceeded("global budget reached")
    monkeypatch.setattr(bench, "label_samples", broke)
    table, verdict, bench_dir = bench.run_bench(BIN, budget_usd=0.25, repeats=1, operators=["plain"],
                                                progress=lambda m: None, allow_small_map=True)
    data = json.loads((bench_dir / "bench.json").read_text())
    assert data["labeling"]["status"] == "stopped" and table["plain"]["n_candidates"]["mean"] > 0


def test_I5_preflight_includes_labeling_estimate(fake_world):
    import herd.bench as bench
    estimate = bench.labeling_estimate(herd(), operators=["plain", "hot"], repeats=2)
    assert estimate > 0


# I6: completed sample counts drive confidence; maps with failed samples are reused.
def test_I6_partial_sampling_is_recorded_and_reused(fake_world, monkeypatch):
    real_post = LLM._post
    def flaky(self, body, attempts=4):
        if body["model"] == MODELS[1] and body["messages"][0]["content"].startswith("# Online bin packing"):
            raise RuntimeError("provider down")
        return real_post(self, body, attempts)
    monkeypatch.setattr(LLM, "_post", flaky)
    study = pipeline.ensure_map(herd(), Task.resolve(BIN))
    meta = study["herd_map"]["meta"]
    assert meta["completed_per_model"] == {MODELS[0]: 3, MODELS[1]: 0}
    assert meta["min_completed_per_model"] == 0
    calls = fake_world["llm"]
    pipeline.ensure_map(herd(), Task.resolve(BIN))
    assert fake_world["llm"] == calls  # reused, not rebuilt


# I7: herd_check returns a partial comparison when the budget runs out.
def test_I7_herd_check_returns_partial_results(fake_world, monkeypatch):
    task = Task.resolve(BIN)
    study = pipeline.ensure_map(herd(), task)
    real = pipeline.ensure_family_scores
    seen = []
    def limited(h, t, s, families, ev, k, *rest):
        seen.extend(f["family"] for f in families)
        if len(seen) > 1:
            raise BudgetExceeded("run budget reached")
        return real(h, t, s, families, ev, k, *rest)
    monkeypatch.setattr(pipeline, "ensure_family_scores", limited)
    result = pipeline.check_approach(herd(), task, study, "first fit", Evaluator(task), k=1)
    assert result["complete"] is False and result["stopped_at"] == seen[1]
    assert result["family"] == "First Fit" and "First Fit" in result["tested"]


# I8: replies missing required keys are retried, then raise ValueError; run_tool reports any error.
def test_I8_missing_keys_are_retried_then_rejected(fake_world, monkeypatch):
    monkeypatch.setattr(LLM, "_post", lambda self, body, attempts=4: {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {"cost": 0.0}})
    with pytest.raises(ValueError, match="families"):
        ask_json(herd().llm, MODELS[0], "anything", required=("families",))


def test_I8_run_tool_turns_any_exception_into_a_structured_error(fake_world, monkeypatch):
    import herd.server as server
    def boom(*a, **k):
        raise KeyError("families")
    monkeypatch.setattr(server, "ensure_map", boom)
    result, _ = call_tool("herd_map", {"task_description": BIN})
    assert result["error"] == "KeyError" and "spent_usd" in result


# I9: repeated held-out comparisons are flagged as exploratory.
def test_I9_repeat_heldout_use_is_flagged(fake_world):
    first, _ = call_tool("herd_check", {"task_description": BIN, "my_approach": "first fit", "k": 1})
    second, _ = call_tool("herd_check", {"task_description": BIN, "my_approach": "first fit", "k": 1})
    assert first["heldout_prior_uses"] == 0 and first["exploratory"] is False
    assert second["heldout_prior_uses"] >= 1 and second["exploratory"] is True


# I10: reservations carry a margin, and settled costs above the reservation are counted.
def test_I10_overruns_are_counted():
    ledger = Ledger(global_cap_usd=10, global_spent_usd=0)
    ledger.reserve(0.1)
    ledger.settle(0.1, 0.15)
    assert ledger.overruns == 1


def test_I10_reservation_includes_completion_margin(fake_world, monkeypatch):
    h = herd()
    amounts = []
    real = h.ledger.reserve
    monkeypatch.setattr(h.ledger, "reserve", lambda a: (amounts.append(a), real(a))[1])
    h.llm.chat(MODELS[0], [{"role": "user", "content": "x"}], max_tokens=1000, sample_idx="margin")
    assert amounts[0] == pytest.approx(1 / 3 * 1e-6 + 1000 * 2e-6 * 1.25)


# I12: blind spots report how many sampler models never used them.
def test_I12_blind_spots_report_models_missing(fake_world):
    study = pipeline.ensure_map(herd(), Task.resolve(BIN))
    assert all(b["models_missing"] == len(MODELS) for b in study["blind_spots"])


# I14: MCP reality_check says which herd families it compared against.
def test_I14_reality_check_names_compared_families(fake_world):
    rc, _ = call_tool("reality_check", {"task_description": BIN, "family": "Best Fit", "k": 1})
    assert rc["compared_herd_families"] == [] and rc["best_tested_herd"] is None


# I16: bench metrics respect the evaluator's direction.
def test_I16_run_metrics_uses_evaluator_direction(tmp_path):
    from herd.bench import run_metrics
    class HigherIsBetter:
        lower_is_better = False
        def best(self, xs): return max(xs)
        def beats(self, a, b): return a > b
        def final(self, path, dev): return dev, "dev"
    cands = [{"operator": "x", "seed": 0, "idx": i, "valid": True, "score": s, "code_path": f"{i}.py"}
             for i, s in enumerate([1.0, 5.0])]
    labels = {f"x-s0-{i:03d}": {"family": "F"} for i in range(2)}
    m = run_metrics({"operator": "x", "seed": 0, "spent_usd": 1.0, "notes": {}, "candidates": cands},
                    labels, HigherIsBetter(), tmp_path, best_baseline=3.0, setup=0.0)
    assert m["best_score"] == 5.0 and m["useful_discoveries"] == 1


# I17: implementation file names cannot collide.
def test_I17_long_family_names_get_distinct_slugs():
    a, b = "x" * 70 + " alpha", "x" * 70 + " beta"
    assert slug(a) != slug(b)


# I18: empty task descriptions are rejected.
def test_I18_empty_task_description_is_rejected():
    with pytest.raises(ValueError):
        Task.resolve("   ")
