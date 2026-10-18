"""Offline tests for the operator variants that answer the operator-benchmark review: a
hybrid-allowed template (blind_spot_hybrid) and a different design direction per retry
(blind_spot_directions). The existing blind_spot prompt must stay byte-identical."""
from diverge.operators import registry, run_operator
from diverge.operators.blind_spot_directions import DIRECTIONS
from diverge.reality import HYBRID_MARKER, HYBRID_TEMPLATE, IMPLEMENT_TEMPLATE, implementation_prompt
from tests.test_new_operators import study_and_task
from tests.test_pipeline_offline import fake_world  # noqa: F401

NEW = ["blind_spot_hybrid", "blind_spot_directions", "blind_spot_directions_hybrid"]


def run(name, budget=0.3, seed=0):
    study, task = study_and_task()
    return run_operator(name, task, budget, seed=seed, study=study, progress=lambda m: None), study, task


def test_new_operators_are_registered():
    reg = registry()
    assert all(name in reg for name in NEW)
    assert "blind_spot" in reg  # the matched control is untouched


def test_hybrid_prompt_allows_hybrids_and_drops_the_faithful_clause(fake_world):
    result, study, task = run("blind_spot_hybrid")
    assert result["candidates"] and 0 < result["spent_usd"] <= 0.3
    assert result["notes"]["families"] == [b["family"] for b in study["blind_spots"]]
    for c in result["candidates"]:
        p = c["prompt"]
        assert c["template"] == "hybrid" and c["intended_family"] in result["notes"]["families"]
        assert p.startswith(task.prompt) and f"Family: {c['intended_family']}" in p
        assert HYBRID_MARKER in p and "components from other mechanisms" in p and "hybrid is welcome" in p
        assert "faithfully" not in p and "within the family" not in p
    assert "faithfully" not in HYBRID_TEMPLATE and "within the family" not in HYBRID_TEMPLATE


def test_directions_differ_per_retry_of_the_same_family(fake_world):
    result, study, _ = run("blind_spot_directions")
    assert result["candidates"] and 0 < result["spent_usd"] <= 0.3
    by_family = {}
    for c in result["candidates"]:
        by_family.setdefault(c["intended_family"], []).append(c)
    assert len(by_family) == len(study["blind_spots"])
    assert any(len(cs) >= 2 for cs in by_family.values())  # retries happened within the budget
    tags = [tag for tag, _ in DIRECTIONS]
    for cs in by_family.values():
        cs.sort(key=lambda c: c["direction_round"])
        assert [c["direction_round"] for c in cs] == list(range(len(cs)))
        assert [c["direction"] for c in cs] == tags[:len(cs)]
        assert len({c["prompt"] for c in cs}) == len(cs)  # successive prompts differ
        assert len({c["idx"] for c in cs}) == len(cs)  # so do their sample indices (cache keys)
        for c in cs:
            text = dict(DIRECTIONS)[c["direction"]]
            assert text in c["prompt"] and "Implement this approach faithfully" in c["prompt"]
            assert c["prompt"].index(text) < c["prompt"].index("Implement this approach faithfully")
    assert all(c["idx"] != d["idx"] for c in result["candidates"] for d in result["candidates"] if c is not d)


def test_directions_hybrid_combines_both_levers(fake_world):
    result, _, _ = run("blind_spot_directions_hybrid")
    assert result["candidates"] and result["notes"]["template"] == "hybrid"
    for c in result["candidates"]:
        assert c["template"] == "hybrid" and HYBRID_MARKER in c["prompt"] and "faithfully" not in c["prompt"]
        assert dict(DIRECTIONS)[c["direction"]] in c["prompt"]


def test_blind_spot_prompts_are_unchanged(fake_world):
    result, study, task = run("blind_spot")
    assert result["candidates"]
    for c in result["candidates"]:
        family = next(b for b in study["blind_spots"] if b["family"] == c["intended_family"])
        expected = IMPLEMENT_TEMPLATE.format(task=task.prompt, family=family["family"],
                                             mechanism=family["one_line_mechanism"],
                                             source_summary=family["source_summary"])
        assert c["prompt"] == expected == implementation_prompt(task, family)
        assert "Implement this approach faithfully and make it as strong as you can within the family." in c["prompt"]
        assert "Design direction" not in c["prompt"] and HYBRID_MARKER not in c["prompt"]
