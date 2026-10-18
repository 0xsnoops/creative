"""diverge.contract offline: verdict branches, render's first line, evidence classes, scope note."""
import json
from pathlib import Path

from diverge.contract import parse, render, verdict

RESULTS = Path(__file__).resolve().parents[1] / "results" / "online_bin_packing"


def _local(name):
    """Local research output (gitignored): skip the test in a fresh checkout without it."""
    import pytest
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"needs local research output results/online_bin_packing/{name}")
    return path
URL = "https://doi.org/10.1145/3828.3833"


def lever(type_="unused_signal", name="history", **detail):
    base = {"unused_signal": {"used_by": 38, "decided": 95, "share": 0.4,
                              "used_by_models": {"openai/gpt-6.1-sol": 19, "anthropic/claude-sonnet-5.5": 4}},
            "blind_spot": {"one_line_mechanism": "pack by harmonic size class", "observed": "0/95",
                           "source_urls": []}}.get(type_, {})
    return {"type": type_, "name": name, "detail": {**base, **detail},
            "mechanism": f"Build the decision rule around {name}.", "why": f"{name} is rarely used"}


def result(action="switch_strategy", scores=(0.042, 0.0415, 0.0418), levers=None, before=0.0418, sim=0.9,
           candidates=None, after=None, **extra):
    levers = [lever()] if levers is None else levers
    attempts = [{"id": f"attempt-{i}", "score": s, "valid": s is not None} for i, s in enumerate(scores)]
    valid = [s for s in scores if s is not None]
    return {"loop_detected": action == "switch_strategy", "strategy_similarity": sim, "code_similarity": 0.35,
            "shared_strategy": "all 3 attempts decide using only item, capacity", "herd_prevalence": 0.6,
            "overlooked_levers": levers,
            "recommended_switch": {"mechanism": levers[0]["mechanism"] if levers else None,
                                   "why": levers[0]["why"] if levers else "no lever found"},
            "candidate": None, "candidate_path": "/tmp/escape/history-0-0.py" if candidates else None,
            "candidates": candidates or [],
            "evaluation": {"before": before, "after": after, "improved": after is not None and after < before,
                           "split": "dev"},
            "action": action,
            "diagnosis": {"action": action, "evidence": {"progress": {
                "first": valid[0] if valid else None, "best": min(valid) if valid else None,
                "relative_improvement": 0.01 if len(valid) >= 2 else None}}},
            "reasons": ["3/3 attempts use only inputs item, capacity", "3/3 attempts assume: bins are independent",
                        "mean pairwise strategy similarity 0.90 against code similarity 0.35"],
            "attempts": attempts, "spent_usd": 0.012, **extra}


# ------------------------------------------------------------------ verdict branches
def test_inconclusive_on_error():
    assert verdict({"error": "budget_exceeded", "detail": "cap", "spent_usd": 2.0}) == "inconclusive"


def test_inconclusive_with_fewer_than_two_scored_attempts():
    assert verdict(result(scores=(None, 0.04))) == "inconclusive"
    assert verdict(result(scores=(0.04,))) == "inconclusive"


def test_inconclusive_without_a_current_score():
    assert verdict(result(before=None)) == "inconclusive"


def test_keep_repair_escape():
    assert verdict(result(action="continue")) == "keep"
    assert verdict(result(action="repair", sim=0.9)) == "repair"
    assert verdict(result()) == "escape"


def test_escalate_when_switching_without_a_lever():
    r = result(levers=[])
    assert r["recommended_switch"]["mechanism"] is None and verdict(r) == "escalate"


def test_escalate_when_everything_is_invalid():
    r = result(scores=(None, None, None), before=None,
               candidates=[{"path": "x.py", "valid": False, "score": None, "error": "boom"}])
    assert verdict(r) == "escalate"


def test_escalate_when_different_strategies_score_identically():
    assert verdict(result(action="repair", scores=(0.04, 0.04, 0.04), before=0.04, sim=0.3)) == "escalate"
    # same strategy with identical scores is a loop, not an evaluator problem
    assert verdict(result(scores=(0.04, 0.04, 0.04), before=0.04, sim=0.9)) == "escape"


def test_verdict_falls_back_to_progress_without_attempt_list():
    r = result()
    del r["attempts"]
    assert verdict(r) == "escape"
    r["diagnosis"]["evidence"]["progress"]["relative_improvement"] = None
    assert verdict(r) == "inconclusive"


# ------------------------------------------------------------------ render
def test_render_first_line_and_json_body():
    text = render(result())
    assert text.splitlines()[0] == "VERDICT: escape"
    body = json.loads(text.split("\n", 1)[1])
    assert body["lever"] == {"type": "unused_signal", "name": "history",
                             "mechanism": "Build the decision rule around history.", "why": "history is rarely used"}
    nx = body["next_experiment"]
    assert nx["hypothesis"] == "Making decisions depend on history will beat the best attempt's dev score 0.0415."
    assert nx["success_threshold"] == "dev score <= 0.041085 (lower is better) and valid"
    assert nx["failure_threshold"] == "dev score no better than 0.0415 after 3 candidates, or invalid"
    assert "measured" not in nx
    assert body["spent_usd"] == 0.012 and body["candidate_path"] is None


def test_render_includes_measured_candidate():
    r = result(candidates=[{"path": "/tmp/escape/history-0-0.py", "valid": True, "score": 0.039}], after=0.039)
    v, body = parse(render(r))
    assert v == "escape"
    assert body["next_experiment"]["measured"] == {"before": 0.0418, "after": 0.039, "improved": True, "split": "dev"}
    assert "after 1 candidate, or invalid" in body["next_experiment"]["failure_threshold"]
    assert body["candidate_path"] == "/tmp/escape/history-0-0.py"
    assert any(e["claim"].startswith("escape candidate dev score 0.039") and e["class"] == "MEASURED"
               for e in body["evidence"])


def test_render_error_starts_inconclusive():
    v, body = parse(render({"error": "evaluator_required", "detail": "pass evaluator_cmd", "spent_usd": 0}))
    assert v == "inconclusive" and body == {"error": "evaluator_required", "detail": "pass evaluator_cmd",
                                            "spent_usd": 0}


# ------------------------------------------------------------------ evidence classes
def classes(body):
    return {e["claim"]: e for e in body["evidence"]}


def test_lever_herd_counts_are_measured_and_mechanism_hypothesized():
    _, body = parse(render(result()))
    ev = classes(body)
    assert ev["history used by 38/95 herd samples"]["class"] == "MEASURED"
    assert ev["history used by 38/95 herd samples"]["detail"]["used_by_models"]["openai/gpt-6.1-sol"] == 19
    assert ev["Expected effect: Build the decision rule around history."]["class"] == "HYPOTHESIZED"
    assert ev["mean pairwise strategy similarity 0.90 against code similarity 0.35"]["class"] == "MEASURED"
    assert sum(e["claim"].startswith("mean pairwise strategy similarity") for e in body["evidence"]) == 1
    assert ev["attempt dev scores: attempt-0=0.042, attempt-1=0.0415, attempt-2=0.0418"]["class"] == "MEASURED"
    # model-extracted assumptions are never MEASURED
    assert ev["3/3 attempts assume: bins are independent"]["class"] == "HYPOTHESIZED"
    assert ev["3/3 attempts use only inputs item, capacity"]["class"] == "MEASURED"
    assert not any(e["class"] == "SOURCE_BACKED" for e in body["evidence"])


def test_taxonomy_family_lever_is_source_backed_with_url():
    fam = lever("blind_spot", "Harmonic size-class packing")
    families = [{"family": "Harmonic size-class packing", "one_line_mechanism": "pack by harmonic size class",
                 "source_summary": "Harmonic-M packs similar-sized items together.", "source_urls": [URL]}]
    v, body = parse(render(result(levers=[fam]), families=families))
    assert v == "escape" and body["lever"]["type"] == "blind_spot"
    assert body["next_experiment"]["hypothesis"].startswith("Switching to Harmonic size-class packing will beat")
    src = [e for e in body["evidence"] if e["class"] == "SOURCE_BACKED"]
    assert len(src) == 1 and src[0]["url"] == URL and "Harmonic" in src[0]["claim"]
    assert classes(body)["Harmonic size-class packing observed in the herd: 0/95"]["class"] == "MEASURED"


def test_source_backed_only_with_a_real_url():
    _, body = parse(render(result(levers=[lever("blind_spot", "Unsourced")]), families=[]))
    assert not any(e["class"] == "SOURCE_BACKED" for e in body["evidence"])


def test_real_taxonomy_families_give_urls():
    families = json.loads(_local("families.json").read_text())
    name = next(f["family"] for f in families if f.get("source_urls"))
    _, body = parse(render(result(levers=[lever("blind_spot", name)]), families=families))
    src = next(e for e in body["evidence"] if e["class"] == "SOURCE_BACKED")
    assert src["url"].startswith("http")


# ------------------------------------------------------------------ scope note
def test_scope_note_with_paraphrase():
    para = json.loads(_local("paraphrase.json").read_text())
    _, body = parse(render(result(), paraphrase=para))
    assert body["scope_note"] == ("Herd shares are for this prompt wording: rewording moved the map (paraphrase "
                                  "TVD 0.51 vs at most 0.40 between resamples; verdict: partly a prompt effect).")


def test_scope_note_without_paraphrase():
    _, body = parse(render(result()))
    assert body["scope_note"] == ("Herd shares are for this prompt wording; robustness to rewording was not "
                                  "measured for this task.")


def test_render_is_json_serializable_on_real_overlook():
    overlook = json.loads(_local("overlook.json").read_text())
    row = next(r for r in overlook["inputs"] if r["input"] == "n_open_bins")
    lv = lever("unused_signal", "n_open_bins", **{k: row[k] for k in ("used_by", "decided", "share", "used_by_models")})
    text = render(result(levers=[lv]), overlook=overlook)
    _, body = parse(text)
    assert classes(body)["n_open_bins used by 5/95 herd samples"]["class"] == "MEASURED"
