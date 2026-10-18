"""diverge.plateau offline: plateau detection against built-in baselines, and the contract's extra
evidence line / next_action sentence (only when stalled at the best known baseline)."""
import json
from pathlib import Path

from diverge import demo, plateau as pl
from diverge.contract import agent_result, parse, render, verdict

REPO = Path(__file__).resolve().parents[1]
BASELINES = {"best_fit": {"score": 4.146, "valid": True}, "first_fit": {"score": 4.545, "valid": True}}
LEVER = {"type": "unused_signal", "name": "n_open_bins", "mechanism": "open a new bin when few are open",
         "why": "unused by every attempt", "detail": {"used_by": 2, "decided": 10}}


def attempts(*scores):
    return [{"id": f"attempt-{i}", "score": s, "valid": s is not None} for i, s in enumerate(scores)]


def result(scores, action="switch_strategy", rel=0.0, levers=None, **extra):
    levers = [LEVER] if levers is None else levers
    valid = [s for s in scores if s is not None]
    return {"strategy_similarity": 0.9, "code_similarity": 0.35, "overlooked_levers": levers,
            "recommended_switch": {"mechanism": levers[0]["mechanism"] if levers else None, "why": "x"},
            "candidate": None, "candidate_path": None, "candidates": [],
            "evaluation": {"before": valid[-1], "after": None, "improved": False, "split": "dev"},
            "action": action,
            "diagnosis": {"action": action, "evidence": {"progress": {
                "first": valid[0], "best": min(valid), "relative_improvement": rel}}},
            "reasons": ["3/3 attempts use only inputs item, capacity"],
            "attempts": attempts(*scores), "lower_is_better": True, **extra}


# ------------------------------------------------------------------ plateau()
def test_detected_at_best_fit_score():
    p = pl.plateau(attempts(4.3, 4.146, 4.2), BASELINES)
    assert p == {"at_reference": True, "reference": "best_fit", "reference_score": 4.146,
                 "best_attempt": 4.146, "gap": 0.0, "eps": 0.005, "split": "dev"}


def test_detected_when_beating_the_baseline_by_less_than_eps():
    p = pl.plateau(attempts(4.145), BASELINES)
    assert p["at_reference"] is True and p["gap"] < 0


def test_not_detected_when_10_percent_worse():
    p = pl.plateau(attempts(4.146 * 1.1, 4.146 * 1.12), BASELINES)
    assert p["at_reference"] is False and p["reference"] == "best_fit" and p["gap"] > 0.4


def test_not_detected_when_clearly_better_than_the_baseline():
    assert pl.plateau(attempts(0.658), BASELINES)["at_reference"] is False


def test_none_without_baselines_or_scores():
    assert pl.plateau(attempts(4.146), {}) is None
    assert pl.plateau(attempts(4.146), None) is None
    assert pl.plateau(attempts(None), BASELINES) is None
    assert pl.plateau([], BASELINES) is None


def test_invalid_baselines_and_attempts_are_ignored():
    base = {"best_fit": {"score": 4.146, "valid": False}, "first_fit": {"score": 4.545, "valid": True}}
    p = pl.plateau([{"id": "a", "score": 4.146, "valid": False}, {"id": "b", "score": 4.545, "valid": True}], base)
    assert p["reference"] == "first_fit" and p["best_attempt"] == 4.545 and p["at_reference"] is True


def test_higher_is_better_picks_the_max_baseline():
    base = {"a": {"score": 0.80, "valid": True}, "b": {"score": 0.90, "valid": True}}
    p = pl.plateau(attempts(0.85, 0.90), base, lower_is_better=False)
    assert p["reference"] == "b" and p["best_attempt"] == 0.90 and p["at_reference"] is True


def test_attach_uses_the_results_attempts_and_direction():
    r = result((4.3, 4.146))
    assert pl.attach(r, BASELINES) is r["plateau"] and r["plateau"]["at_reference"] is True
    assert pl.attach(result((4.3, 4.146)), {}) is None


# ------------------------------------------------------------------ contract hook
def _render(r):
    v, body = parse(render(r))
    return v, body, [e["claim"] for e in body["evidence"]]


def test_contract_adds_evidence_and_next_action_when_stalled_at_reference():
    r = result((4.3, 4.146, 4.2))
    pl.attach(r, BASELINES)
    assert verdict(r) == "escape"
    v, body, claims = _render(r)
    line = next(c for c in claims if "best known baseline" in c)
    assert line == ("best attempt 4.146 equals the best known baseline best_fit 4.146 on dev; more variations of "
                    "this framing rarely help (preliminary rescue test: 0/3 such states rescued by any method)")
    item = next(e for e in body["evidence"] if e["claim"] == line)
    assert item["class"] == "MEASURED"
    assert body["plateau"]["at_reference"] is True
    text, structured = agent_result(r)
    assert structured["verdict"] == "escape"
    assert structured["next_action"].endswith(
        "If the lever does not beat best_fit (4.146 on dev), stop varying: reframe the problem (restate the "
        "objective/constraints, change the representation) or escalate to a human.")
    assert structured["next_action"].startswith("Implement the named lever")
    assert "best_fit (4.146 on dev)" in text  # markdown block carries it too


def test_numbers_in_the_text_come_from_the_result():
    r = result((5.0, 4.98))
    pl.attach(r, {"greedy": {"score": 5.0, "valid": True}})
    assert r["plateau"]["at_reference"] is True and r["plateau"]["gap"] != 0
    _, structured = agent_result(r)
    line = next(e["claim"] for e in structured["evidence"] if "best known baseline" in e["claim"])
    assert line.startswith("best attempt 4.98 matches (within 0.5%) the best known baseline greedy 5 on dev")
    assert "does not beat greedy (5 on dev)" in structured["next_action"]


def test_repair_verdict_gets_the_extension_too():
    r = result((4.146, 4.2), action="repair")
    pl.attach(r, BASELINES)
    assert verdict(r) == "repair"
    _, structured = agent_result(r)
    assert "If the repair does not beat best_fit (4.146 on dev)" in structured["next_action"]
    assert structured["next_action"].startswith("Fix the existing strategy")


def test_nothing_added_when_not_at_reference():
    r = result((4.6, 4.56))
    pl.attach(r, BASELINES)
    assert r["plateau"]["at_reference"] is False
    _, body, claims = _render(r)
    _, structured = agent_result(r)
    assert not any("best known baseline" in c for c in claims)
    assert "stop varying" not in structured["next_action"]


def test_nothing_added_when_not_stalled():
    r = result((4.6, 4.146), rel=0.1)  # 10% better than the first: improving
    pl.attach(r, BASELINES)
    assert r["plateau"]["at_reference"] is True
    _, _, claims = _render(r)
    _, structured = agent_result(r)
    assert not any("best known baseline" in c for c in claims)
    assert "stop varying" not in structured["next_action"]


def test_nothing_added_for_keep_or_escalate_or_without_plateau():
    r = result((4.3, 4.146), action="continue")
    pl.attach(r, BASELINES)
    assert verdict(r) == "keep"
    assert "stop varying" not in agent_result(r)[1]["next_action"]
    r = result((4.3, 4.146), levers=[])
    pl.attach(r, BASELINES)
    assert verdict(r) == "escalate"
    assert "stop varying" not in agent_result(r)[1]["next_action"]
    r = result((4.3, 4.146))  # no hook ran: no "plateau" key
    _, body, claims = _render(r)
    assert "plateau" not in body and not any("best known baseline" in c for c in claims)


def test_verdict_unchanged_by_plateau():
    for action, expected in (("switch_strategy", "escape"), ("repair", "repair"), ("continue", "keep")):
        r = result((4.3, 4.146), action=action)
        before = verdict(r)
        pl.attach(r, BASELINES)
        assert before == expected == verdict(r) == agent_result(r)[1]["verdict"]


# ------------------------------------------------------------------ recorded fixtures and the demo
def test_recorded_fixture_verdicts_unchanged():
    for path in sorted((REPO / "fixtures" / "runs").glob("*.json")):
        rec = json.loads(path.read_text())
        esc = rec.get("escape") or {}
        r, recorded = esc.get("result"), esc.get("contract")
        if not r or not recorded:
            continue
        assert render(r).splitlines()[0] == recorded.splitlines()[0]
        base = (rec.get("baselines") or {}).get("dev") or {}
        pl.attach(r, {k: {"score": s, "valid": True} for k, s in base.items()})
        assert render(r).splitlines()[0] == recorded.splitlines()[0]


def test_demo_strategy_loop_runs():
    lines = []
    assert demo.run("strategy_loop", root=REPO, out=lines.append) == 0
    text = "\n".join(lines)
    assert "VERDICT:" in text and "== 7. Results" in text
