"""The offline demo, the preflight check and the escape CLI's contract output. No network, no API."""
import json
import sys
import types
from types import SimpleNamespace

import httpx
import pytest

import diverge as herd
import diverge.pipeline as pipeline
from diverge import demo
from diverge.llm import LLM, Ledger

ATTEMPTS = [
    {"id": "attempt-0", "code": "def priority(item, capacities):\n    return [-(c - item) for c in capacities]\n",
     "dev_score": 4.21, "feedback": "best fit; 4.21% excess\nmore detail", "source": "agent"},
    {"id": "attempt-1", "code": "import numpy as np\ndef priority(item, capacities):\n    caps = np.asarray(capacities)\n"
     "    return -(caps - item)\n", "dev_score": 4.21, "feedback": "no change", "source": "agent"},
    {"id": "attempt-2", "code": "def priority(item, capacities):\n    out = []\n    for c in capacities:\n"
     "        out.append(item - c)\n    return out\n", "dev_score": 4.20, "feedback": None, "source": "agent"},
]

RESULT = {
    "loop_detected": True, "strategy_similarity": 0.91, "code_similarity": 0.28, "action": "switch_strategy",
    "shared_strategy": "tightest remaining capacity", "reasons": ["3 attempts share one strategy",
                                                                  "no progress after 3 attempts"],
    "herd_prevalence": 0.62,
    "overlooked_levers": [{"type": "unused_signal", "name": "history", "mechanism": "Build the rule around history.",
                           "why": "history is used by 38/95 herd samples (40%); none of your 3 attempts use it"}],
    "recommended_switch": {"mechanism": "Build the decision rule around history.",
                           "why": "history is used by 38/95 herd samples (40%); none of your 3 attempts use it"},
    "evaluation": {"before": 4.20, "after": 3.90, "improved": True, "split": "dev"},
}


def write_fixture(root, name="loop", escape_held=3.95, fresh_held=4.10, contract="VERDICT: escape\n{}"):
    (root / "fixtures" / "trajectories").mkdir(parents=True, exist_ok=True)
    (root / "fixtures" / "runs").mkdir(parents=True, exist_ok=True)
    (root / "fixtures" / "trajectories" / f"{name}.json").write_text(json.dumps(
        {"name": name, "task": "online_bin_packing", "how_generated": "three best-fit rewrites", "attempts": ATTEMPTS}))
    run = {"fixture": name, "task": "online_bin_packing", "generated_at": "2026-10-18T00:00:00", "commit": "abc123",
           "preliminary": True, "n_runs": 1, "budget_usd_each": 1.0,
           "baselines": {"dev": {"best_fit": 4.21, "first_fit": 4.88}, "heldout": {"best_fit": 4.30, "first_fit": 4.95}},
           "best_attempt": {"id": "attempt-2", "dev_score": 4.20, "heldout_score": 4.29},
           "escape": {"result": RESULT, "contract": contract, "candidate_code": "def priority(i, c): ...",
                      "dev_score": 3.90, "heldout_score": escape_held, "cost_usd": 0.0412, "calls": 3},
           "fresh_start": {"prompt": "solve it", "candidates": [{"code": "x", "dev_score": 4.0, "valid": True,
                                                                  "error": None}],
                           "candidate_code": "x", "dev_score": 4.0, "heldout_score": fresh_held,
                           "cost_usd": 0.0388, "calls": 3},
           "overlook_excerpt": {"inputs": [{"input": "history", "share": 0.4, "used_by": 38, "decided": 95,
                                            "overlooked": True}],
                                "assumptions": [{"assumption": "bins are scored independently", "kind": "incidental",
                                                 "shared": True, "held_by": 90, "checked": 95}]},
           "paraphrase": {"verdict": "prompt_sensitive", "mean_P": 0.51, "max_R": 0.40}}
    (root / "fixtures" / "runs" / f"{name}.json").write_text(json.dumps(run))


def capture(name, root):
    lines = []
    code = demo.run(name, root=root, out=lines.append)
    return code, "\n".join(lines)


SECTIONS = ["DIVERGE demo: loop", "== 1.", "attempt-0", "== 2. Loop diagnosis", "strategy similarity 0.91 vs code "
            "similarity 0.28", "== 3. Consensus evidence", "history: used by 38/95 consensus samples", "== 4. Scope",
            "prompt_sensitive", "== 5. Escape lever", "why: history is used by", "== 6. Contract",
            "VERDICT: escape", "== 7. Results", "best attempt (attempt-2)", "escape candidate", "fresh_start candidate",
            "best_fit", "first_fit", "PRELIMINARY: 1 run", "escape beat fresh_start on held-out by 0.1500"]


@pytest.fixture
def offline(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network or client used during the demo")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for name in ("get", "post", "request", "stream"):
        monkeypatch.setattr(httpx, name, boom)
    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(LLM, "_post", boom)
    monkeypatch.setattr(pipeline.Herd, "__init__", boom)


def test_demo_prints_every_section_in_order(tmp_path, offline):
    write_fixture(tmp_path)
    code, text = capture("loop", tmp_path)
    assert code == 0
    positions = [text.find(s) for s in SECTIONS]
    assert -1 not in positions, [s for s, p in zip(SECTIONS, positions) if p == -1]
    assert positions == sorted(positions)
    assert "best fit; 4.21% excess" in text and "more detail" not in text  # first feedback line only
    assert "return [-(c - item) for c in capacities]" in text


@pytest.mark.parametrize("escape_held,fresh_held,sentence", [
    (3.95, 4.10, "escape beat fresh_start on held-out by 0.1500"),
    (4.10, 4.10, "escape tied fresh_start on held-out"),
    (4.30, 4.10, "escape lost to fresh_start on held-out by 0.2000"),
    (None, 4.10, "cannot compare on held-out: no held-out score for escape"),
])
def test_demo_comparison_sentence(tmp_path, offline, escape_held, fresh_held, sentence):
    write_fixture(tmp_path, escape_held=escape_held, fresh_held=fresh_held)
    code, text = capture("loop", tmp_path)
    assert code == 0 and text.splitlines()[-1].startswith(sentence)


def test_demo_is_deterministic(tmp_path, offline):
    write_fixture(tmp_path)
    assert capture("loop", tmp_path) == capture("loop", tmp_path)


def test_demo_renders_contract_when_not_recorded(tmp_path, offline, monkeypatch):
    stub = types.ModuleType("diverge.contract")
    stub.render = lambda result, overlook=None, paraphrase=None, families=None: \
        f"VERDICT: stubbed\n{json.dumps({'loop': result['loop_detected']})}"
    monkeypatch.setitem(sys.modules, "diverge.contract", stub)
    monkeypatch.setattr(herd, "contract", stub, raising=False)
    write_fixture(tmp_path, contract=None)
    _, text = capture("loop", tmp_path)
    assert 'VERDICT: stubbed\n{"loop": true}' in text


def test_demo_missing_run_file_explains_and_fails(tmp_path, offline):
    write_fixture(tmp_path)
    (tmp_path / "fixtures" / "runs" / "loop.json").unlink()
    code, text = capture("loop", tmp_path)
    assert code == 1 and "fixtures/runs/loop.json is missing" in text and "preflight" in text


def test_cli_demo_exit_code_for_missing_fixture(monkeypatch, offline, capsys):
    from diverge.__main__ import main
    monkeypatch.setattr(sys, "argv", ["diverge", "demo", "no_such_fixture_xyz"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1 and "no_such_fixture_xyz" in capsys.readouterr().out


CFG = {"models": {"samplers": ["anthropic/claude-sonnet-5.5"], "labeler": "anthropic/claude-opus-5.5",
                  "implementer": "google/gemini-4-argon", "extractor": "not/a-real-model"},
       "paths": {"cache": "cache"}}


@pytest.mark.parametrize("spent,env,expected", [
    (27.5, {"OPENROUTER_API_KEY": "sk-secret-value"}, 1),   # $0.50 left
    (5.0, {"OPENROUTER_API_KEY": "sk-secret-value"}, 0),
    (5.0, {}, 1),                                            # key missing
])
def test_preflight_exit_codes(spent, env, expected):
    lines = []
    fake = SimpleNamespace(ledger=Ledger(28.0, spent), cfg=CFG)
    assert demo.preflight(fake, out=lines.append, live=False, env=env) == expected
    text = "\n".join(lines)
    assert "sk-secret-value" not in text
    assert f"remaining ${28.0 - spent:.4f}" in text
    assert "EXA_API_KEY: NOT set" in text


def test_preflight_tolerates_live_catalog_failure(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "get", boom)
    lines = []
    fake = SimpleNamespace(ledger=Ledger(28.0, 1.0), cfg=CFG)
    assert demo.preflight(fake, out=lines.append, env={"OPENROUTER_API_KEY": "k"}) == 0
    assert any("live catalog: unavailable" in l for l in lines)


def test_escape_cli_prints_contract_after_summary(tmp_path, monkeypatch, capsys):
    import diverge.__main__ as cli
    import diverge.reality
    import diverge.server
    (tmp_path / "overlook.json").write_text(json.dumps({"inputs": []}))
    (tmp_path / "paraphrase.json").write_text(json.dumps({"verdict": "stable", "mean_P": 0.1, "max_R": 0.2}))
    seen = {}

    class FakeHerd:
        def __init__(self, budget_usd=None):
            self.ledger = Ledger(28.0, 1.0)

        def out_dir(self, task):
            return tmp_path

    def render(result, overlook=None, paraphrase=None, families=None):
        seen.update(overlook=overlook, paraphrase=paraphrase, families=families)
        return "VERDICT: escape\n{}"
    stub = types.ModuleType("diverge.contract")
    stub.render = render
    monkeypatch.setitem(sys.modules, "diverge.contract", stub)
    monkeypatch.setattr(herd, "contract", stub, raising=False)
    monkeypatch.setattr(cli, "Herd", FakeHerd)
    monkeypatch.setattr(herd.reality, "Evaluator", lambda *a, **k: None)
    monkeypatch.setattr(herd.server, "escape_for_agent", lambda *a, **k: dict(RESULT))
    candidate = tmp_path / "cand.py"
    candidate.write_text(ATTEMPTS[0]["code"])
    args = SimpleNamespace(task="online_bin_packing", candidate=str(candidate), history=None, evaluator="",
                           higher_is_better=False, budget=1.0)
    cli.escape_cmd(args)
    out = capsys.readouterr().out
    assert out.index("Loop detected: True") < out.index("spent $") < out.index("VERDICT: escape")
    assert seen == {"overlook": {"inputs": []}, "paraphrase": {"verdict": "stable", "mean_P": 0.1, "max_R": 0.2},
                    "families": None}
