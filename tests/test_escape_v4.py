"""Escape v4 offline: fake models keyed by prompt markers, the v2 stubbed trajectory analysis and
admission gate, and (except in the end-to-end test) a fake dev evaluator whose score is read from
a marker comment in the candidate code, so parent/child rankings are controlled."""
import json
import re

import pytest

from diverge.escape import (ANCHOR_MARKER, CRITIC4_MARKER, DEEPEN_MARKER, DEFAULT_MODE, EVOLVE_MARKER,
                            IMPLEMENT4_MARKER, MODES, REPAIR_MARKER, SKETCH3_MARKER, escape_v4, run_escape,
                            syntax_check)
from diverge.llm import LLM
from diverge.reality import Evaluator
from tests.test_escape_v2 import ATTEMPTS, CURRENT, code_lines, overlook
from tests.test_escape_v2 import world as v2_world  # noqa: F401
from tests.test_pipeline_offline import FUNSEARCH, MODELS, fake_world  # noqa: F401
from tests.test_pipeline_offline import reply_to

LEVER_RE = re.compile(r"Build your mechanism on this one starting point: (\S+)")
NAME_RE = re.compile(r"^Name: (\S+)", re.M)
SCORE_RE = re.compile(r"# SCORE=([0-9.]+)")
BROKEN = "def priority(item, capacities)\n    return capacities\n"


def code(score):
    return f"{FUNSEARCH}# SCORE={score}\n"


def fence(text):
    return f"```python\n{text}```"


class FakeEvaluator:
    """Dev-only evaluator stand-in: score from '# SCORE=x' (attempts without it score 10)."""
    command, lower_is_better = None, True

    def __init__(self):
        self.paths = []

    def __call__(self, path):
        self.paths.append(str(path))
        m = SCORE_RE.search(open(path).read())
        score = float(m.group(1)) if m else 10.0
        rows = [{"seed": s, "excess_pct": score + 0.1 * s} for s in range(4)]
        return {"valid": True, "score": score, "detail": {"split": "dev", "instances": rows}}

    def best(self, scores):
        return min(scores)

    def beats(self, a, b):
        return a < b


def rubric(n):
    """0: lever unused; 1: feasibility 3; 2: constraint_fit 4; 3: fatal trap -> all dropped.
    Kept, ranked by distinctness + feasibility + constraint_fit: 5 (27) > 4 (24, non-fatal trap) > 6 (15)."""
    rows = {0: dict(uses_target_mechanism=False), 1: dict(feasibility=3), 2: dict(constraint_fit=4),
            3: dict(trap="breaks the capacity rule", fatal_trap=True),
            4: dict(strategy_distinctness=8, feasibility=8, constraint_fit=8, trap="slow on long inputs"),
            5: dict(strategy_distinctness=9, feasibility=9, constraint_fit=9)}
    out = []
    for i in range(n):
        r = {"index": i, "uses_target_mechanism": True, "strategy_distinctness": 5, "feasibility": 5,
             "constraint_fit": 5, "trap": None, "fatal_trap": False, "strength": f"strength {i}"}
        out.append({**r, **rows.get(i, {})})
    return {"scores": out}


@pytest.fixture
def world(v2_world, monkeypatch):
    h, task, state = v2_world
    state.update(prompts=[], models=[], sketch_n=0, critic=rubric, impl={}, repair=[], evolve={},
                 raise_deepen=set())

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        state["prompts"].append(prompt)
        state["models"].append((body["model"], prompt))
        if prompt.startswith(ANCHOR_MARKER):
            text = json.dumps({"summary": "Attempts score about the same. STRIPPED", "constraints": []})
        elif prompt.startswith(REPAIR_MARKER):
            text = fence(state["repair"].pop(0) if state["repair"] else code(4.0))
        elif prompt.startswith(CRITIC4_MARKER):
            text = json.dumps(state["critic"](len(re.findall(r"^\[\d+\] ", prompt, re.M))))
        elif EVOLVE_MARKER in prompt:
            parent = float(re.search(r"Its dev score: ([0-9.]+)", prompt).group(1))
            child = state["evolve"].get(parent, parent + 1.0)
            if isinstance(child, Exception):
                raise child
            text = fence(child if isinstance(child, str) else code(child))
        elif IMPLEMENT4_MARKER in prompt:
            reply = state["impl"].get(NAME_RE.search(prompt).group(1).split("-")[0], 4.0)
            if isinstance(reply, list):
                reply = reply.pop(0) if len(reply) > 1 else reply[0]
            text = fence(reply if isinstance(reply, str) else code(reply))
        elif DEEPEN_MARKER in prompt:
            name = NAME_RE.search(prompt).group(1)
            if name.split("-")[0] in state["raise_deepen"]:
                raise RuntimeError("deepen branch exploded")
            text = json.dumps({"mechanism": f"mechanism of {name}", "main_risk": "risk", "first_step": "step"})
        elif SKETCH3_MARKER in prompt:
            lever = LEVER_RE.search(prompt).group(1)
            n = state["sketch_n"]
            state["sketch_n"] += 1
            text = json.dumps({"name": f"Sketch{n}-{lever}", "state": f"state-token-{n}",
                               "decision_rule": f"rule-token-{n} using {lever}", "uses_lever": f"via {lever}"})
        else:
            text = reply_to(prompt, body["temperature"])
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001, "completion_tokens": 10}}
    monkeypatch.setattr(LLM, "_post", post)
    h.cfg["models"]["labeler"] = MODELS[1]
    return h, task, state


def run(h, task, evaluator=None, **kw):
    kw.setdefault("n_sketches", 7)
    return escape_v4(h, task, CURRENT, ATTEMPTS, evaluator or FakeEvaluator(), overlook=overlook(["s-000"]), **kw)


def of(state, marker):
    return [p for p in state["prompts"] if marker in p]


def test_modes_registered_default_unchanged():
    assert MODES["v4"] is escape_v4 and DEFAULT_MODE == "v2"


def test_syntax_check():
    assert syntax_check(code(1.0), "priority") is None
    assert "SyntaxError" in syntax_check(BROKEN, "priority")
    assert "not defined at the top level" in syntax_check("class X:\n    def priority(self): pass\n", "priority")


def test_critic_rubric_hard_filters_and_ranking(world):
    h, task, state = world
    result = run(h, task)
    assert result["critic"]["model"] == MODELS[1] != h.cfg["models"]["implementer"]
    assert {m for m, p in state["models"] if p.startswith(CRITIC4_MARKER)} == {MODELS[1]}
    assert len(of(state, CRITIC4_MARKER)) == 1 and len(result["critic.rubric"]) == 7
    dropped = {r["index"]: r["reasons"][0] for r in result["rejections"] if r["stage"] == "critic"}
    assert set(dropped) == {0, 1, 2, 3}
    assert "target mechanism" in dropped[0] and "feasibility" in dropped[1]
    assert "constraint_fit" in dropped[2] and "fatal trap" in dropped[3]
    assert [s["index"] for s in result["kept_sketches"]] == [5, 4, 6]
    assert [d["index"] for d in result["deepened"]] == [5, 4]
    deepen4 = [p for p in of(state, DEEPEN_MARKER) if "Name: Sketch4-" in p]
    assert deepen4 and "slow on long inputs" in deepen4[0]  # non-fatal trap passed on as a risk


def test_implementation_prompt_hybrid_siblings_no_attempt_code(world):
    h, task, state = world
    result = run(h, task)
    prompts = of(state, IMPLEMENT4_MARKER)
    assert len(prompts) == 2
    names = {s["index"]: s["name"] for s in result["kept_sketches"]}
    for p in prompts:
        assert "load-bearing" in p and "must not replace" in p and "Do not fall back" in p
        assert not [line for line in code_lines() if line in p]
        assert "held-out" not in p.lower()
    top = next(p for p in prompts if f"Name: {names[5]}" in p)
    siblings = top.split("sibling designs:")[1].split("Helpers like these")[0]
    assert f"- {names[4]}: " in siblings and f"- {names[6]}: " in siblings and names[5] not in siblings
    assert "rule-token-" in siblings


def test_broken_reply_gets_exactly_one_repair_and_proceeds(world):
    h, task, state = world
    state["impl"]["Sketch5"] = [BROKEN, code(2.0)]
    state["repair"] = [code(3.0)]
    result = run(h, task, n_deepen=1)
    assert len(of(state, REPAIR_MARKER)) == 1 and result["calls"]["repair"] == 1
    rp = of(state, REPAIR_MARKER)[0]
    assert "SyntaxError" in rp and "def priority(item, capacities)" in rp and "Do not redesign" in rp
    assert result["repairs"][0]["fixed"] is True
    adm = result["admitted"][0]
    assert adm["repaired"] is True and adm["score"] == 3.0 and adm["attempt"] == 0


def test_still_broken_after_repair_is_rejected_without_second_repair(world):
    h, task, state = world
    state["impl"]["Sketch5"] = BROKEN
    state["repair"] = [BROKEN]
    result = run(h, task, n_deepen=1, max_resample=0)
    assert len(of(state, REPAIR_MARKER)) == 1 and result["calls"]["repair"] == 1
    assert [r["stage"] for r in result["rejections"] if r["stage"] == "syntax"] == ["syntax"]
    assert result["repairs"] == [{**result["repairs"][0], "fixed": False}] and "after_repair" in result["repairs"][0]
    assert not result["admitted"] and result["candidate"] is None and not state["admit"]
    assert not result["evolve"]["parents"]


def test_evolve_once_top_two_parents_dev_only(world):
    h, task, state = world
    state["impl"] = {"Sketch5": 3.0, "Sketch4": 2.0, "Sketch6": 5.0}
    result = run(h, task, n_deepen=3)
    assert sorted(a["score"] for a in result["admitted"]) == [2.0, 3.0, 5.0]
    assert [p["score"] for p in result["evolve"]["parents"]] == [2.0, 3.0]
    prompts = of(state, EVOLVE_MARKER)
    assert len(prompts) == 2 and len(result["evolve"]["children"]) <= 2
    for p, score in zip(prompts, (2.0, 3.0)):
        assert f"Its dev score: {score:g}" in p and "seed:3" in p and "seed:0" in p
        assert "Its worst dev cases: seed:3, seed:2, seed:1" in p
        assert "held-out" not in p.lower() and "test split" not in p.lower()
        assert "may not revert" in p and f"SCORE={score}" in p
    assert {c["parent"] for c in result["evolve"]["children"]} == {p["path"] for p in result["evolve"]["parents"]}
    assert result["calls"]["implementer"] == 7 + 3 + 3 + 2


@pytest.mark.parametrize("child_scores, best, source", [({2.0: 1.0, 3.0: 6.0}, 1.0, "child"),
                                                        ({2.0: 7.0, 3.0: 8.0}, 2.0, "parent")])
def test_final_pick_is_best_on_dev_across_parents_and_children(world, child_scores, best, source):
    h, task, state = world
    state["impl"] = {"Sketch5": 3.0, "Sketch4": 2.0}
    state["evolve"] = child_scores
    result = run(h, task)
    assert result["evaluation"]["after"] == best and result["evaluation"]["from"] == source
    assert f"SCORE={best}" in result["candidate"] and result["evaluation"]["split"] == "dev"
    assert all("code" not in c and "dev_detail" not in c for c in result["admitted"] + result["evolve"]["children"])


def test_branch_failures_do_not_kill_the_run(world):
    h, task, state = world
    state["raise_deepen"] = {"Sketch5"}
    state["impl"] = {"Sketch4": 2.0, "Sketch6": 3.0}
    state["evolve"] = {2.0: RuntimeError("evolve branch exploded"), 3.0: 1.5}
    result = run(h, task, n_deepen=3)
    stages = [r["stage"] for r in result["rejections"]]
    assert stages.count("deepen") == 1 and stages.count("evolve") == 1
    assert len(result["evolve"]["children"]) == 1
    assert result["evaluation"]["after"] == 1.5 and result["candidate"]


def test_end_to_end_real_evaluator_via_run_escape(world):
    h, task, state = world
    result = run_escape("v4", h, task, CURRENT, ATTEMPTS, Evaluator(task), overlook=overlook(["s-000"]),
                        n_sketches=7, n_deepen=1, k=7)
    ev = result["evaluation"]
    assert result["version"] == 4 and ev["split"] == "dev" and ev["after"] is not None
    assert "max_bin_cap" in result["candidate"] and ev["improved"]
    assert len(result["evolve"]["parents"]) == 1 and "seed:" in of(state, EVOLVE_MARKER)[0]
    assert result["repairs"] == [] and result["spent_usd"] >= 0
