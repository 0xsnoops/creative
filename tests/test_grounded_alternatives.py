"""The agent's own inputs, offline: already_tried, constraints and context shape free-text alternatives
(screened by a separate model, grounding-checked, ranked by test cost), depth="deep" samples several
models with a critic, diverge_report_outcome logs results that later calls treat as already tried, and
diverge_escape puts the same inputs into its implementation prompt."""
import json
from pathlib import Path

import anyio
from mcp import Client

import diverge.pipeline as pipeline
from diverge import alternatives as A
from diverge.contract import parse
from diverge.llm import LLM
from diverge.task import Task
from tests.test_escape_tool import BEST_FITS, v2_reply, call_escape
from tests.test_pipeline_offline import FUNSEARCH, MODELS, fake_world, reply_to  # noqa: F401

FREE = "Write a function that packs items into bins of capacity 100."
SECRET = "sk-or-v1-" + "0123456789abcdef" * 2


def alt(name, cost="hours", decisive=False, **extra):
    return {"name": name, "mechanism": f"{name} decides by the 22 ms gap", "strength": f"strength {name}",
            "trap": f"trap {name}", "how_to_test": f"measure {name}", "builds_on": "", "grounding": "",
            "test_cost": cost, "decisive": decisive, "kill_result": f"{name} loses", **extra}


ALTS = [alt("Slow Idea", "days"), alt("Retry First Fit", "minutes", True), alt("GPU Packing", "minutes", True),
        alt("Cheap Kill", "minutes", True, grounding="relay delay 22-61 ms, p99 99 ms"),
        alt("Middle", "hours", True)]


def install(monkeypatch, alts=ALTS, screen=None, critic=None):
    """Fake models routed by prompt; returns the list of (model, temperature, prompt) asked."""
    asked = []

    def post(self, body, attempts=4):
        prompt, model = body["messages"][0]["content"], body["model"]
        asked.append((model, body["temperature"], prompt))
        if prompt.startswith("You are proposing alternative"):
            names = alts(model) if callable(alts) else alts
            text = json.dumps({"alternatives": names, "relax_hint": "Allowing a GPU would open GPU Packing."})
        elif prompt.startswith(A.SCREEN_MARKER):
            if screen == "broken":
                text = "not json"
            else:
                text = json.dumps({"screen": screen if screen is not None else [
                    {"index": 1, "duplicate_of": 0, "violates": None, "reason": "first fit again"},
                    {"index": 2, "duplicate_of": None, "violates": 0, "reason": "needs a GPU"}]})
        elif prompt.startswith(A.CRITIC_MARKER):
            text = json.dumps({"scores": critic or []})
        else:
            text = reply_to(prompt, body["temperature"])
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001, "completion_tokens": 10}}
    monkeypatch.setattr(LLM, "_post", post)
    return asked


def call(name, args):
    from diverge.server import mcp

    async def go():
        async with Client(mcp) as client:
            r = await client.call_tool(name, args)
            return r.content[0].text, r.structured_content
    return anyio.run(go)


def check(**args):
    return call("diverge_check", {"task_description": FREE, "my_approach": "first fit", **args})


TRIED = [{"approach": "first fit with sorted bins", "outcome": "stuck at 4.1% waste", "score": 0.041}]
RULES = ["CPU only, no GPU", "no new dependencies"]


# ------------------------------------------------------------------ default is unchanged
def test_without_new_inputs_the_free_text_call_is_unchanged(fake_world, monkeypatch):
    asked = install(monkeypatch)
    text, s = check()
    assert len(asked) == 1 and "Already tried" not in asked[0][2] and "test_cost" not in asked[0][2]
    assert "filtered_out" not in s and "agent_inputs" not in s and "relax_hint" not in s
    assert all(set(a) == {"name", "mechanism", "strength", "trap", "how_to_test", "evidence_class"}
               for a in s["alternatives"])
    assert [a["name"] for a in s["alternatives"]] == [a["name"] for a in ALTS]  # model order kept


# ------------------------------------------------------------------ already_tried and constraints
def test_duplicates_and_violators_are_filtered_out_by_a_separate_model(fake_world, monkeypatch):
    asked = install(monkeypatch)
    text, s = check(already_tried=TRIED, constraints=RULES)
    gen = next(p for p in asked if p[2].startswith("You are proposing alternative"))
    scr = next(p for p in asked if p[2].startswith(A.SCREEN_MARKER))
    assert gen[0] != scr[0]  # the reviewer did not write the alternatives
    assert "first fit with sorted bins: stuck at 4.1% waste (score 0.041)" in gen[2]
    assert "- CPU only, no GPU" in gen[2] and "Hard rules" in gen[2]
    names = [a["name"] for a in s["alternatives"]]
    assert "Retry First Fit" not in names and "GPU Packing" not in names and len(names) == 3
    assert s["filtered_out"] == [
        {"name": "Retry First Fit", "reason": "duplicate_of", "of": "first fit with sorted bins", "why": "first fit again"},
        {"name": "GPU Packing", "reason": "violates", "of": "CPU only, no GPU", "why": "needs a GPU"}]
    assert s["relax_hint"].startswith("Allowing a GPU")
    assert s["agent_inputs"] == {"already_tried": 1, "logged_outcomes": 0, "constraints": 2, "context_chars": 0,
                                 "context_truncated": False, "depth": "quick"}
    assert all(a["evidence_class"] == "HYPOTHESIZED" for a in s["alternatives"])
    assert "**Filtered out:** Retry First Fit (duplicate_of" in text and "**Relax hint:**" in text
    assert parse(text) == ("untested", s)


def test_plain_string_tried_and_a_broken_screen_keep_everything(fake_world, monkeypatch):
    install(monkeypatch, screen="broken")
    _, s = check(already_tried=["best fit"])
    assert len(s["alternatives"]) == 5 and s["filtered_out"] == [] and "screen_error" in s
    assert s["verdict"] == "untested" and s["complete"] is True


# ------------------------------------------------------------------ ranking
def test_alternatives_are_ranked_decisive_and_cheapest_first(fake_world, monkeypatch):
    install(monkeypatch, screen=[])
    text, s = check(constraints=["CPU only"])
    assert [a["name"] for a in s["alternatives"]] == ["Retry First Fit", "GPU Packing", "Cheap Kill", "Middle",
                                                      "Slow Idea"]
    first = s["alternatives"][0]
    assert first["test_cost"] == "minutes" and first["decisive"] is True
    assert s["next_action"].startswith("Test Retry First Fit first (minutes to test, decisive)")
    assert "Retry First Fit loses" in s["next_action"] and "diverge_report_outcome" in s["next_action"]
    assert "HYPOTHESIZED" in s["next_action"]
    assert "Cost: minutes, decisive. Wrong if: Retry First Fit loses" in text


def test_rank_and_cost_normalization_units():
    a = [A.normalize({"name": "x", "test_cost": "WEEKS", "decisive": "true"}, True),
         A.normalize({"name": "y", "test_cost": "Hours", "decisive": False}, True),
         A.normalize({"name": "z", "test_cost": "minutes", "decisive": "no"}, True)]
    assert [x["test_cost"] for x in a] == ["unknown", "hours", "minutes"] and a[0]["decisive"] is True
    assert [x["name"] for x in A.rank(a)] == ["x", "z", "y"]


# ------------------------------------------------------------------ context grounding
def test_context_is_redacted_capped_and_numbers_are_checked(fake_world, monkeypatch):
    asked = install(monkeypatch, screen=[])
    context = f"relay delay 22-61 ms over 3 runs\nkey={SECRET}\n" + "x" * 9000
    text, s = check(context=context)
    gen = next(p for p in asked if p[2].startswith("You are proposing alternative"))
    assert SECRET not in gen[2] and SECRET not in text and "[REDACTED]" in gen[2]
    assert "truncated by DIVERGE to 8000 characters" in gen[2]
    assert s["agent_inputs"]["context_truncated"] is True
    cheap = next(a for a in s["alternatives"] if a["name"] == "Cheap Kill")
    assert cheap["grounding"] == "relay delay 22-61 ms, p99 99 ms"
    assert cheap["ungrounded_numbers"] == [99.0]  # 22 and 61 are in the context; 99 is not
    slow = next(a for a in s["alternatives"] if a["name"] == "Slow Idea")
    assert slow["ungrounded_numbers"] == []
    assert "Numbers not in your context: 99.0" in text


def test_ungrounded_numbers_units():
    ctx = "p50 1,200 ms; hit rate 0.37; 3 replicas"
    a = {"mechanism": "use the 1200 ms p50 and 2 queues", "strength": "beats 0.37 by 0.05", "grounding": "45%"}
    assert A.ungrounded_numbers(a, ctx) == [0.05, 45.0]  # single-digit counts (2) are ignored
    assert A.ungrounded_numbers({"mechanism": "no numbers"}, ctx) == []


# ------------------------------------------------------------------ deep mode
def test_deep_mode_samples_several_models_and_a_critic_drops_repeats(fake_world, monkeypatch):
    by_model = {MODELS[0]: [alt("A1", "minutes", True), alt("A2")], MODELS[1]: [alt("B1"), alt("A1 renamed")]}
    critic = [{"index": i, "novelty": n, "constraint_fit": 8, "testability": 6, "same_as": same}
              for i, n, same in [(0, 5, None), (1, 2, None), (2, 9, None), (3, 7, 0)]]
    asked = install(monkeypatch, alts=lambda m: by_model[m], critic=critic)
    text, s = check(depth="deep")
    gens = [p for p in asked if p[2].startswith("You are proposing alternative")]
    assert {m for m, _, _ in gens} == set(MODELS) and len({t for _, t, _ in gens}) == len(gens)
    assert sum(p[2].startswith(A.CRITIC_MARKER) for p in asked) == 1
    deep = s["deep"]
    assert deep["models_used"] == MODELS and deep["n_proposed"] == 4 and deep["n_same_mechanism_dropped"] == 1
    names = [a["name"] for a in s["alternatives"]]
    assert "A1 renamed" not in names and set(names) == {"A1", "A2", "B1"}
    assert names[0] == "A1"  # decisive and minutes ranks first even though B1 has the best critic score
    b1 = next(a for a in s["alternatives"] if a["name"] == "B1")
    assert b1["critic"] == {"novelty": 9.0, "constraint_fit": 8.0, "testability": 6.0} and b1["author"] == MODELS[1]
    assert s["agent_inputs"]["depth"] == "deep" and s["spent_usd"] > 0


def test_unknown_depth_is_a_structured_error(fake_world, monkeypatch):
    install(monkeypatch)
    text, s = check(depth="ultra")
    assert text.startswith("VERDICT: inconclusive") and s["error"] == "ValueError" and "depth" in s["detail"]


# ------------------------------------------------------------------ outcome log
def test_reported_outcomes_are_free_and_become_already_tried(fake_world, monkeypatch):
    asked = install(monkeypatch, screen=[])
    bad = json.loads(call("diverge_report_outcome", {"task_description": FREE, "alternative": "x",
                                                     "outcome": "maybe"})[0])
    assert bad["error"] == "invalid_argument" and bad["missing"] == ["outcome"]
    out = json.loads(call("diverge_report_outcome", {
        "task_description": FREE, "alternative": "Harmonic classes", "outcome": "wrong",
        "note": f"lost by 3% {SECRET}", "score": 0.07})[0])
    assert out["recorded"]["outcome"] == "WRONG" and out["n_outcomes"] == 1 and out["spent_usd"] == 0
    assert asked == []  # no model call
    path = pipeline.ROOT / "results" / Task.resolve(FREE).name / "outcomes.jsonl"
    assert SECRET not in path.read_text() and "[REDACTED]" in path.read_text()
    _, s = check()  # no new inputs, but a logged outcome switches on the grounded path
    gen = next(p for p in asked if p[2].startswith("You are proposing alternative"))
    assert "Harmonic classes: WRONG: lost by 3% [REDACTED] (score 0.07)" in gen[2]
    assert s["agent_inputs"]["logged_outcomes"] == 1 and s["agent_inputs"]["already_tried"] == 1


# ------------------------------------------------------------------ diverge_escape
def test_escape_puts_the_agents_inputs_in_the_implementation_prompt(fake_world, monkeypatch):
    pipeline.ensure_map(pipeline.Session(progress=lambda m: None), Task.resolve("online_bin_packing"))
    prompts = []

    def post(self, body, attempts=4):
        prompt = body["messages"][0]["content"]
        prompts.append(prompt)
        text = v2_reply(prompt)
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"cost": 0.001}}
    monkeypatch.setattr(LLM, "_post", post)
    verdict, body = call_escape({"task_description": "online_bin_packing", "current_solution": BEST_FITS[2],
                                 "previous_attempts": BEST_FITS[:2], "already_tried": TRIED,
                                 "constraints": ["pure Python"], "context": f"waste 4.1% {SECRET}"})
    assert verdict == "escape"
    impl = next(p for p in prompts if "Propose mechanism sketches" in p)  # v2: notes join the failure summary
    assert "Notes from the agent:" in impl and "first fit with sorted bins: stuck at 4.1% waste" in impl
    assert "- pure Python" in impl and "waste 4.1%" in impl and SECRET not in impl
    assert body["agent_inputs"] == {"already_tried": 1, "logged_outcomes": 0, "constraints": 1,
                                    "context_chars": len("waste 4.1% [REDACTED]"), "context_truncated": False}


def test_escape_prompt_is_unchanged_without_notes():
    from diverge.escape import escape_prompt
    task = Task.resolve("online_bin_packing")
    choice = {"lever": {"type": "unused_signal", "name": "n_open_bins"}, "mechanism": "m", "why": "w", "evidence": {}}
    diagnosis = {"shared_cause": "best fit", "failure": "waste"}
    plain = escape_prompt(task, choice, diagnosis, "code")
    assert "Evidence: w\n\nThe current solution" in plain and "Notes from the agent" not in plain
    noted = escape_prompt(task, choice, diagnosis, "code", "Notes from the agent:\nx")
    assert noted == plain.replace("Evidence: w\n\n", "Evidence: w\n\nNotes from the agent:\nx\n\n")


# ------------------------------------------------------------------ the preregistered measurement script
def test_grounded_eval_script_runs_offline(fake_world, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("grounded_eval", Path(__file__).parent.parent
                                                  / "scripts" / "grounded_eval.py")
    ge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ge)
    monkeypatch.setattr(ge, "JUDGE", MODELS[1])
    monkeypatch.setattr(ge, "ROOT", pipeline.ROOT)
    install(monkeypatch, screen=[], critic=[])
    orig = LLM._post

    def post(self, body, attempts=4):
        if body["messages"][0]["content"].startswith("You are judging proposed strategies"):
            labels = [{"index": i, "duplicate": i == 0, "violates": False} for i in range(5)]
            return {"choices": [{"message": {"content": json.dumps({"labels": labels})}, "finish_reason": "stop"}],
                    "usage": {"cost": 0.001}}
        return orig(self, body, attempts)
    monkeypatch.setattr(LLM, "_post", post)
    monkeypatch.setattr(ge, "SCENARIOS", ge.SCENARIOS[2:3])
    ge.main()
    summary = json.loads((pipeline.ROOT / "results" / "grounded_eval" / "summary.json").read_text())
    assert summary["errors"] == [] and set(summary["summary"]) == set(ge.ARMS)
    before, quick = summary["summary"]["before"], summary["summary"]["quick"]
    assert before["n_alternatives"] == 5 and before["dup_rate"] == 0.2
    assert quick["grounded_share"] is not None and before["spent_usd"] > 0
