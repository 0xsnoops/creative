"""Opt-in admission feedback for escape v2, with offline models and admission."""
import sys

import pytest

from diverge.escape import ADMISSION_FEEDBACK, IMPLEMENT_MARKER, escape_v2, implementation_prompt_v2
from diverge.reality import Evaluator
from tests.test_escape_v2 import ATTEMPTS, CURRENT, overlook, world  # noqa: F401
from tests.test_pipeline_offline import fake_world  # noqa: F401


@pytest.fixture
def implementation_calls(world, monkeypatch):
    h, _, _ = world
    calls = []
    chat = h.llm.chat

    def record(model, messages, **kwargs):
        if IMPLEMENT_MARKER in messages[0]["content"]:
            calls.append((messages[0]["content"], kwargs["sample_idx"]))
        return chat(model, messages, **kwargs)

    monkeypatch.setattr(h.llm, "chat", record)
    return calls


@pytest.mark.parametrize("rejections", [
    [],
    [["agrees with the stuck attempts", "does not depend on history"]],
    [["agrees with the stuck attempts"], ["still insensitive to history"]],
    [[]],
])
def test_feedback_uses_latest_admission_reasons(world, monkeypatch, implementation_calls, rejections):
    h, task, state = world
    state["reject"] = len(rejections)
    admission = sys.modules["diverge.admission"]
    admit = admission.admit

    def with_reasons(*args, **kwargs):
        verdict = admit(*args, **kwargs)
        if not verdict["admitted"]:
            verdict["reasons"] = rejections[len(state["admit"]) - 1]
        return verdict

    monkeypatch.setattr(admission, "admit", with_reasons)
    context = overlook(["s-000"])
    result = escape_v2(h, task, CURRENT, ATTEMPTS, Evaluator(task), overlook=context,
                       feedback_on_reject=True)
    original = implementation_prompt_v2(task, result["kept_sketches"][0], result["lever"], context)
    expected = [original] + [original + ADMISSION_FEEDBACK.format(
        reasons="\n".join(f"- {reason}" for reason in (reasons or ["not admitted"])), lever="history")
        for reasons in rejections]

    assert result["implementation_prompts"] == expected
    assert [p for p in state["prompts"] if IMPLEMENT_MARKER in p] == expected
    assert implementation_calls == [(prompt, f"escape_v2:impl:0:0:{i}") for i, prompt in enumerate(expected)]
    assert result["admitted"][0]["attempt"] == len(rejections)
    assert result["candidate"]


@pytest.mark.parametrize("options", [{}, {"feedback_on_reject": False}])
def test_default_preserves_blind_resampling(world, implementation_calls, options):
    h, task, state = world
    state["reject"] = 1
    context = overlook(["s-000"])
    result = escape_v2(h, task, CURRENT, ATTEMPTS, Evaluator(task), overlook=context, **options)
    original = implementation_prompt_v2(task, result["kept_sketches"][0], result["lever"], context)

    assert [p for p in state["prompts"] if IMPLEMENT_MARKER in p] == [original, original]
    assert implementation_calls == [(original, f"escape_v2:impl:0:0:{i}") for i in range(2)]
    assert result["implementation_prompts"] == [original]  # Preserve v2's recorded behavior.
    assert result["admitted"][0]["attempt"] == 1
    assert result["candidate"]
