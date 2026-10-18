"""HERD as an MCP server for coding agents. Every tool returns compact JSON (herd_escape: a
VERDICT line, then JSON, plus the same object as structuredContent), streams progress
notifications, and is capped by its own budget_usd (and the config's global cap). Failures come
back as structured errors with what is missing and a next_action; herd_status is free."""
import asyncio
import hashlib
import json
import os
import re
import threading
from pathlib import Path

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp_types import CallToolResult, TextContent

from .llm import BudgetExceeded
from .pipeline import (Herd, baseline_scores, check_approach, effective_n, ensure_family_scores, ensure_map,
                       record_heldout_use)
from .reality import Evaluator
from .task import Task

INSTRUCTIONS = """HERD: different code isn't always different thinking. It tells a coding agent when its attempts \
are one strategy in different code, and names one evidence-backed alternative it has scored.
Use it for strategy stagnation on a task with a score (evaluator_cmd, or a built-in task: online_bin_packing, \
cache_eviction). Not for generic debugging, build errors or a first attempt.
- Unsure whether HERD is configured here: herd_status (free, no network).
- Choosing a strategy before coding: herd_check.
- Several scored attempts have stalled, the same failure keeps coming back, lots of code change but little \
score change, or you are about to retry another variation or escalate to a human: call herd_escape yourself; \
do not wait for the user to ask. Follow its VERDICT and next_action.
- Evidence is labeled MEASURED, SOURCE_BACKED or HYPOTHESIZED. Never treat HYPOTHESIZED evidence as measured.
- herd_map, blind_spots and reality_check are lower-level research primitives.
evaluator_cmd takes a code file path as its last argument and prints {"valid": bool, "score": float}; lower is \
better unless lower_is_better=false. Errors come back as JSON with error, missing and next_action. Never invent \
a credential: a missing key needs the user."""

mcp = MCPServer("herd", instructions=INSTRUCTIONS)


def compact(data):
    return json.dumps(data, separators=(",", ":"), default=str)


def failure(error, detail, next_action, missing=(), partial=None, can_continue=False):
    """A structured error an agent can act on (never contains key values)."""
    from .demo import OFFLINE_DEMO, REPO
    out = {"error": error, "detail": detail, "missing": list(missing), "complete": False,
           "live_calls_available": bool(os.environ.get("OPENROUTER_API_KEY")),
           "offline_demo_available": (REPO / "fixtures" / "runs" / f"{OFFLINE_DEMO}.json").exists(),
           "can_continue_autonomously": can_continue, "next_action": next_action}
    if partial:
        out["partial"] = partial
    return out


def classify(e, spent):
    """failure() for an exception raised by a tool's work."""
    from .demo import OFFLINE_DEMO_COMMAND
    msg = str(e)
    partial = {"spent_usd": round(spent, 4), "resumable": True,
               "note": "completed model and search calls are cached; the same call resumes from them"} if spent else None
    if isinstance(e, BudgetExceeded):
        if msg.startswith("global budget"):
            action = ("The global spend cap (budget_usd in config.yaml) is reached. Stop calling HERD's live tools and "
                      "ask the user whether to raise it.")
        else:
            action = ("This call's budget_usd was reached. Completed calls are cached, so the same call with a larger "
                      "budget_usd resumes where it stopped; ask the user before raising spend.")
        return failure("budget_exceeded", msg, action, ["budget"], partial)
    missing_key = re.search(r"\b([A-Z][A-Z0-9_]*_API_KEY) is not set", msg)
    if isinstance(e, RuntimeError) and missing_key:  # herd.llm.require_env, possibly wrapped
        key = missing_key.group(1)
        return failure("missing_requirement", f"{key} is not set", (
            f"Live HERD calls need {key}, which only the user can provide (in .env); ask them, and never invent or "
            f"guess a credential. Meanwhile the offline demo works: {OFFLINE_DEMO_COMMAND} (no API calls). "
            f"herd_status shows what is configured."), [key], partial)
    if isinstance(e, ValueError) and "no built-in evaluator" in msg:
        return failure("evaluator_required", msg, (
            "Pass evaluator_cmd (a command that takes a code file path as its last argument and prints "
            '{"valid": bool, "score": float}) or use a built-in task, then call again.'), ["evaluator_cmd"],
            partial, can_continue=True)
    return failure(type(e).__name__, msg, "Report this error to the user; do not retry the same call unchanged.",
                   partial=partial)


async def run_tool(ctx, budget_usd, work):
    """Run blocking pipeline work in a thread, forwarding its progress messages to the client."""
    loop, lock, step = asyncio.get_running_loop(), threading.Lock(), [0]

    def progress(message):
        with lock:
            step[0] += 1
            asyncio.run_coroutine_threadsafe(ctx.report_progress(step[0], None, message), loop)

    def call():
        herd = None
        try:
            herd = Herd(budget_usd, progress)
            result = work(herd)
        except Exception as e:  # any failure becomes a structured answer the agent can read
            result = classify(e, herd.ledger.spent_this_run if herd else 0.0)
        return {**result, "spent_usd": round(herd.ledger.spent_this_run, 4) if herd else 0.0}

    return compact(await anyio.to_thread.run_sync(call))


def low_confidence(study, cfg):
    """Too few labeled samples for some model to tell an overlooked family from an unlucky one."""
    return effective_n(study) < cfg["reality_check"]["min_samples_per_model"]


def map_summary(study, cfg):
    m = study["herd_map"]
    return {"task": m["task"], "n_samples": m["n_labeled"], "n_unlabeled": m["n_unlabeled"],
            "low_confidence": low_confidence(study, cfg),
            "concentration_top3": m["concentration_top3"], "top3": m["top3"],
            "families": [[r["family"], r["share"], r["count"]] for r in m["families"]],
            "per_model": m["per_model"]}


@mcp.tool()
async def herd_map(task_description: str, ctx: Context, budget_usd: float = 5.0) -> str:
    """Research primitive: which solution families do AI models converge on for this task? Call it
    when you want the landscape itself; to act, prefer herd_check (choosing a strategy) or herd_escape
    (stalled attempts). Not for debugging. Needs OPENROUTER_API_KEY, and EXA_API_KEY when the map is
    not cached yet. Default budget_usd 5.0; about $0 once cached. Cold start: the first call on a task without a cached herd map builds one (samples from every configured model, an Exa literature search, labeling): several minutes and real spend, capped by budget_usd (recorded 100-samples-per-model builds cost about $3); cached after (herd_status lists cached tasks).
    Returns task, n_samples, low_confidence, concentration_top3, top3, families [[family, share,
    count]], per_model, spent_usd."""
    return await run_tool(ctx, budget_usd,
                          lambda herd: map_summary(ensure_map(herd, Task.resolve(task_description)), herd.cfg))


@mcp.tool()
async def blind_spots(task_description: str, ctx: Context, budget_usd: float = 5.0) -> str:
    """Research primitive: solution families from the literature that model samples rarely or
    never used. Call it when you want unexplored, sourced directions to read; to decide whether to
    switch, prefer herd_check, or herd_escape once attempts stall. Needs OPENROUTER_API_KEY, and
    EXA_API_KEY when the map is not cached yet. Default budget_usd 5.0; shares herd_map's cached map
    ($0 after it is built). Cold start: the first call on a task without a cached herd map builds one (samples from every configured model, an Exa literature search, labeling): several minutes and real spend, capped by budget_usd (recorded 100-samples-per-model builds cost about $3); cached after (herd_status lists cached tasks).
    Returns low_confidence (true: too few samples per model to tell an overlooked family from an
    unlucky one) and blind_spots [{family, mechanism, observed (k of n), share_upper_95 (95% upper
    bound on the true share), per_model_counts, source_urls}], spent_usd."""
    def work(herd):
        study = ensure_map(herd, Task.resolve(task_description))
        return {"low_confidence": low_confidence(study, herd.cfg),
                "blind_spots": [{"family": b["family"], "mechanism": b["one_line_mechanism"],
                                 "observed": b["observed"], "share_upper_95": b["share_upper_95"],
                                 "per_model_counts": b["per_model_counts"], "source_urls": b["source_urls"]}
                                for b in study["blind_spots"]]}
    return await run_tool(ctx, budget_usd, work)


@mcp.tool()
async def reality_check(task_description: str, family: str, ctx: Context, evaluator_cmd: str = "",
                        k: int = 3, lower_is_better: bool = True, baseline_files: list[str] | None = None,
                        contrast: bool = False, budget_usd: float = 3.0) -> str:
    """Research primitive: implement one named family (from herd_map or blind_spots) k times
    (default 3) and score each with the evaluator. Call it to test one specific family; prefer
    herd_check, which chooses the families to test, or herd_escape for stalled attempts. Needs an
    evaluator (evaluator_cmd, or a built-in task) and OPENROUTER_API_KEY; an unknown family returns
    error unknown_family with the valid names. Default budget_usd 3.0 (a hard cap, $3): k implementer
    calls plus sandboxed evaluator runs; builds the herd map first if it is not cached (see herd_map). The best
    implementation is picked on dev; final_score is its held-out score on built-in tasks (dev for
    evaluator_cmd; see scored_on). Returns family, kind, share, n_valid, n_invalid, dev_best_score,
    final_score, scored_on, baselines, beats_baselines, best_tested_herd, best_code, invalid_errors,
    spent_usd. contrast=True adds the herd-contrast section (experimental; see README)."""
    def work(herd):
        task = Task.resolve(task_description)
        study = ensure_map(herd, task)
        by_name = {f["family"]: f for f in study["families"]}
        if family not in by_name:
            return {**failure("unknown_family", f"{family!r} is not a family in this task's herd map",
                              "Call again with one of the names in families.", ["family"], can_continue=True),
                    "families": sorted(by_name)}
        evaluator = Evaluator(task, evaluator_cmd or None, lower_is_better)
        scores = ensure_family_scores(herd, task, study, [by_name[family]], evaluator, k, contrast)
        s, out = scores[family], herd.out_dir(task)
        # Only herd families that were already scored under this evaluator; none are tested here.
        herd_scores = {f: v["final_score"] for f, v in scores.items()
                       if v["kind"] == "herd" and v.get("final_score") is not None}
        baselines = baseline_scores(task, evaluator, baseline_files or ())
        result = {"family": family, "kind": s["kind"], "share": s["share"], "k": s["k"],
                  "n_valid": s["n_valid"], "n_invalid": s["n_invalid"],
                  "dev_best_score": s["best_score"], "dev_median_score": s["median_score"],
                  "final_score": s["final_score"], "scored_on": s["final_split"],
                  "invalid_errors": [m["error"] for m in s["implementations"] if not m["valid"]],
                  "baselines": baselines, "compared_herd_families": sorted(herd_scores), "best_tested_herd": None}
        if evaluator.has_heldout and s["final_score"] is not None:
            prior = record_heldout_use(herd, task, evaluator)
            result["heldout_prior_uses"], result["exploratory"] = prior, prior > 0
        if s["final_score"] is not None:
            result["beats_baselines"] = [b for b, v in baselines.items()
                                         if v is not None and evaluator.beats(s["final_score"], v)]
        if s["best_code_path"]:
            result["best_code"] = (out / s["best_code_path"]).read_text()
        if herd_scores:
            top = evaluator.best(herd_scores.values())
            result["best_tested_herd"] = {"family": next(f for f, v in herd_scores.items() if v == top),
                                          "final_score": top}
        return result
    return await run_tool(ctx, budget_usd, work)


@mcp.tool()
async def herd_check(task_description: str, my_approach: str, ctx: Context, evaluator_cmd: str = "",
                     k: int = 3, lower_is_better: bool = True, contrast: bool = False,
                     budget_usd: float = 5.0) -> str:
    """Call before committing to a strategy: which family does my_approach (a plain-language
    description) belong to, how common is it among AI models, and does a tested family beat it?
    Not for debugging a failing solution; once scored attempts have stalled, use herd_escape. Needs
    OPENROUTER_API_KEY (and EXA_API_KEY when the map is not cached yet). With an evaluator
    (evaluator_cmd, or a built-in task) it implements your family, the blind spots and the top herd
    families k times each (default 3) and scores them (held-out on built-in tasks); without one it
    only classifies (verdict "untested"). Default budget_usd 5.0. Cold start: the first call on a task without a cached herd map builds one (samples from every configured model, an Exa literature search, labeling): several minutes and real spend, capped by budget_usd (recorded 100-samples-per-model builds cost about $3); cached after (herd_status lists cached tasks).
    Returns your family and share, and verdict keep | try_blind_spot | try_other_family (also
    inconclusive, unclassified, untested), with blind_spot or better_family, diverse_alternative when
    a blind spot also beat yours, and spent_usd. contrast=True is experimental (see README)."""
    def work(herd):
        task = Task.resolve(task_description)
        study = ensure_map(herd, task)
        evaluator = Evaluator(task, evaluator_cmd or None, lower_is_better) \
            if evaluator_cmd or task.evaluator else None
        return check_approach(herd, task, study, my_approach, evaluator, k, contrast)
    return await run_tool(ctx, budget_usd, work)


@mcp.tool()
async def herd_status(check_model_catalog: bool = False) -> str:
    """Free readiness check: call it first when unsure whether HERD is configured here, or after a
    missing_requirement error. Costs $0: no model, search or network call (check_model_catalog=true
    adds one free, unauthenticated GET of OpenRouter's public model list; optional). Reports whether
    each key is configured, never its value. Returns ready, live_mode_available,
    remaining_budget_usd, requirements {openrouter, exa (optional)}, missing, offline_demo_available
    and offline_demo_command, cached_tasks (built-in tasks with a cached herd map: no cold start),
    recommended_primary_tool, when_to_call, next_action."""
    from .demo import readiness

    def call():
        try:
            return readiness(live=check_model_catalog)
        except Exception as e:  # e.g. an unreadable config: still structured
            return failure(type(e).__name__, str(e), "Report this error to the user.")
    return compact({**await anyio.to_thread.run_sync(call), "spent_usd": 0.0})


def escape_for_agent(herd, task, current_solution, previous_attempts, evaluator, k=1, counterexamples=True):
    """herd.escape.escape plus what the agent needs to trust it: strategy counterexamples between its
    attempts and the escape candidate (tasks with a probe mode), and how stable the herd map is
    under rewording (the paraphrase test), when those exist."""
    from . import counterexample
    from .escape import escape
    from .pipeline import read_json
    from .strategy import llm_extractor
    out = herd.out_dir(task)
    result = escape(herd, task, current_solution, previous_attempts, evaluator,
                    overlook=read_json(out / "overlook.json"), families=read_json(out / "blind_spots.json"),
                    extract=llm_extractor(herd, task), k=k)
    para = read_json(out / "paraphrase.json")
    result["herd_stability"] = ({"verdict": para["verdict"], "paraphrase_tvd": para["mean_P"],
                                 "resampling_tvd_max": para["max_R"]} if para else None)
    if counterexamples and result.get("candidate") and task.evaluator and hasattr(task.evaluator, "probe"):
        codes = [a if isinstance(a, str) else a["code"] for a in [*previous_attempts, current_solution]]
        found = counterexample.find(task, {f"attempt-{i}": c for i, c in enumerate(codes)},
                                    {"escape": result["candidate"]}, n=200, max_results=3)
        result["counterexamples"] = found["counterexamples"]
        result["counterexample_summary"] = found["summary"]
    return result


@mcp.tool(structured_output=False)
async def herd_escape(task_description: str, current_solution: str, ctx: Context,
                      previous_attempts: list[str] | None = None, evaluator_cmd: str = "",
                      lower_is_better: bool = True, k: int = 1, budget_usd: float = 2.0) -> CallToolResult:
    """Primary tool. Call it yourself, without waiting to be asked, when several scored attempts
    have stalled, the same failure keeps coming back, code changes a lot but the score barely moves,
    or you are about to retry another variation or hand the problem to a human. Not for a first
    attempt, syntax errors or generic debugging. Pass current_solution and previous_attempts as code
    (oldest first). HERD scores them on the dev split, fingerprints their strategies, says whether
    they are one strategy in different code and, when they loop, implements one evidence-backed lever
    and scores it before/after (with counterexamples on tasks that support probes).
    Needs an evaluator (evaluator_cmd, or a built-in task), OPENROUTER_API_KEY, and at least 2 valid
    scored attempts for any verdict but inconclusive. It does not build a herd map: levers come from
    the task's cached overlook study and blind spots (built-in tasks); without them a loop may end in
    escalate. Default budget_usd 2.0, k=1 candidate; recorded runs that generated no candidate spent
    under $0.01, generating one costs more.
    Returns "VERDICT: keep|repair|escape|escalate|inconclusive", then JSON: next_action,
    can_continue_autonomously, complete, missing_requirements, candidate_available,
    evidence_summary, lever, next_experiment (success and failure thresholds), evidence (each
    MEASURED, SOURCE_BACKED or HYPOTHESIZED), candidate_path, spent_usd; the same object (plus
    verdict) is the structuredContent. Follow next_action; never treat HYPOTHESIZED as measured."""
    from .contract import agent_result
    from .pipeline import read_json
    loaded = {}

    def work(herd):
        task = Task.resolve(task_description)
        if not evaluator_cmd and not task.evaluator:
            return failure("evaluator_required", "pass evaluator_cmd or use a registered task", (
                "Pass evaluator_cmd (a command that takes a code file path as its last argument and prints "
                '{"valid": bool, "score": float}) or use a built-in task, then call herd_escape again.'),
                ["evaluator_cmd"], can_continue=True)
        evaluator = RecordingEvaluator(Evaluator(task, evaluator_cmd or None, lower_is_better))
        codes = [*(previous_attempts or []), current_solution]
        result = escape_for_agent(herd, task, current_solution, previous_attempts or [], evaluator, k)
        result["attempts"] = evaluator.attempts(codes)
        result["lower_is_better"] = evaluator.lower_is_better
        out = herd.out_dir(task)
        loaded.update(overlook=read_json(out / "overlook.json"), paraphrase=read_json(out / "paraphrase.json"),
                      families=[*(read_json(out / "blind_spots.json") or []), *(read_json(out / "families.json") or [])])
        return result
    # Text first line stays "VERDICT: <verdict>" (contract.parse reads it); the same object is also
    # structuredContent. structured_output=False: no outputSchema, so the dict is passed through as is.
    text, structured = agent_result(json.loads(await run_tool(ctx, budget_usd, work)), **loaded)
    return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=structured)


class RecordingEvaluator:
    """An Evaluator that remembers each dev result by file name, so herd_escape can report the
    attempts' own scores (escape() scores them at escape/attempts/<sha256(code)[:12]>.py)."""

    def __init__(self, evaluator):
        self._evaluator, self._seen = evaluator, {}

    def __getattr__(self, name):
        return getattr(self._evaluator, name)

    def __call__(self, path, *args, **kwargs):
        result = self._evaluator(path, *args, **kwargs)
        self._seen[Path(path).name] = result
        return result

    def attempts(self, codes):
        out = []
        for i, code in enumerate(codes):
            r = self._seen.get(f"{hashlib.sha256(code.encode()).hexdigest()[:12]}.py") or {}
            out.append({"id": f"attempt-{i}", "score": r.get("score"), "valid": r.get("valid")})
        return out


def serve():
    mcp.run("stdio")
