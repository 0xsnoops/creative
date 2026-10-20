"""DIVERGE as an MCP server for coding agents. Every tool returns compact JSON (diverge_escape and
diverge_check: a VERDICT line, a short markdown summary, then JSON, plus the same object as structuredContent), streams progress
notifications, and is capped by its own budget_usd (and the config's global cap). Failures come
back as structured errors with what is missing and a next_action; diverge_status is free."""
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

from .alternatives import ALT_FIELDS
from .llm import BudgetExceeded
from .pipeline import (Session, baseline_scores, check_approach, effective_n, ensure_family_scores, ensure_map,
                       record_heldout_use)
from .reality import Evaluator
from .task import Task

INSTRUCTIONS = """DIVERGE helps AI agents find, test and pick alternatives when they keep landing on the same answer. \
Built for agents: it shows what you (and most models) converge on, proposes genuinely different, unconventional \
alternatives, and returns a VERDICT and next_action you can act on without a human. Different code isn't always \
different thinking.
Use it on strategy decisions and strategy stagnation, ideally on a task with a score (evaluator_cmd, or a built-in \
task: online_bin_packing, cache_eviction). Not for generic debugging or build errors.
- Unsure whether DIVERGE is configured here: diverge_status (free, no network).
- Before committing to a strategy: diverge_check. Is your plan just the default answer most models give? It \
returns alternatives with strengths and traps; free-text mode needs no evaluator.
- After each scored attempt: diverge_loop_check ($0, no model call) says whether you have stalled and whether \
your attempts fail on the same cases, i.e. whether to call diverge_escape.
- You keep getting the same answer: several scored attempts have stalled, the same failure keeps coming back, lots \
of code change but little score change, or you are about to retry another variation or escalate to a human. Call \
diverge_escape yourself; do not wait for the user to ask. It tells you whether your attempts are one strategy in \
different code, and proposes and tests an unconventional alternative on your evaluator. Not for a first attempt. \
Follow its VERDICT and next_action.
- Pass what you know: already_tried (approaches with outcomes), constraints (hard rules) and context (your \
real figures). After testing an idea, record it with diverge_report_outcome ($0) so it is never offered again.
- Evidence is labeled MEASURED, SOURCE_BACKED or HYPOTHESIZED. Never treat HYPOTHESIZED evidence as measured.
- diverge_map (the consensus map: what most models converge on), blind_spots and reality_check are lower-level \
research primitives.
evaluator_cmd takes a code file path as its last argument and prints {"valid": bool, "score": float}; lower is \
better unless lower_is_better=false. Errors come back as JSON with error, missing and next_action. Never invent \
a credential: a missing key needs the user."""

mcp = MCPServer("diverge", instructions=INSTRUCTIONS)


def compact(data):
    return json.dumps(data, separators=(",", ":"), default=str)


BUILTIN_TASKS = ("online_bin_packing", "cache_eviction")


def failure(error, detail, next_action, missing=(), partial=None, can_continue=False):
    """A structured error an agent can act on (never contains key values)."""
    from .demo import OFFLINE_DEMO, REPO
    out = {"error": error, "detail": detail, "missing": list(missing), "complete": False,
           "live_calls_available": bool(os.environ.get("ANTHROPIC_API_KEY" if os.environ.get("DIVERGE_PROVIDER", "")
                                                       .lower() == "anthropic" else "OPENROUTER_API_KEY")),
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
            action = ("The global spend cap (budget_usd in config.yaml) is reached. Stop calling DIVERGE's live tools and "
                      "ask the user whether to raise it.")
        else:
            action = ("This call's budget_usd was reached. Completed calls are cached, so the same call with a larger "
                      "budget_usd resumes where it stopped; ask the user before raising spend.")
        return failure("budget_exceeded", msg, action, ["budget"], partial)
    missing_key = re.search(r"\b([A-Z][A-Z0-9_]*_API_KEY) is not set", msg)
    if isinstance(e, RuntimeError) and missing_key:  # diverge.llm.require_env, possibly wrapped
        key = missing_key.group(1)
        return failure("missing_requirement", f"{key} is not set", (
            f"Live DIVERGE calls need {key}, which only the user can provide (in .env); ask them, and never invent or "
            f"guess a credential. Meanwhile the offline demo works: {OFFLINE_DEMO_COMMAND} (no API calls). "
            f"diverge_status shows what is configured."), [key], partial)
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
        session = None
        try:
            session = Session(budget_usd, progress)
            result = work(session)
        except Exception as e:  # any failure becomes a structured answer the agent can read
            result = classify(e, session.ledger.spent_this_run if session else 0.0)
        return {**result, "spent_usd": round(session.ledger.spent_this_run, 4) if session else 0.0}

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
async def diverge_map(task_description: str, ctx: Context, budget_usd: float = 5.0) -> str:
    """Research primitive: which solution families do AI models converge on for this task? Call it
    when you want the landscape itself; to act, prefer diverge_check (choosing a strategy) or diverge_escape
    (stalled attempts). Not for debugging. Needs OPENROUTER_API_KEY, and EXA_API_KEY when the map is
    not cached yet. Default budget_usd 5.0; about $0 once cached. Cold start: the first call on a task without a cached consensus map builds one (samples from every configured model, an Exa literature search, labeling): several minutes and real spend, capped by budget_usd (recorded 100-samples-per-model builds cost about $3); cached after (diverge_status lists cached tasks).
    Returns task, n_samples, low_confidence, concentration_top3, top3, families [[family, share,
    count]], per_model, spent_usd."""
    return await run_tool(ctx, budget_usd,
                          lambda session: map_summary(ensure_map(session, Task.resolve(task_description)), session.cfg))


@mcp.tool()
async def blind_spots(task_description: str, ctx: Context, budget_usd: float = 5.0) -> str:
    """Research primitive: solution families from the literature that model samples rarely or
    never used. Call it when you want unexplored, sourced directions to read; to decide whether to
    switch, prefer diverge_check, or diverge_escape once attempts stall. Needs OPENROUTER_API_KEY, and
    EXA_API_KEY when the map is not cached yet. Default budget_usd 5.0; shares diverge_map's cached map
    ($0 after it is built). Cold start: the first call on a task without a cached consensus map builds one (samples from every configured model, an Exa literature search, labeling): several minutes and real spend, capped by budget_usd (recorded 100-samples-per-model builds cost about $3); cached after (diverge_status lists cached tasks).
    Returns low_confidence (true: too few samples per model to tell an overlooked family from an
    unlucky one) and blind_spots [{family, mechanism, observed (k of n), share_upper_95 (95% upper
    bound on the true share), per_model_counts, source_urls}], spent_usd."""
    def work(session):
        study = ensure_map(session, Task.resolve(task_description))
        return {"low_confidence": low_confidence(study, session.cfg),
                "blind_spots": [{"family": b["family"], "mechanism": b["one_line_mechanism"],
                                 "observed": b["observed"], "share_upper_95": b["share_upper_95"],
                                 "per_model_counts": b["per_model_counts"], "source_urls": b["source_urls"]}
                                for b in study["blind_spots"]]}
    return await run_tool(ctx, budget_usd, work)


@mcp.tool()
async def reality_check(task_description: str, family: str, ctx: Context, evaluator_cmd: str = "",
                        k: int = 3, lower_is_better: bool = True, baseline_files: list[str] | None = None,
                        contrast: bool = False, budget_usd: float = 3.0) -> str:
    """Research primitive: implement one named family (from diverge_map or blind_spots) k times
    (default 3) and score each with the evaluator. Call it to test one specific family; prefer
    diverge_check, which chooses the families to test, or diverge_escape for stalled attempts. Needs an
    evaluator (evaluator_cmd, or a built-in task) and OPENROUTER_API_KEY; an unknown family returns
    error unknown_family with the valid names. Default budget_usd 3.0 (a hard cap, $3): k implementer
    calls plus sandboxed evaluator runs; builds the consensus map first if it is not cached (see diverge_map). The best
    implementation is picked on dev; final_score is its held-out score on built-in tasks (dev for
    evaluator_cmd; see scored_on). Returns family, kind, share, n_valid, n_invalid, dev_best_score,
    final_score, scored_on, baselines, beats_baselines, best_tested_consensus (the best already-scored consensus family), best_code, invalid_errors,
    spent_usd. contrast=True adds the consensus-contrast section (experimental; see README)."""
    def work(session):
        task = Task.resolve(task_description)
        study = ensure_map(session, task)
        by_name = {f["family"]: f for f in study["families"]}
        if family not in by_name:
            return {**failure("unknown_family", f"{family!r} is not a family in this task's consensus map",
                              "Call again with one of the names in families.", ["family"], can_continue=True),
                    "families": sorted(by_name)}
        evaluator = Evaluator(task, evaluator_cmd or None, lower_is_better)
        scores = ensure_family_scores(session, task, study, [by_name[family]], evaluator, k, contrast)
        s, out = scores[family], session.out_dir(task)
        # Only consensus families that were already scored under this evaluator; none are tested here.
        consensus_scores = {f: v["final_score"] for f, v in scores.items()
                       if v["kind"] == "herd" and v.get("final_score") is not None}
        baselines = baseline_scores(task, evaluator, baseline_files or ())
        result = {"family": family, "kind": s["kind"], "share": s["share"], "k": s["k"],
                  "n_valid": s["n_valid"], "n_invalid": s["n_invalid"],
                  "dev_best_score": s["best_score"], "dev_median_score": s["median_score"],
                  "final_score": s["final_score"], "scored_on": s["final_split"],
                  "invalid_errors": [m["error"] for m in s["implementations"] if not m["valid"]],
                  "baselines": baselines, "compared_herd_families": sorted(consensus_scores), "best_tested_herd": None,
                  "compared_consensus_families": sorted(consensus_scores), "best_tested_consensus": None}
        if evaluator.has_heldout and s["final_score"] is not None:
            prior = record_heldout_use(session, task, evaluator)
            result["heldout_prior_uses"], result["exploratory"] = prior, prior > 0
        if s["final_score"] is not None:
            result["beats_baselines"] = [b for b, v in baselines.items()
                                         if v is not None and evaluator.beats(s["final_score"], v)]
        if s["best_code_path"]:
            result["best_code"] = (out / s["best_code_path"]).read_text()
        if consensus_scores:
            top = evaluator.best(consensus_scores.values())
            result["best_tested_herd"] = {"family": next(f for f, v in consensus_scores.items() if v == top),
                                          "final_score": top}
            result["best_tested_consensus"] = result["best_tested_herd"]
        return result
    return await run_tool(ctx, budget_usd, work)


ALTERNATIVES_PROMPT = """You are proposing alternative strategies for a programming task, before any code is written.

Task:
{task}

The agent's planned approach:
{approach}
{herd}
Propose {n} genuinely different strategies (different mechanisms, not tuned variants of the planned approach). \
For each give: name (a short family name), mechanism (one sentence: how it decides), strength (when it should \
beat the planned approach), trap (the most likely way it fails or misleads), how_to_test (one concrete \
measurement that would confirm or refute it, against the planned approach).
Return JSON only: {{"alternatives": [{{"name": "...", "mechanism": "...", "strength": "...", "trap": "...", \
"how_to_test": "..."}}]}}"""
N_ALTERNATIVES = (3, 5)


def cached_study(session, task):
    """The task's consensus map if one is saved for the current inputs (exactly the case where ensure_map
    returns it without building), else None. Never starts a cold-start build."""
    from .pipeline import inputs_hash, read_json
    cfg = session.cfg
    saved = read_json(session.out_dir(task) / "herd_map.json")
    inputs = {"samplers": cfg["models"]["samplers"], "labeler": cfg["models"]["labeler"],
              "n_per_model": cfg["sampling"]["n_per_model"], "inputs_hash": inputs_hash(task, cfg)}
    if not saved or {k: (saved.get("meta") or {}).get(k) for k in inputs} != inputs:
        return None
    return ensure_map(session, task)


def free_text_check(session, task, my_approach, already_tried=None, constraints=None, context="", depth="quick"):
    """diverge_check without an evaluator: the consensus comparison when a map is cached (no map is built),
    plus untested alternatives, every one HYPOTHESIZED: from one extractor-model call, or, when the agent
    passes already_tried, constraints, context or depth="deep" (or has logged outcomes for this task),
    from diverge.alternatives (screened, grounding-checked, ranked by test cost)."""
    from . import alternatives
    from .llm import ask_json
    from .overlook import extractor
    study, consensus_cmp = cached_study(session, task), None
    if study:
        placed = check_approach(session, task, study, my_approach, None, 0)
        m = study["herd_map"]
        consensus_cmp = {"family": placed["family"], "why": placed.get("why", ""), "share": placed["share"],
                    "rank": placed["rank"], "in_top_families": placed["family"] in m["top3"],
                    "is_blind_spot": placed["is_blind_spot"],
                    "top_families": [[r["family"], r["share"]] for r in m["families"] if r["family"] in m["top3"]],
                    "concentration_top3": m["concentration_top3"], "n_samples": m["n_labeled"],
                    "low_confidence": low_confidence(study, session.cfg),
                    "blind_spots": [b["family"] for b in study["blind_spots"]], "evidence_class": "MEASURED"}
    consensus_text = ("\nAI models asked this task mostly converge on: " + ", ".join(m["top3"]) +
                 ". Prefer strategies outside these.\n") if study else ""
    out = {"verdict": "untested", "mode": "free_text", "my_approach": my_approach, "herd_comparison": consensus_cmp,
           "consensus_comparison": consensus_cmp}
    if alternatives.uses_inputs(already_tried, constraints, context, depth,
                                alternatives.logged_outcomes(task)):
        out.update(alternatives.grounded_alternatives(session, task, my_approach, consensus_text, already_tried,
                                                      constraints, context, depth, N_ALTERNATIVES[1]))
        alts = out["alternatives"]
    else:
        reply = ask_json(session.llm, extractor(session), ALTERNATIVES_PROMPT.format(
            task=task.prompt, approach=my_approach, herd=consensus_text, n=N_ALTERNATIVES[1]), required=("alternatives",))
        alts = [{k: str(a.get(k) or "").strip() for k in ALT_FIELDS} | {"evidence_class": "HYPOTHESIZED"}
                for a in reply.get("alternatives") or [] if isinstance(a, dict) and a.get("name")][:N_ALTERNATIVES[1]]
        out["alternatives"] = alts
    out["complete"] = True
    if not study:
        out["herd_note"] = out["consensus_note"] = (
            "No cached consensus map for this task, so the consensus comparison was skipped; free-text mode "
            "never builds one (diverge_map builds it: cold start, several minutes and real spend).")
    if len(alts) < N_ALTERNATIVES[0]:
        out["alternatives_note"] = f"the model returned {len(alts)} usable alternatives"
    return out


@mcp.tool(structured_output=False)
async def diverge_check(task_description: str, my_approach: str, ctx: Context, evaluator_cmd: str = "",
                     k: int = 3, lower_is_better: bool = True, contrast: bool = False,
                     budget_usd: float = 5.0, already_tried: list[dict | str] | None = None,
                     constraints: list[str] | None = None, context: str = "",
                     depth: str = "quick") -> CallToolResult:
    """Call before committing to a strategy: is my_approach (a plain-language description) just the
    default answer most AI models give? Get alternatives with strengths and traps (free-text mode needs
    no evaluator), or, with an evaluator, which family your plan belongs to, how common it is, and
    whether a tested family beats it. Not for debugging a failing solution; once scored attempts have stalled, use diverge_escape. Needs
    OPENROUTER_API_KEY (and EXA_API_KEY when the map is not cached yet). With an evaluator
    (evaluator_cmd, or a built-in task) it implements your family, the blind spots and the top consensus
    families k times each (default 3) and scores them (held-out on built-in tasks). Default budget_usd 5.0. Cold start: the first call on a task without a cached consensus map builds one (samples from every configured model, an Exa literature search, labeling): several minutes and real spend, capped by budget_usd (recorded 100-samples-per-model builds cost about $3); cached after (diverge_status lists cached tasks).
    Free-text mode (no evaluator): never builds a map; compares my_approach with the consensus only if a map is
    cached, and returns 3-5 untested alternatives {name, mechanism, strength, trap, how_to_test}, all
    HYPOTHESIZED, from one cheap model call (about $0.01; verdict "untested").
    Returns "VERDICT: <verdict>", a short markdown summary, then JSON (also the structuredContent):
    verdict keep | try_blind_spot | try_other_family (also inconclusive, unclassified, untested), complete,
    next_action, your family and share, blind_spot or better_family, diverse_alternative when a blind
    spot also beat yours, and spent_usd. contrast=True is experimental (see README).
    Pass what you know (free-text mode; all optional):
    - already_tried [{approach, outcome, score}] (or plain strings): never offered again; each alternative
      names the outcome it builds_on. Outcomes logged with diverge_report_outcome are added automatically.
    - constraints [str]: hard rules (scope, hardware, budget, "no new dependencies"). Alternatives that
      duplicate already_tried or break a rule are dropped by a separate reviewer model and listed in
      filtered_out with the reason; relax_hint names the one rule whose relaxation would open the most.
    - context: your real figures (a log tail, a results table; redacted, capped at 8000 characters).
      Alternatives quote the figures they rely on (grounding); ungrounded_numbers flags numbers that do
      not appear in your context (a deterministic check).
    - depth "quick" (default) or "deep": 3 isolated samples from models of different companies at
      different temperatures, one critic that wrote none of them (novelty, constraint_fit, testability),
      same-mechanism repeats dropped; costs cents, not dollars (models_used and spent_usd are returned).
    With any of these, each alternative also has test_cost (minutes | hours | days), decisive and
    kill_result, sorted decisive and cheapest first (coarse, HYPOTHESIZED estimates), and next_action
    names the first one's kill test. The evaluator path ignores these inputs (agent_inputs_note says so)."""
    from .contract import check_result

    def work(session):
        task = Task.resolve(task_description)
        if not evaluator_cmd and not task.evaluator:
            return free_text_check(session, task, my_approach, already_tried, constraints, context, depth)
        study = ensure_map(session, task)
        evaluator = Evaluator(task, evaluator_cmd or None, lower_is_better)
        result = check_approach(session, task, study, my_approach, evaluator, k, contrast)
        if already_tried or constraints or context or depth != "quick":
            result["agent_inputs_note"] = ("already_tried, constraints, context and depth shape free-text "
                                           "alternatives and diverge_escape; this evaluator run tested consensus "
                                           "families and blind spots and did not use them.")
        return result
    text, structured = check_result(json.loads(await run_tool(ctx, budget_usd, work)))
    return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=structured)


@mcp.tool()
async def diverge_status(check_model_catalog: bool = False) -> str:
    """Free readiness check: call it first when unsure whether DIVERGE is configured here, or after a
    missing_requirement error. Costs $0: no model, search or network call (check_model_catalog=true
    adds one free, unauthenticated GET of OpenRouter's public model list; optional). Reports whether
    each key is configured, never its value. Returns ready, live_mode_available,
    remaining_budget_usd, requirements {openrouter, exa (optional)}, missing, offline_demo_available
    and offline_demo_command, cached_tasks (built-in tasks with a cached consensus map: no cold start),
    recommended_primary_tool, when_to_call, next_action."""
    from .demo import readiness

    def call():
        try:
            return readiness(live=check_model_catalog)
        except Exception as e:  # e.g. an unreadable config: still structured
            return failure(type(e).__name__, str(e), "Report this error to the user.")
    return compact({**await anyio.to_thread.run_sync(call), "spent_usd": 0.0})


LEVER_POLICIES = ("chosen", "random", "mixed")


def escape_result(r):
    """An escape_v2 result in the shape diverge.contract reads (the fields escape v1 returned)."""
    d, rep, lever = r["diagnosis"], r.get("analysis") or {}, r.get("lever")
    first = [{k: lever.get(k) for k in ("type", "name", "detail", "mechanism", "why")}] if lever else []
    rejections = r.get("rejections") or []
    return {"loop_detected": rep.get("loop_detected"), "strategy_similarity": rep.get("strategy_similarity"),
            "code_similarity": rep.get("code_similarity"), "shared_strategy": d["shared_cause"],
            "shared_assumptions": (rep.get("shared") or {}).get("assumptions", []),
            "herd_prevalence": d["herd_prevalence"], "consensus_prevalence": d["herd_prevalence"],
            "overlooked_levers": first + list(r.get("lever_alternatives") or []),
            "recommended_switch": {"mechanism": lever.get("mechanism") if lever else None,
                                   "why": lever.get("why") if lever else "no lever found"},
            "candidate": r.get("candidate"), "candidate_path": r.get("candidate_path"),
            "candidates": [{k: c.get(k) for k in ("path", "valid", "score", "error")} for c in r.get("admitted") or []],
            "evaluation": r["evaluation"], "action": r["action"], "diagnosis": d, "reasons": d["reasons"],
            "escape": {"version": 2, "lever_policy": r.get("lever_policy"), "lever_source": r.get("lever_source"),
                       "sketches": len(r.get("sketches") or []), "kept_sketches": len(r.get("kept_sketches") or []),
                       "admitted": len(r.get("admitted") or []), "rejected": len(rejections),
                       "rejections": [{k: x.get(k) for k in ("stage", "sketch", "reasons")} for x in rejections[:5]],
                       "counterexample": (r.get("counterexample") or {}).get("text")}}


def fallback_lever(session, task, tried, rules, context, policy="chosen", seed=0):
    """lever_fallback for escape_v2 on tasks with no consensus data: a grounded alternative to the attempts'
    shared strategy (diverge.alternatives, one model call plus a screen), HYPOTHESIZED. The shared strategy
    itself goes in as already tried, so the screen drops alternatives that restate it (the planted-answer
    control's fallback proposed the stuck approach in 4 of 5 states without this). policy "random" draws one
    of the screened alternatives (seeded) instead of taking the top-ranked one."""
    import random

    from . import alternatives

    def make(diagnosis, report):
        stuck = {"approach": diagnosis["shared_cause"], "source": "attempts",
                 "outcome": f"stalled: {diagnosis.get('failure') or 'no progress'}"}
        try:
            got = alternatives.grounded_alternatives(session, task, diagnosis["shared_cause"], "", [stuck, *tried],
                                                     rules, context, "quick", N_ALTERNATIVES[1])
        except BudgetExceeded:
            raise
        except Exception:
            return None
        if not got["alternatives"]:
            return None
        a = got["alternatives"][0] if policy != "random" else \
            random.Random(f"{diagnosis['shared_cause']}:{seed}").choice(got["alternatives"])
        return {"type": "agent_alternative", "name": a["name"], "detail": {"description": a["mechanism"]},
                "mechanism": a["mechanism"],
                "why": "HYPOTHESIZED: proposed by a model from your attempts' shared strategy"
                       + (", already_tried, constraints and context" if tried or rules or context else "")
                       + " (no consensus data for this task)"}
    return make


def escape_for_agent(session, task, current_solution, previous_attempts, evaluator, k=3, counterexamples=True,
                     notes="", lever_policy="chosen", tried=(), rules=(), context=""):
    """diverge.escape.escape_v2 (the version the rescue tests measured) in the contract's shape, plus what
    the agent needs to trust it: strategy counterexamples between its attempts and the escape candidate
    (tasks with a probe mode), and how stable the consensus map is under rewording (the paraphrase test).
    lever_policy "mixed": the chosen lever, then, if its candidate did not improve, a random lever."""
    from . import counterexample
    from .escape import escape_v2
    from .pipeline import read_json
    from .strategy import llm_extractor
    if lever_policy not in LEVER_POLICIES:
        raise ValueError(f"lever_policy must be one of {list(LEVER_POLICIES)}, got {lever_policy!r}")
    out = session.out_dir(task)
    overlook, families = read_json(out / "overlook.json"), read_json(out / "blind_spots.json")

    def run(policy, seed):
        return escape_v2(session, task, current_solution, previous_attempts, evaluator, overlook=overlook,
                         families=families, extract=llm_extractor(session, task), k_sketches=max(1, k), seed=seed,
                         notes=notes, lever_policy=policy, only_on_switch=True,
                         lever_fallback=fallback_lever(session, task, list(tried), list(rules), context, policy, seed))
    raw = run("random" if lever_policy == "random" else "chosen", 0)
    tries = [raw]
    if lever_policy == "mixed" and raw["action"] == "switch_strategy" and not raw["evaluation"]["improved"]:
        try:
            second = run("random", 1)
            tries.append(second)
            if second["evaluation"]["after"] is not None and (
                    raw["evaluation"]["after"] is None or evaluator.beats(second["evaluation"]["after"],
                                                                          raw["evaluation"]["after"])):
                raw = second
        except BudgetExceeded:
            tries.append(None)
    result = escape_result(raw)
    if lever_policy == "mixed":
        result["escape"]["lever_attempts"] = [
            {"policy": t["lever_policy"], "lever": (t.get("lever") or {}).get("name"),
             "after": t["evaluation"]["after"], "improved": t["evaluation"]["improved"]} if t else
            {"policy": "random", "error": "budget_usd reached before the random lever ran"} for t in tries]
    para = read_json(out / "paraphrase.json")
    result["herd_stability"] = result["consensus_stability"] = ({"verdict": para["verdict"], "paraphrase_tvd": para["mean_P"],
                                 "resampling_tvd_max": para["max_R"]} if para else None)
    if counterexamples and result.get("candidate") and task.evaluator and hasattr(task.evaluator, "probe"):
        codes = [a if isinstance(a, str) else a["code"] for a in [*previous_attempts, current_solution]]
        found = counterexample.find(task, {f"attempt-{i}": c for i, c in enumerate(codes)},
                                    {"escape": result["candidate"]}, n=200, max_results=3)
        result["counterexamples"] = found["counterexamples"]
        result["counterexample_summary"] = found["summary"]
    return result


@mcp.tool(structured_output=False)
async def diverge_escape(task_description: str, current_solution: str, ctx: Context,
                      previous_attempts: list[str] | None = None, evaluator_cmd: str = "",
                      lower_is_better: bool = True, k: int = 3, budget_usd: float = 2.0,
                      already_tried: list[dict | str] | None = None, constraints: list[str] | None = None,
                      context: str = "", lever_policy: str = "chosen") -> CallToolResult:
    """Primary tool for when you keep getting the same answer: tells you whether your attempts are
    one strategy in different code and proposes and tests an unconventional alternative. Call it
    yourself, without waiting to be asked, when several scored attempts have stalled, the same failure
    keeps coming back, code changes a lot but the score barely moves, or you are about to retry another
    variation or hand the problem to a human. Not for a first attempt, syntax errors or generic
    debugging. Pass current_solution and previous_attempts as code (oldest first). DIVERGE scores them
    on the dev split and decides loop vs stall from measurements: behavior probes (online_bin_packing) or
    shared worst dev cases (cache_eviction, or an evaluator_cmd whose result carries per-case losses,
    detail.cases [{"id", "loss"}]). With neither it returns inconclusive and says how to add them; it
    never decides from model-written strategy descriptions. On a loop it runs escape v2: k mechanism
    sketches (default 3) built on one lever, judged, implemented in a fresh context and scored on dev
    (with counterexamples on tasks that support probes).
    task_description: a built-in task's exact name (online_bin_packing, cache_eviction; then omit
    evaluator_cmd) or free text with evaluator_cmd. evaluator_cmd runs sandboxed (no network, no writes
    outside scratch, no .env, no nested sandbox), so make it self-contained with absolute paths.
    Needs an evaluator (evaluator_cmd, or a built-in task), OPENROUTER_API_KEY, and at least 2 valid
    scored attempts for any verdict but inconclusive. It does not build a consensus map. Levers come from
    the task's cached overlook study and blind spots (built-in tasks); without them, from inputs your
    attempts never use, else one model-proposed alternative (HYPOTHESIZED). lever_policy: "chosen"
    (default: the first lever in preference order), "random" (a random lever; in the preliminary rescue
    test a random lever rescued 3/4 stuck states and the chosen one 1/4), or "mixed" (chosen, then random
    if the chosen candidate did not improve; up to twice the spend). Default budget_usd 2.0; recorded
    runs that generated no candidate spent under $0.01, generating candidates costs more.
    Returns "VERDICT: keep|repair|escape|escalate|inconclusive", a short markdown summary, then JSON: next_action,
    can_continue_autonomously, complete, missing_requirements, candidate_available,
    evidence_summary, lever, next_experiment (success and failure thresholds), evidence (each
    MEASURED, SOURCE_BACKED or HYPOTHESIZED), candidate_path, spent_usd; the same object (plus
    verdict) is the structuredContent. Follow next_action; never treat HYPOTHESIZED as measured.
    Optional, plain language: already_tried [{approach, outcome, score}] (what the attempts were and why
    each failed, plus outcomes logged with diverge_report_outcome), constraints [str] (hard rules) and
    context (your real figures; redacted, capped at 8000 characters) are added to the prompt that writes
    the escape candidate (only written when the verdict is escape); agent_inputs counts what was passed."""
    from . import alternatives, plateau
    from .contract import agent_result
    from .pipeline import read_json
    loaded = {}

    def work(session):
        task = Task.resolve(task_description)
        if not evaluator_cmd and not task.evaluator:
            return failure("evaluator_required", "pass evaluator_cmd or use a registered task", (
                "Pass evaluator_cmd (a command that takes a code file path as its last argument and prints "
                '{"valid": bool, "score": float}) or use a built-in task, then call diverge_escape again.'),
                ["evaluator_cmd"], can_continue=True)
        evaluator = RecordingEvaluator(Evaluator(task, evaluator_cmd or None, lower_is_better))
        codes = [*(previous_attempts or []), current_solution]
        logged = alternatives.logged_as_tried(alternatives.logged_outcomes(task))
        tried = alternatives.merge_tried(alternatives.clean_tried(already_tried), logged)
        rules, (notes_context, truncated) = alternatives.clean_constraints(constraints), alternatives.clean_context(context)
        result = escape_for_agent(session, task, current_solution, previous_attempts or [], evaluator, k,
                                  notes=alternatives.escape_notes(tried, rules, notes_context),
                                  lever_policy=lever_policy, tried=tried, rules=rules, context=notes_context)
        if tried or rules or notes_context:
            result["agent_inputs"] = {"already_tried": len(tried), "logged_outcomes": len(logged),
                                      "constraints": len(rules), "context_chars": len(notes_context),
                                      "context_truncated": truncated}
        result["attempts"] = evaluator.attempts(codes)
        result["lower_is_better"] = evaluator.lower_is_better
        plateau.attach(result, task.baselines("dev"))  # None for external evaluators (no built-in baselines)
        if result["attempts"] and not any(a["valid"] for a in result["attempts"]):
            # Every attempt failed to score: the evaluator (not the strategy) is the problem, so
            # "get more scored attempts" would send the agent in a loop.
            first = next((a["error"] for a in result["attempts"] if a["error"]), None) or "no JSON result"
            builtin = [n for n in BUILTIN_TASKS if n in task_description and n != task_description.strip()]
            hint = (f" This looks like the built-in task {builtin[0]!r}: pass task_description={builtin[0]!r} "
                    f"exactly and omit evaluator_cmd." if builtin else "")
            return failure("evaluator_failed", f"none of the {len(result['attempts'])} attempts scored: {first[:600]}", (
                "DIVERGE could not score any attempt, so it cannot judge the strategy." + hint + " An evaluator_cmd runs "
                "inside DIVERGE's sandbox: no network, no writes outside a scratch directory, no reading .env or home "
                "secrets, and it cannot start another sandbox; use a self-contained command with absolute paths. "
                "Check it by scoring one file yourself, then call diverge_escape again."),
                ["working evaluator"], can_continue=True)
        out = session.out_dir(task)
        loaded.update(overlook=read_json(out / "overlook.json"), paraphrase=read_json(out / "paraphrase.json"),
                      families=[*(read_json(out / "blind_spots.json") or []), *(read_json(out / "families.json") or [])])
        return result
    # Text: "VERDICT: <verdict>" (contract.parse reads it), a markdown summary, then JSON; the same object is also
    # structuredContent. structured_output=False: no outputSchema, so the dict is passed through as is.
    text, structured = agent_result(json.loads(await run_tool(ctx, budget_usd, work)), **loaded)
    return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=structured)


@mcp.tool()
async def diverge_loop_check(scores: list[float | None], failing_cases: list[list[str]] | None = None,
                             lower_is_better: bool = True) -> str:
    """Free triage before diverge_escape: call it after each scored attempt to decide whether you are
    stuck. Costs $0: no model, search or network call, no code is run. scores: your attempts' scores,
    oldest first (None for an attempt that failed). failing_cases (optional, same order): for each
    attempt, the ids of the cases it does worst on (e.g. its 2 worst tests or records). Uses the same
    rules as diverge_escape: stalled = no improvement above 1% over the last 3 attempts; a loop = stalled
    and the recent attempts' failing cases overlap (Jaccard >= 0.8). Returns decision
    (keep_going | call_diverge_escape | repair | stalled_unknown | need_more_attempts), stalled,
    improvement_last3, same_failures, failure_jaccard_min, reasons, next_action."""
    from .trajectory import EPSILON, RECENT, SAME_FAILURES, improvement_last3
    from .failures import jaccard
    valid = [s for s in scores if s is not None]
    imp = improvement_last3(scores, lower_is_better)
    out = {"n_scored": len(valid), "improvement_last3": imp, "stalled": None, "same_failures": None,
           "failure_jaccard_min": None, "reasons": []}
    if len(valid) < 2:
        out.update(decision="need_more_attempts", next_action="Score at least 2 attempts, then call again.")
        return compact({**out, "spent_usd": 0.0})
    out["stalled"] = imp <= EPSILON
    out["reasons"].append(f"improvement over the last 3 attempts {imp:+.1%} (stalled at or below {EPSILON:.0%})")
    recent = [set(c) for c in (failing_cases or [])[-RECENT:] if c]
    if failing_cases and len(recent) >= 2 and len(recent) == min(RECENT, len(failing_cases)):
        jmin = min(jaccard(a, b) for i, a in enumerate(recent) for b in recent[i + 1:])
        out.update(failure_jaccard_min=round(jmin, 3), same_failures=jmin >= SAME_FAILURES)
        out["reasons"].append(f"recent attempts' failing cases overlap: min Jaccard {jmin:.2f} "
                              f"({'same' if jmin >= SAME_FAILURES else 'different'} cases)")
    if not out["stalled"]:
        out.update(decision="keep_going", next_action="You are still improving: keep going and score each change.")
    elif out["same_failures"]:
        out.update(decision="call_diverge_escape", next_action=(
            "Stalled and failing on the same cases: a strategy loop. Call diverge_escape with your attempts (and "
            "already_tried, constraints, context) before trying another variation."))
    elif out["same_failures"] is False:
        out.update(decision="repair", next_action=(
            "Stalled, but your attempts fail on different cases: fix the current strategy (bug, tuning, edge "
            "cases) rather than switching."))
    else:
        out.update(decision="stalled_unknown", next_action=(
            "Stalled. Pass failing_cases (each attempt's worst cases) to tell a loop from a fixable stall, or call "
            "diverge_escape with an evaluator that reports per-case losses (detail.cases)."))
    return compact({**out, "spent_usd": 0.0})


@mcp.tool()
async def diverge_report_outcome(task_description: str, alternative: str, outcome: str, ctx: Context,
                                 note: str = "", score: float | None = None) -> str:
    """Record what happened when you tested an alternative DIVERGE suggested (or any approach), so later
    diverge_check and diverge_escape calls on the same task_description treat it as already tried and
    never offer a failed idea again. Call it after each test of an idea. Costs $0: no model or network
    call. outcome: LEGIT (it worked), WEAK
    (it helped a little or is unclear) or WRONG (it failed or was refuted); note says why; score is
    optional. The log is only replayed as already_tried; DIVERGE does not learn weights or re-rank from
    it (far too little data: tuning on it would overfit). Returns recorded, n_outcomes, next_action."""
    from . import alternatives

    def call():
        try:
            task = Task.resolve(task_description)
            row = alternatives.record_outcome(task, alternative, outcome, note, score)
            n = len(alternatives.logged_outcomes(task))
            return {"recorded": row, "n_outcomes": n, "complete": True,
                    "next_action": "Recorded. Later diverge_check and diverge_escape calls on this exact "
                                   "task_description include it in already_tried automatically."}
        except ValueError as e:
            return failure("invalid_argument", str(e), "Fix the argument and call diverge_report_outcome again.",
                           ["outcome" if "outcome" in str(e) else "alternative"], can_continue=True)
        except Exception as e:
            return failure(type(e).__name__, str(e), "Report this error to the user.")
    return compact({**await anyio.to_thread.run_sync(call), "spent_usd": 0.0})


class RecordingEvaluator:
    """An Evaluator that remembers each dev result by file name, so diverge_escape can report the
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
            out.append({"id": f"attempt-{i}", "score": r.get("score"), "valid": r.get("valid"),
                        "error": (r.get("detail") or {}).get("error") if r and not r.get("valid") else None})
        return out


def serve():
    mcp.run("stdio")
