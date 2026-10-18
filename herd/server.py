"""HERD as an MCP server for coding agents. Every tool returns compact JSON, streams
progress notifications, and is capped by its own budget_usd (and the config's global cap)."""
import asyncio
import json
import threading

import anyio
from mcp.server.mcpserver import Context, MCPServer

from .llm import BudgetExceeded
from .pipeline import Herd, baseline_scores, check_approach, ensure_family_scores, ensure_map
from .reality import Evaluator
from .task import Task

mcp = MCPServer("herd", instructions=(
    "HERD maps which solution families AI models converge on for a task, finds sourced families "
    "they overlook, and tests them with an evaluator. task_description is the full task prompt, "
    "or a registered task name (online_bin_packing, cache_eviction). evaluator_cmd is a command "
    "that takes a code file path as its last argument and prints {\"valid\": bool, \"score\": float} "
    "JSON; lower scores are better unless lower_is_better is false."))


def compact(data):
    return json.dumps(data, separators=(",", ":"), default=str)


async def run_tool(ctx, budget_usd, work):
    """Run blocking pipeline work in a thread, forwarding its progress messages to the client."""
    loop, lock, step = asyncio.get_running_loop(), threading.Lock(), [0]

    def progress(message):
        with lock:
            step[0] += 1
            asyncio.run_coroutine_threadsafe(ctx.report_progress(step[0], None, message), loop)

    def call():
        herd = Herd(budget_usd, progress)
        try:
            result = work(herd)
        except BudgetExceeded as e:
            result = {"error": "budget_exceeded", "detail": str(e)}
        except (RuntimeError, ValueError) as e:
            result = {"error": type(e).__name__, "detail": str(e)}
        return {**result, "spent_usd": round(herd.ledger.spent_this_run, 4)}

    return compact(await anyio.to_thread.run_sync(call))


def low_confidence(study, cfg):
    """Too few samples per model to tell an overlooked family from an unlucky one."""
    return study["herd_map"]["meta"]["n_per_model"] < cfg["reality_check"]["min_samples_per_model"]


def map_summary(study, cfg):
    m = study["herd_map"]
    return {"task": m["task"], "n_samples": m["n_labeled"], "n_unlabeled": m["n_unlabeled"],
            "low_confidence": low_confidence(study, cfg),
            "concentration_top3": m["concentration_top3"], "top3": m["top3"],
            "families": [[r["family"], r["share"], r["count"]] for r in m["families"]],
            "per_model": m["per_model"]}


@mcp.tool()
async def herd_map(task_description: str, ctx: Context, budget_usd: float = 5.0) -> str:
    """Which solution families do AI models converge on for this task? Returns family shares,
    overall and per model, and the share held by the top 3. Built on first call, cached after."""
    return await run_tool(ctx, budget_usd,
                          lambda herd: map_summary(ensure_map(herd, Task.resolve(task_description)), herd.cfg))


@mcp.tool()
async def blind_spots(task_description: str, ctx: Context, budget_usd: float = 5.0) -> str:
    """Solution families from the literature that fewer than 2% of model samples use,
    with their mechanism and source URLs."""
    def work(herd):
        study = ensure_map(herd, Task.resolve(task_description))
        return {"low_confidence": low_confidence(study, herd.cfg),
                "blind_spots": [{"family": b["family"], "mechanism": b["one_line_mechanism"],
                                 "share": b["share"], "source_urls": b["source_urls"]}
                                for b in study["blind_spots"]]}
    return await run_tool(ctx, budget_usd, work)


@mcp.tool()
async def reality_check(task_description: str, family: str, ctx: Context, evaluator_cmd: str = "",
                        k: int = 3, lower_is_better: bool = True, baseline_files: list[str] | None = None,
                        budget_usd: float = 3.0) -> str:
    """Implement one solution family k times and score each with the evaluator. Returns the best
    code and score, the median score, and how it compares with baselines and the top herd family."""
    def work(herd):
        task = Task.resolve(task_description)
        study = ensure_map(herd, task)
        by_name = {f["family"]: f for f in study["families"]}
        if family not in by_name:
            return {"error": "unknown_family", "families": sorted(by_name)}
        evaluator = Evaluator(task, evaluator_cmd or None, lower_is_better)
        scores = ensure_family_scores(herd, task, study, [by_name[family]], evaluator, k)
        s, out = scores[family], herd.out_dir(task)
        herd_scores = {f: v["best_score"] for f, v in scores.items()
                       if v["kind"] == "herd" and v["best_score"] is not None}
        baselines = baseline_scores(task, evaluator, baseline_files or ())
        result = {"family": family, "kind": s["kind"], "share": s["share"], "k": s["k"],
                  "n_valid": s["n_valid"], "n_invalid": s["n_invalid"],
                  "best_score": s["best_score"], "median_score": s["median_score"],
                  "invalid_errors": [m["error"] for m in s["implementations"] if not m["valid"]],
                  "baselines": baselines}
        if s["best_score"] is not None:
            result["beats_baselines"] = [b for b, v in baselines.items()
                                         if v is not None and evaluator.beats(s["best_score"], v)]
            result["best_code"] = (out / s["best_code_path"]).read_text()
        if herd_scores:
            top = evaluator.best(herd_scores.values())
            result["best_tested_herd"] = {"family": next(f for f, v in herd_scores.items() if v == top),
                                          "best_score": top}
        return result
    return await run_tool(ctx, budget_usd, work)


@mcp.tool()
async def herd_check(task_description: str, my_approach: str, ctx: Context, evaluator_cmd: str = "",
                     k: int = 3, lower_is_better: bool = True, budget_usd: float = 5.0) -> str:
    """Which family does my_approach belong to and how common is it? With an evaluator (or a
    registered task), also tests it against the blind spots and top herd families and returns
    verdict "keep", "try_blind_spot" (with the winning family) or "try_other_family"."""
    def work(herd):
        task = Task.resolve(task_description)
        study = ensure_map(herd, task)
        evaluator = Evaluator(task, evaluator_cmd or None, lower_is_better) \
            if evaluator_cmd or task.evaluator else None
        return check_approach(herd, task, study, my_approach, evaluator, k)
    return await run_tool(ctx, budget_usd, work)


def serve():
    mcp.run("stdio")
