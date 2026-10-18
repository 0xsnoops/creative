# Autonomous tool selection: does a coding agent pick DIVERGE on its own?

**Harness.** `scripts/autonomous_demo.py --run N`:
1. Creates a fresh temporary project and runs `python -m diverge onboard --client claude --target <project>`.
   The project then holds only the installed skill, the `/diverge` command and `.mcp.json`.
2. Copies in the stalled bin-packing trajectory (`fixtures/trajectories/strategy_loop.json`) as
   `attempts/` with its dev scores, the task statement (`TASK.md`) and an evaluator script (`evaluate.py`).
3. Runs a separate headless agent (`claude -p`, stream-json output) in that project, with this prompt:

> Previous attempts at the task in TASK.md and their dev scores are in attempts/. Improve the dev
> score using the evaluator (`python evaluate.py <file.py>`). Stop when you can't make progress.

The prompt never mentions DIVERGE. No human intervened during any run. The agent kept its own
Claude Code login, and API keys were kept out of its environment.

**Every run is reported, including misses.**

## Run 1: the agent chose DIVERGE itself, and the verdict was unusable

Transcript: `docs/demo/autonomous-run-1.jsonl`. Summary: `docs/demo/autonomous-run-1.summary.json`.
This run predates the rename, so the transcript shows the old tool names.

**The moment of selection.** The first DIVERGE call is at transcript line 478. The agent had made
51 earlier tool calls (reading, evaluating and tuning its own solution). The agent's words just
before it:

> Gains are now within noise: 0.094 → 0.085 on the tuning seeds, flat on the validation seeds.
> I'll check HERD for a different strategy before tuning any further.

**Calls**
1. **Readiness tool (`diverge_status` today), no arguments.** Result: ready, with
   `online_bin_packing` among the cached tasks.
2. **Escape tool (`diverge_escape` today), line 512.**
   - Arguments: a long `task_description` beginning with the benchmark name, its current solution,
     two previous attempts, `evaluator_cmd` = the project's `evaluate.py`, and `budget_usd` = 1.
   - Verdict: `inconclusive`. Every attempt scored invalid, so the next action told the agent to
     get more scored attempts. The tool cost $0.0022.

**Why the verdict was unusable.** The project's `evaluate.py` calls DIVERGE's built-in evaluator,
which starts its own sandbox. DIVERGE runs every `evaluator_cmd` inside its sandbox, and macOS
does not allow a second sandbox inside the first. While reporting that failure, the secret
redaction step also crashed because it could not read `.env`. The resulting next action ("get
more scored attempts") was misleading.

**Fixed after run 1 (commit 16c8980)**
- Redaction tolerates an unreadable `.env`.
- If every attempt fails to score, the tool returns a structured `evaluator_failed` error with
  the first error and the sandbox rules, instead of `inconclusive`.
- The tool description and the skill say to pass a built-in task by its exact name, with no
  `evaluator_cmd`.

**Final result.** The agent kept working on its own. Its `solution.py` scored **0.1596 on dev**,
against 0.658 for the best previous attempt. That gain is the agent's own work, not DIVERGE's.
It then stopped and explained that further tuning was fitting noise.

**Costs.** The agent itself cost $2.53, as reported by `claude -p`. DIVERGE cost $0.0022.

## Run 2: aborted (counted as one of the two allowed revisions)

The description and skill changes above were revision 1. Run 2 started after the rename, but
two things went wrong:
- **It could not have called DIVERGE.** The harness still allowlisted the old MCP tool prefix,
  so the renamed `mcp__diverge` tools were never permitted.
- **It was stopped partway.** The user disabled OpenRouter, which DIVERGE then used for model
  calls.

Both are fixed: the harness allowlists `mcp__diverge`, and DIVERGE can call the Anthropic API
directly (`DIVERGE_PROVIDER=anthropic`). Run 2 produced no result.

## Run 3: same prompt, DIVERGE on the Anthropic API

Transcript: `docs/demo/autonomous-run-3.jsonl`. Summary: `docs/demo/autonomous-run-3.summary.json`.
This was revision 2 of the allowed two. The fixes were the evaluator guidance and the
`mcp__diverge` allowlist, with DIVERGE calling the Anthropic API (`DIVERGE_PROVIDER=anthropic`).

**The moment of selection.** The first DIVERGE call is at transcript line 586, after the agent's
own tuning. The agent's words just before it:

> Parameter tuning has converged around 0.116 on train (dev 0.1496). I'll check whether the
> DIVERGE tool is set up here so I can ask it for a different strategy.

**Calls**
1. **`diverge_status`, no arguments.** Result: ready, with the Anthropic key configured and
   `online_bin_packing` cached.
2. **`diverge_escape`, at line 599.**
   - Arguments: `task_description` = `online_bin_packing` (the built-in name, as the revised
     description asks), the agent's current solution, two previous versions, and `budget_usd` = 1.
   - Verdict: **`keep`**. The next action was "Continue the existing strategy; do not switch yet.
     Score each change with the evaluator and call diverge_escape again if scores stall."
   - Evidence: 12 MEASURED claims and 2 HYPOTHESIZED. The lever on record, unused here, was
     `n_open_bins`, used by 5 of 95 consensus samples.

**What the agent did next.** It had already reached **0.1496 dev** before calling DIVERGE,
against 0.658 for the best previous attempt. That gain is the agent's own work. After the `keep`
verdict, it kept its strategy, saved the solution it had, and stopped. It made no further
improvement. DIVERGE's contribution was a correct "do not switch". It changed nothing in the
outcome; it confirmed the agent was not looping.

**Costs.** The agent itself cost $2.58, as reported by `claude -p`; DIVERGE made 2 model calls,
both on the Anthropic API, for $0.0017 (trace: `results/traces/autonomous-demo-run3/`). Run time was 39 minutes. No human intervened.

## Summary

| run | chose DIVERGE itself | verdict | outcome |
|---|---|---|---|
| 1 | yes | inconclusive (evaluator could not be nested; fixed) | the agent improved on its own |
| 2 | could not (tool allowlist bug; aborted) | none | none |
| 3 | yes | keep (do not switch) | the agent kept its strategy and finished (0.1496 dev, reached before the call) |

In both completed runs the agent chose DIVERGE on its own, from a prompt that never mentions it.
That is n = 2: it shows the behavior happens, not how often.
