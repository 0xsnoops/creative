# External review panel (2026-10-20) and what changed

Six non-Anthropic models each reviewed the same repo digest (README, agent instructions, results, core
code) from one role. Five returned full reviews; the sixth (GLM, systems/security) returned an empty
reply, and the voting round produced one usable ballot, so there is no panel ranking. Cost $1.60 on
OpenRouter. Every claim below that led to a change was checked against the code and results first.

| role | model |
|---|---|
| research scientist (creative search) | deepseek/deepseek-v4-pro-0813 |
| skeptical statistician | moonshotai/kimi-k3 |
| agent-tool product engineer | qwen/qwen3.8-max-prime |
| evolutionary-search practitioner | x-ai/grok-4.7 |
| wild-card inventor | minimax/minimax-m3 |

## What all five agreed on (checked)

- The chosen lever has not beaten a random one: rescue round 2 rescued 3/4 stuck states with a random
  lever and 1/4 with escape v2's chosen lever, and state 0 (stuck at 148.12) was rescued by every arm.
- Off bin packing, the loop decision used LLM-description strategy similarity, which E1b measured at
  chance (AUC 0.50-0.53).
- With exactly 2 attempts the behavior detector could never report a loop (`RECENT = 3`), though the
  docs said 2 attempts were enough.

Found while checking:

- The MCP tool and CLI ran escape v1, not v2, the version the rescue tests measured.
- Escape v2's admission check raised on every task without probes (cache_eviction, any evaluator_cmd),
  so v2 rejected every candidate there.

## What changed

- **Loop detection** (`diverge/trajectory.py`, `diverge/diagnose.py`): strategy similarity is a caption
  only. Tasks without probes use shared worst dev cases ("failures" detector: same worst cases and
  stalled = switch; stalled only = repair). With no per-case results the diagnosis is `undetermined`
  and the verdict `inconclusive`, with a next_action that says how to add them. 2 attempts can now be
  a behavior loop.
- **Per-case results for any evaluator** (`diverge/failures.py`): `detail.cases [{"id", "loss"}]`;
  `examples/replay_evaluator.py` prints them.
- **Escalate rule B** (`diverge/contract.py`): identical scores escalate only when the behavior detector
  measured different behavior, never on strategy similarity.
- **Escape v2 in the tool** (`diverge/server.py`): `escape_for_agent` runs `escape_v2` (k=3 sketches by
  default, as in the rescue tests) and generates nothing unless the diagnosis is switch_strategy.
  Admission passes through (and says so) on tasks without probes.
- **`lever_policy`**: `chosen` (default, unchanged), `random` (an input at least half the consensus
  uses, as the control arm drew), `mixed` (chosen, then random if the chosen candidate did not improve).
- **New tasks**: with no overlook study the lever is an input none of the attempts use, else the top
  grounded alternative to the attempts' shared strategy (one model call, HYPOTHESIZED), instead of
  `escalate`.
- **`diverge_loop_check`**: free triage from scores and each attempt's failing cases, same rules.

## Not done (still open)

- The decisive experiment: chosen vs random lever on about 24 stuck states, trivial states excluded
  (all reviewers' first priority). It needs a paid run; the default stays `chosen` until then.
- A patch operator that edits the agent's best code, and admission as a logged descriptor instead of
  a gate (evolutionary-search reviewer).
- Re-analysis of the operator benchmark at equal candidate counts against the strongest baseline, with
  cold-start cost (statistician). The raw results are not in this checkout.
- A planted-mechanism positive-control task (statistician's wild card).
- A `test_cmd` adapter that turns pytest or npm test output into an evaluator result.

Claims checked and rejected: "RecordingEvaluator drops scores of identical attempts" (identical code
gets the identical score, which is correct); "OVERLOOKED_SHARE 0.10 vs blind_spot_max_share 0.02 is a
mismatch" (one is an input's share, the other a family's).
