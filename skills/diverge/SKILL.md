---
name: diverge
description: Spot a strategy loop on a scored task (different code, same idea) and get one evidence-backed next step from DIVERGE. Use it on your own initiative, without being asked, when several scored attempts show flat or barely improving progress, the same failure returns after several changes, lots of code changed but the score hardly moved, other models or agents you tried got the same result, you are about to retry or tune the same mechanism again, or you are thinking you may need to escalate to a human. Also use it before implementing, to check whether your planned strategy is just the default most models pick. Not for generic debugging, a bug with a known cause, API or documentation questions, or tasks with no evaluator and no scored attempts.
---

# DIVERGE: different code isn't always different thinking

DIVERGE fingerprints the strategy behind your attempts, tells you whether they are one idea in
different code, compares them with what many models converge on for the task, and returns a
verdict with at most one experiment to run.

**Do not wait for the user to ask you to use DIVERGE. If you recognize this situation yourself,
invoke it.** **Before escalating a scored strategy problem to a human, call `diverge_escape`.**

DIVERGE is for strategy stagnation on a task with a score, not for generic debugging. It has not
been shown to beat continuing, a fresh start or another model (see `docs/RESULTS.md` in the
DIVERGE repo); treat its output as one well-founded experiment, not a guaranteed fix.

## Setup (once per project)

If the `diverge_*` MCP tools (e.g. `diverge_status`) are missing, onboard yourself from the DIVERGE repo (no human needed):

```sh
uv --directory <diverge repo> run python -m diverge onboard --client claude --target <project> --json
```

Then follow its `next_steps` (reload the client and approve the `diverge` MCP server). Never
invent an API key: if one is missing, ask the user to set `OPENROUTER_API_KEY` in the DIVERGE
repo's `.env`. To see DIVERGE work offline at $0: `uv --directory <diverge repo> run python -m diverge demo strategy_loop`.

## Tools

| tool | when | cost |
|---|---|---|
| `diverge_status` | first, every session: are budget, keys and models ready? Stop and tell the user if not. | $0 |
| `diverge_check` | before implementing: is my planned strategy the obvious/default one? Args: `task_description`, `my_approach` (one or two sentences on the mechanism), `evaluator_cmd` if any. | paid |
| `diverge_escape` | after 2+ scored attempts stall. Args: `task_description`, `current_solution` (code), `previous_attempts` (code, oldest first), `evaluator_cmd`, `lower_is_better`, `budget_usd` (default 2). | paid |
| `diverge_map`, `blind_spots`, `reality_check` | lower-level research primitives (what models converge on, what they miss, scoring one family). Use only when asked for that research. | paid |

`evaluator_cmd` is run as `<cmd> <candidate.py>`, prints `{"valid": bool, "score": number}` as
its last JSON line, and needs absolute paths. It runs inside DIVERGE's sandbox (no network, no writes
outside a scratch directory, no reading `.env`, no nested sandbox), so it must be self-contained.
If the task is a registered one (`online_bin_packing`, `cache_eviction`), pass exactly that name as
`task_description` and omit `evaluator_cmd`: DIVERGE uses its own prompt and evaluator. Without MCP, the CLI equivalents are
`python -m diverge preflight` and `python -m diverge escape --task ... --candidate ... --history ... --evaluator "..."`.

## Act on the verdict

`diverge_escape` returns `VERDICT: <verdict>` (lowercase) then JSON with `lever`, `next_experiment`
(`hypothesis`, `success_threshold`, `failure_threshold`), `evidence` and `scope_note`.

| VERDICT | do this |
|---|---|
| `keep` | Continue; do not switch yet. A named lever is an option for later, not an instruction. |
| `repair` | Fix the existing strategy (bug, tuning, edge case) and preserve its mechanism. |
| `escape` | Implement the named `lever`, run the evaluator, compare against the before score and the thresholds. Report which threshold it met; if neither, say undecided, not success. |
| `escalate` | Stop autonomous retries and ask a human (no legal alternative, or a contradictory evaluator). |
| `inconclusive` | Get more scored attempts (run the evaluator), then call again. |

`diverge_check` returns a `verdict` field: `keep` implement your plan; `try_blind_spot` or
`try_other_family` consider the returned family (it scored better in DIVERGE's own
implementations, not yours); `inconclusive` or `untested` treat it as context only.

## Report honestly

Group `evidence` by class: MEASURED (evaluator runs, sample counts), SOURCE_BACKED (cited
literature), HYPOTHESIZED (DIVERGE's untested reasoning). Never treat HYPOTHESIZED evidence as
measured. Repeat the `scope_note` (consensus shares depend on the prompt wording) and `spent_usd`.
Keep it short: verdict, what you will do next, evidence by class, scope note, cost.
