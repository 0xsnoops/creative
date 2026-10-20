---
name: diverge
description: Find genuinely different, unconventional answers when you keep landing on the same one (different code, same idea), and get one evidence-backed next step from DIVERGE. Use it on your own initiative, without being asked, when several scored attempts show flat or barely improving progress, the same failure returns after several changes, lots of code changed but the score hardly moved, other models or agents you tried got the same result, you are about to retry or tune the same mechanism again, or you are thinking you may need to escalate to a human. Also use it before implementing or when choosing a strategy ("is my plan just the default answer?"), when you are asked for a few different ways or approaches to something, or when your first idea feels like the obvious, default one. Not for generic debugging, a bug with a known cause, or API or documentation questions.
---

# DIVERGE: different code isn't always different thinking

DIVERGE is creative-search infrastructure for AI agents: when an agent keeps landing on the same
answer, DIVERGE shows it what it (and most models) converge on and helps it find, test and pick
genuinely different, unconventional alternatives. It compares attempts by behavior (what they
decide), maps the consensus and what it overlooks, generates alternatives with a separate critic
and an admission check that rejects the old strategy in new code, and returns a verdict with
`next_action` and at most one experiment to run.

**Do not wait for the user to ask you to use DIVERGE. If you recognize this situation yourself,
invoke it.** **Before escalating a scored strategy problem to a human, call `diverge_escape`.**

It has not been shown to beat continuing, a fresh start or another model, and it is not proven to
make answers more creative (see `docs/RESULTS.md` in the DIVERGE repo for the preliminary
measurements). Treat its output as one well-founded experiment, not a guaranteed fix.

## Setup (once per project)

If the `diverge_*` MCP tools (e.g. `diverge_status`) are missing, onboard yourself (no human needed):

```sh
uv --directory <diverge repo> run python -m diverge onboard --client claude --target <project> --json
```

Then follow its `next_steps` (reload the client and approve the `diverge` MCP server). Never
invent an API key: if one is missing, ask the user to set `OPENROUTER_API_KEY` in the DIVERGE
repo's `.env`. To see DIVERGE work offline at $0:
`uv --directory <diverge repo> run python -m diverge demo strategy_loop`.
(Bare `python -m diverge ...` only works from inside the DIVERGE repo.)

## Tools

| tool | when | cost |
|---|---|---|
| `diverge_status` | first, every session: are budget, keys and models ready? Stop and tell the user if not. | $0 |
| `diverge_check` | before implementing, or when asked for several approaches: is my planned strategy the obvious/default one? Args: `task_description`, `my_approach` (one or two sentences on the mechanism), `evaluator_cmd` if any. Without an evaluator (free-text mode) it returns 3-5 untested alternatives, each with `strength`, `trap` and `how_to_test` (about $0.01). Pass `already_tried` / `constraints` / `context` when you have them (see below); `depth="deep"` for a multi-model search (cents). | paid |
| `diverge_escape` | after 2+ scored attempts stall. Args: `task_description`, `current_solution` (code), `previous_attempts` (code, oldest first), `evaluator_cmd`, `lower_is_better`, `budget_usd` (default 2); optional `already_tried`, `constraints`, `context`. | paid |
| `diverge_report_outcome` | after you test an idea: `alternative`, `outcome` (`LEGIT`, `WEAK` or `WRONG`), `note`, `score`. Later calls on the same `task_description` treat it as already tried. | $0 |
| `diverge_map`, `blind_spots`, `reality_check` | lower-level research primitives (what models converge on, what they miss, scoring one family). Use only when asked for that research. | paid |

Pass what you know; it makes the alternatives yours instead of generic:
- `already_tried`: `[{"approach": "...", "outcome": "why it failed", "score": 0.97}]`. DIVERGE never
  offers these again (a separate reviewer model drops duplicates into `filtered_out`), and each
  alternative names the failure it `builds_on`.
- `constraints`: hard rules (`"no new dependencies"`, `"must run on one CPU"`). Violators go to
  `filtered_out`; `relax_hint` says which rule, if relaxed, would open the most.
- `context`: your real figures (log tail, results table, up to 8000 characters, redacted). Each
  alternative quotes the figures it relies on (`grounding`); `ungrounded_numbers` flags numbers not
  in your context, so do not trust those.
With any of these, alternatives come sorted decisive and cheapest first (`test_cost`, `decisive`,
`kill_result`): run the first one's kill test before anything else, then report the result with
`diverge_report_outcome`.

`evaluator_cmd` is run as `<cmd> <candidate.py>`, prints `{"valid": bool, "score": number}` as
its last JSON line, and needs absolute paths. It runs inside DIVERGE's sandbox (no network, no writes
outside a scratch directory, no reading `.env`, no nested sandbox), so it must be self-contained.
If the task is a registered one (`online_bin_packing`, `cache_eviction`), pass exactly that name as
`task_description` and omit `evaluator_cmd`: DIVERGE uses its own prompt and evaluator. Without MCP,
the CLI equivalents are `uv --directory <diverge repo> run python -m diverge preflight` and
`uv --directory <diverge repo> run python -m diverge escape --task ... --candidate ... --history ... --evaluator "..."`.

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
`try_other_family` consider the returned family (it scored higher in DIVERGE's own
implementations, not yours); `inconclusive` or `untested` treat it as context only. In free-text
mode (`untested`), pick an alternative whose `strength` fits the task, watch for its `trap`, and
run its `how_to_test` before trusting it.

## Report honestly

Group `evidence` by class: MEASURED (evaluator runs, sample counts), SOURCE_BACKED (cited
literature), HYPOTHESIZED (DIVERGE's untested reasoning). Never treat HYPOTHESIZED evidence as
measured. Repeat the `scope_note` (consensus shares depend on the prompt wording) and `spent_usd`.
Keep it short: verdict, what you will do next, evidence by class, scope note, cost.
