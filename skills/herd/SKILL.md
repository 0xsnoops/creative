---
name: herd
description: Detect whether a coding agent is stuck in a strategy loop (different code, same idea) and get one evidence-backed alternative to test, using HERD. Use when the user or agent asks "am I stuck", "am I trying the same approach again", "is my approach what everyone does", "try something different that works", "my last attempts all score about the same", "should I keep tuning this or switch strategy", or similar, on a task that has an evaluator or a scored history of attempts. Do NOT use for tasks with no way to score a solution and no trajectory of attempts; syntax, API or documentation lookup questions; one-off bugs with a known root cause; or when the user wants a brainstorm list of ideas rather than one tested next step.
---

# HERD: escape a strategy loop

HERD fingerprints the strategy behind each attempt, checks whether the attempts are one
strategy in different code, and compares them with what many models converge on for the task.
It returns a verdict and, when switching is warranted, one experiment with pass/fail thresholds.

## 1. Preflight (always first)

```sh
uv run python -m herd preflight
```

Run it from the HERD repo (or with `uv run --directory <herd repo>`). It prints budget
remaining, whether the API keys are present and whether the configured models are available.
If it exits non-zero, or reports a configured model as unavailable, **stop**: tell the user
what failed (budget under $1, missing key, unavailable model) and do not call any other HERD
tool.

## 2. Pick the call

Gather: the task description, the evaluator command (`<cmd> <candidate.py>` printing
`{"valid": bool, "score": number}` as its last JSON line; absolute paths) or a registered task
name (`online_bin_packing`, `cache_eviction`), and whether lower scores are better.

- **Trajectory available** (current solution plus 1-3 earlier attempts) **and an evaluator**:
  call the `herd_escape` MCP tool with `task_description`, `current_solution` (code),
  `previous_attempts` (code, oldest first), `evaluator_cmd`, `lower_is_better`, and a
  `budget_usd` the user is comfortable with (default 2). Without MCP:

  ```sh
  uv run python -m herd escape --task <name or task file> --candidate <current.py> \
      --history <dir of earlier attempts, sorted oldest first> --evaluator "<cmd>" [--higher-is-better]
  ```

- **No trajectory yet** (choosing an approach before implementing): call `herd_check` with
  `task_description`, `my_approach` (one or two sentences: the mechanism you plan) and
  `evaluator_cmd` if there is one.

- **Neither an evaluator nor a trajectory**: this skill does not apply. Say so.

## 3. Act on the verdict

`herd_escape` output starts with `VERDICT: <verdict>` followed by JSON with `lever`,
`next_experiment` (`hypothesis`, `success_threshold`, `failure_threshold`), `evidence` and
`scope_note`.

| VERDICT | what to do |
|---|---|
| `keep` | Not a loop worth escaping (for example, scores are still improving). Continue the current strategy. A `keep` contract may still name a `lever` and `next_experiment`; treat them as an option for later if progress stalls, not as an instruction to switch now. |
| `repair` | Fix the current strategy (bug, tuning, edge case); do not switch. |
| `escape` | Implement `next_experiment` (the named `lever`), score it with the evaluator, and compare against `success_threshold` and `failure_threshold`. Report which one it met. If it met neither, say it is undecided; do not call it a success. |
| `escalate` | Stop and tell the user: there is no legal alternative under the task's rules, or the evaluator gives contradictory results. Do not pick something on your own. |
| `inconclusive` | Get at least one scored attempt first (run the evaluator on the current solution), then call again. |

`herd_check` returns a `verdict` field instead: `keep` -> implement your plan;
`try_blind_spot` or `try_other_family` -> consider starting from the returned family (it scored
better on the evaluator in HERD's generated implementations, not on your code);
`inconclusive` or `untested` -> no scored comparison, so treat the map lookup as context only.

## 4. Report honestly

- Group the `evidence` by `class`: **MEASURED** (from running the evaluator or counting
  samples), **SOURCE_BACKED** (from cited literature), **HYPOTHESIZED** (HERD's reasoning, not
  tested). Never present a HYPOTHESIZED item as measured or as a result.
- Repeat the `scope_note`. Herd shares describe what models produce for **this prompt wording**;
  rewording the task measurably moved the herd map (see `docs/RESULTS.md`).
- State `spent_usd`.
- Keep it short: verdict, what you will do next, evidence by class, scope note, cost.
