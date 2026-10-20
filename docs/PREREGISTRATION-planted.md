# Preregistration: planted-answer positive control (frozen 2026-10-20, before any paid run)

Question: when the right mechanism is known by construction, does DIVERGE's escape find it from a
stuck state? Every earlier test was observational: nobody knew which lever "should" help. This one
has a ground truth. It tests the path a new task takes (no consensus map), not the consensus-map
levers.

## Protocol (scripts/planted_control.py, as committed with this file; run once)

- Task: predict a relay's delivery delay (ms) from size_kb, queue_len, hour, link_id, retries. Truth:
  18 + 1.5*queue_len + 0.1*size_kb + 25*retries, plus a fixed extra delay on links 7 (+160), 19 (+130)
  and 33 (+90), plus Gaussian noise (sd 2). hour has no effect (a decoy). The prompt shows 40 example
  records, two from each slow link. Dev and held-out: 600 records each, fixed seeds.
- Evaluator: scripts/planted/eval_link.py. Score = mean absolute error; detail.cases = one case per link
  (loss = that link's MAE).
- Stuck states: 5, each three human-written attempts (plain, rounded constants, weight-table loop) of
  one least-squares fit on numeric features that ignores link_id. Feature sets differ per state.
- Arms per state, implementer deepseek/deepseek-v4-pro-0813 everywhere, K = 3 candidates, $0.40 cap per
  arm, $5 total cap:
  - diverge: escape_for_agent with lever_policy "chosen";
  - random: escape_for_agent with lever_policy "random";
  - fresh_start: task prompt + failure summary (scores, two worst links), no code;
  - patch: escape_patch on the best attempt.
- Other roles: extractor qwen/qwen3.8-flash, labeler (screen/critic) mistralai/mistral-large-4-0. No
  Anthropic model.
- Selection on dev only; held-out scored once per arm.

## Outcome

- "Found M" for an arm on a state: held-out MAE on slow-link records at most 20% of the stuck best's,
  and held-out MAE below the stuck best's.
- Primary: the diverge arm finds M in at least 3 of 5 states.
- Reported for every arm: found-M count, held-out MAE, the lever chosen, cost.
- If fresh_start also finds M in most states, the task is too easy to separate the arms; that is
  reported as such, not fixed and rerun. Any change after seeing results is a new preregistration.

## Amendment 1 (2026-10-20, before any outcome data)

The first launch was stopped after 4 model calls ($0.012), none of them an arm's result. The
extractor qwen/qwen3.8-flash spent its whole 16,000-token allowance on hidden reasoning and returned
no answer to the alternatives prompt (186 s); it would have retried and then given "no lever", a
plumbing failure rather than a finding. Changes, made before any arm produced a candidate:
- extractor: meta-llama/llama-4-maverick (no hidden reasoning; tested on the same prompt);
- states run in parallel (arms within a state stay sequential);
- per-arm cap $0.25 (was $0.40), so 5 states x 4 arms is a hard $5 bound in parallel.
Everything else, including the outcome definitions, is unchanged.

## Round 2 (new preregistration, 2026-10-20, after round 1's results)

Round 1 (docs/RESULTS.md): the diverge arm found M in 1/5 states. Every lever came from the no-map
fallback, which proposed the stuck approach itself ("LinearModel") in 4 of 5 states because the attempts'
shared strategy was not passed to it as already tried, and lever_policy "random" repeated diverge's lever.
The fix (diverge/server.py fallback_lever): the shared strategy goes in as an already-tried entry (so the
screen drops restatements), and "random" draws one of the screened alternatives.

Round 2 runs `scripts/planted_control.py --round 2` once, with the same task, data, stuck states, models,
caps and outcome definitions. Only the code under test changed. fresh_start and patch send the same
prompts as round 1, so their results are round 1's cached draws, which makes the comparison paired.
Primary: the diverge arm finds M in at least 3 of 5 states. Both rounds are reported, whatever round 2
shows. No round 3 of this task will be run to chase a better number.
