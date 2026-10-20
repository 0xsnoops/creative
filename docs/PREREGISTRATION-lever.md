# Preregistration: chosen lever vs random lever, round 3 (frozen 2026-10-20, before any paid run)

Question: does the lever DIVERGE chooses (from the consensus map and overlook study) rescue stuck
agents better than a random lever? Round 2 (n = 4, one trivially rescuable state) had the random
control ahead, 3/4 vs 1/4. All five external reviewers named this the first thing to settle.

## Protocol (scripts/lever_test.py, as committed with this file; run once)

- Task: online_bin_packing. A fresh consensus map (20 samples from each of qwen3.8-max, gpt-6.1-sol,
  gemini-4-argon, llama-4-maverick, deepseek-v4-pro; labeler gpt-6.1-sol) and overlook study (extractor
  qwen3.8-flash), built once before any trajectory. No Anthropic model anywhere. This map differs from the
  recorded one, so round 3 is not a replication of round 2's exact levers.
- Trajectories: scripts/rescue_test.py's protocol (6 attempts; revisions see the previous code and its dev
  score), agent and implementer deepseek/deepseek-v4.1-flash, in batches of 20 until at least 24 eligible
  stuck states exist or 80 trajectories were run.
- Eligible state: stuck by rescue_test.stuck (the last 3 attempts are no more than 1% better than the first
  3), and the stuck best's held-out score is no worse than the first-fit baseline's held-out score. States
  stuck far above a trivial baseline are rescued by every arm and are excluded before any arm runs.
- Arms per eligible state (same implementer, K = 3, $0.40 cap each):
  - chosen: escape v2 with DIVERGE's lever (choose(): overlooked input, assumption, minority, blind spot);
  - random_lever: escape v2 with a random input that at least half the consensus uses (round 2's control);
  - fresh_start: rescue_test's fresh prompt with a failure summary, no code;
  - patch: escape_patch on the best attempt (new, exploratory).
  Escape arms run from the stuck state whatever the diagnosis, as in round 2.
- Selection on dev only. Held-out once per arm (its dev-best candidate) and once for the stuck best.

## Outcome and decision

- Rescue: held-out score below the stuck best's held-out score.
- Primary: paired held-out scores per eligible state, chosen vs random_lever (an arm with no valid
  candidate counts as worse than one with a candidate; equal or both missing is a tie). Two-sided exact
  sign test on the non-tied states.
- Decision for diverge_escape's default lever_policy:
  - random_lever better with p < 0.05: change the default to "random";
  - chosen better with p < 0.05: keep "chosen" and report that it beat a random lever in this test;
  - otherwise: keep "chosen" and state that the lever choice has not been shown to matter.
- Reported for every arm: rescued count, states with no candidate, cost. patch and fresh_start are
  descriptive (no decision rides on them).
- Fewer than 24 eligible states after 80 trajectories: run the arms on those that exist and report the
  result as preliminary. Any change after seeing results is a new preregistration.

## Amendment 1 (2026-10-20, before any paid run of this script)

Before launch, the planted-answer control showed that qwen/qwen3.8-flash can spend its whole token
allowance on hidden reasoning and return no answer. The extractor role (judges, overlook reads) is
therefore meta-llama/llama-4-maverick, which does not reason. Trajectories of a batch and eligible states
run in parallel (8 workers; each trajectory and each state's arms stay sequential). Nothing else changes.
