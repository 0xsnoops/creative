# Phase 2 live-experiment protocol (written before any live call)

Branch `model-review-changes`. Implementer `google/gemini-4-argon` for every operator.
Labeler `anthropic/claude-opus-5.5`. Every call is traced (`DIVERGE_TRACE_DIR=results/traces/phase2-20261018`).

## Spending
- Hard cap: DIVERGE `budget_usd: 25.0` (counts everything in DIVERGE's cache; this experiment starts
  from an empty experiment cache).
- I stop and ask if the two maps together cost more than $10, or if total spend is heading past $25.

## Step 0: smoke test
`diverge run cache_eviction --n 1 --no-reality`. This checks the live request formats. Its sampler
responses are reused by step 1 (same cache keys).

## Step 1: consensus maps
`diverge run <task> --n 20 --no-reality` for cache_eviction, then online_bin_packing.
Then run `diverge fingerprints online_bin_packing` on the map samples ($0).

## Step 2: budgets, fixed from map costs only
- `remaining = 25 - total spent after step 1 - 1.00` (the $1.00 is held back for overruns).
- 65% of `remaining` goes to cache_eviction and 35% to online_bin_packing.
- Per task: `budget per operator = (task share - labeling allowance) / (operators x 3 repeats)`,
  rounded down to a cent.
- `labeling allowance = 0.10 USD x expected candidate batches`, where expected candidates are
  8 per operator-run.
- `bench.max_candidates` is set to 12.
- The implementer's `max_tokens` stays at 16000. If Argon's sampler completions on these tasks
  exceed 16000 tokens at the 95th percentile, I'll note it and still keep 16000.

## Step 3: benchmarks (3 repeats each, generation seeds 0-2)
- **cache_eviction** (9 operators): plain, adhd_frames, blind_spot, blind_spot_contrast,
  blind_spot_contrast_random, blind_spot_feedback, blind_spot_feedback_sham, literature_all,
  literature_random.
- **online_bin_packing** (5 operators): blind_spot, blind_spot_contrast, blind_spot_feedback,
  adhd_frames, literature_random.

## Primary comparisons, decided now
- **Is the thesis supported?** blind_spot vs literature_random (count-matched) and vs
  literature_all, on best held-out score and on cold-start discoveries per dollar.
- **Does contrast help?** blind_spot_contrast vs blind_spot and vs blind_spot_contrast_random,
  on best held-out score, purity, and pure-candidate held-out score.
- **Does feedback help?** blind_spot_feedback vs blind_spot and vs blind_spot_feedback_sham, on
  valid rate and best held-out score.
- **ADHD:** adhd_frames vs plain and vs blind_spot, on best held-out score and discoveries per dollar.
- **Measurement only:** behavior clusters for bin packing.

## Reading rules
- "Did better in every repeat" means the ranges across the 3 repeats don't overlap. Anything
  else is reported as no clear difference.
- Losses and inconclusive results are reported as they come out.
- Evaluators, seeds and instances don't change after this point.
- Held-out scores are never shown to generation.
