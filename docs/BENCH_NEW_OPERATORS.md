# Benchmark: do the new blind-spot operators close the gap to plain sampling?

Date: 2026-10-18. Task: `cache_eviction` (average miss ratio over held-out traces; lower is better).
Status: preliminary, 2 repeats.

## Question

In the Anthropic 3-operator run below, `blind_spot` found more useful families than `plain`
(4 vs 2 per repeat) but lost on the number that matters to a user, the best held-out score
(0.5877 vs 0.5681). Two reasons were suspected: the shared "implement this family faithfully"
template blocks hybrids (plain's best policy was a hybrid of two mechanisms), and `blind_spot`'s
retries resample the same prompt. Two operators test those suspicions one at a time:

- `blind_spot_hybrid`: the same blind-spot families in the same round-robin order as `blind_spot`,
  but implemented with `HYBRID_TEMPLATE` (the family is the core mechanism; borrowing from other
  mechanisms is allowed). `diverge/operators/blind_spot_hybrid.py`.
- `blind_spot_directions`: the same families and schedule as `blind_spot`, but each retry of a
  family appends a different fixed design direction to the prompt instead of resampling it.
  `diverge/operators/blind_spot_directions.py`.

`plain` and `blind_spot` are the matched controls.

## Setup

- Provider: the Anthropic API (`DIVERGE_PROVIDER=anthropic`; the key is loaded by DIVERGE from
  `DIVERGE_ENV_FILE`). OpenRouter was disabled. Under this provider DIVERGE maps every single-model
  role to a Claude model (`diverge/llm.py`, `anthropic_only_models`): implementer
  `anthropic/claude-sonnet-5.5` (API id `claude-sonnet-5-5`), extractor
  `anthropic/claude-haiku-5.5`, labeler `anthropic/claude-opus-5.5`.
- Consensus map: the cached cache_eviction map (`--n 20`, 20 samples per model, 5 sampler models),
  not rebuilt; the samplers list is left as configured so the cached map is reused. The blind
  spots that `blind_spot*` implement come from that map.
- Budget: $0.30 per operator per repeat, at most 12 candidates per operator per repeat
  (`config.yaml` `bench.max_candidates`). Repeats: 2, generation seeds 0 and 1 (the seeds vary
  generation only, not the evaluation data).
- Selection on dev; each family's dev-selected candidate is scored once on held-out, and every
  score and "useful" decision below is a held-out number. Held-out baselines: LRU 0.6747,
  FIFO 0.7088. "Useful" = a distinct labeled family whose held-out score beats LRU.
- `plain` and `blind_spot` were expected to replay from the LLM cache of the 3-operator run (same
  seeds, same budget) at no cost. They did not: their `run.json` files were rewritten during this
  run with different candidates and spend (plain seed 0: 12 candidates, $0.265, dev best 0.5665
  here vs $0.263, 0.5687 archived; blind_spot seed 0: 5 candidates, $0.213 vs 6, $0.256). So the
  two controls in this run are fresh samples at the same seeds, not copies of the reference run,
  and they were billed. Why the cache keys differed is not yet diagnosed.
- Command (run once; not rerun to chase a number):
  `python -u -m diverge bench cache_eviction --budget 0.3 --repeats 2 --n 20 --operators plain,blind_spot,blind_spot_hybrid,blind_spot_directions`
  Log: `results/bench-new-ops.log`; outputs: `results/cache_eviction/bench/`; traces:
  `results/traces/bench-new-ops-20261018/`.
- Machine load during the run: load average above 100 (other jobs). The evaluator has a 600 s
  timeout per candidate; under load a slow candidate can be marked invalid ("timeout after 600s").
  Invalid candidates are counted by reason below so the valid rate can be read with that in mind.

## Results (held-out, 2 repeats)

Cells are mean [min, max] across the 2 repeats unless a single number. Spend is the operator's
own calls per repeat; labeling the 64 valid candidates cost $0.471 and is billed to no operator.
Lower miss ratio is better. Bench outputs: `results/cache_eviction/bench/` (bench.md, bench.json,
candidates.jsonl, bench.png), log `results/bench-new-ops.log`.

| operator | candidates | valid rate | useful discoveries | best held-out | median held-out of family picks | median dev score (valid) | spent $ |
|---|---|---|---|---|---|---|---|
| plain | 12.0 [12.0, 12.0] | 1.00 | 2.0 [2.0, 2.0] | **0.5672** [0.5666, 0.5678] | 0.5760 | 0.5856 | 0.260 [0.256, 0.265] |
| blind_spot | 5.5 [5.0, 6.0] | 1.00 | 3.5 [3.0, 4.0] | 0.5887 [0.5867, 0.5908] | 0.5930 | 0.5962 | 0.242 [0.213, 0.272] |
| blind_spot_hybrid | 8.0 [8.0, 8.0] | 1.00 | **4.5** [4.0, 5.0] | 0.5843 [0.5838, 0.5848] | 0.5941 | 0.5960 | 0.255 [0.253, 0.256] |
| blind_spot_directions | 7.0 [7.0, 7.0] | 0.93 [0.86, 1.00] | 4.0 [4.0, 4.0] | 0.5862 [0.5850, 0.5874] | 0.5944 | 0.5958 | 0.265 [0.256, 0.274] |

Held-out baselines: LRU 0.6747, FIFO 0.7088. Discoveries per $ (cached): plain 7.68,
blind_spot 14.41, blind_spot_hybrid 17.67, blind_spot_directions 15.09.

Per repeat (best held-out; useful families):

- plain: s0 0.5666, s1 0.5678; LFU and 2Q/S3-FIFO in both repeats.
- blind_spot: s0 0.5908 (LRU-K, CLOCK/SIEVE, LeCaR/CACHEUS), s1 0.5867 (LRU-K, LeCaR/CACHEUS,
  LIRS/CLOCK-Pro, 2Q/S3-FIFO).
- blind_spot_hybrid: s0 0.5848 (LRU-K, LeCaR/CACHEUS, LIRS/CLOCK-Pro, 2Q/S3-FIFO), s1 0.5838
  (those four plus CLOCK/SIEVE).
- blind_spot_directions: s0 0.5874, s1 0.5850 (LRU-K, LeCaR/CACHEUS, LIRS/CLOCK-Pro, 2Q/S3-FIFO
  in both).

### Invalid candidates by reason

| operator | generated | invalid | reasons |
|---|---|---|---|
| plain | 24 | 0 | |
| blind_spot | 11 | 0 | |
| blind_spot_hybrid | 16 | 0 | |
| blind_spot_directions | 14 | 1 | no code in reply x1 (seed 1) |

1 of 65 invalid in total. No evaluator timeouts occurred despite a load average above 100 during
the run, so the valid rates are not depressed by machine load; the one failure is the
implementer replying without a code block.

### Verdict (preliminary, 2 repeats)

- **No. Neither new operator closes the gap to plain sampling on the best held-out score.**
  plain reached 0.5666-0.5678 in both repeats; the best blind-spot operator,
  `blind_spot_hybrid`, reached 0.5838-0.5848. The ranges do not overlap in either direction
  between plain and any blind-spot operator, so with 2 repeats plain is ahead of all three.
- **The hybrid allowance helps, a little, on every measure.** `blind_spot_hybrid` beat `blind_spot`
  in both repeats on best held-out (0.5848 vs 0.5908; 0.5838 vs 0.5867), on candidates per budget
  (8 vs 5-6), on useful families (4 and 5 vs 3 and 4) and on discoveries per $ (17.7 vs 14.4). It
  recovers about 0.004 of the roughly 0.021 gap to plain: the template was part of the problem,
  not most of it.
- **The directions lever did less.** `blind_spot_directions` improved on `blind_spot` in both
  repeats on best held-out (0.5874 vs 0.5908; 0.5850 vs 0.5867) and matched it on useful
  families, but its within-family retries still land within about 0.001 of each other on dev, so
  a text direction does not by itself produce a different program.
- The blind-spot operators keep their advantage in breadth: 3.5-4.5 distinct useful families per
  repeat versus plain's 2, at 2x the discoveries per dollar. plain's edge is a single family
  (an LFU/2Q-style policy) that is simply better on this task than anything in the blind spots.
- Why it is preliminary: 2 repeats with the same evaluation data; the controls here are fresh
  samples, not the archived reference run (see Setup); the dev/held-out split is fixed; and
  `blind_spot`'s best is ~0.004 worse here than in the reference run, which is the scale of the
  effects being claimed. A third repeat and the directions-plus-hybrid operator
  (`blind_spot_directions_hybrid`) are the obvious next measurements.

Spend on this step: $2.61 (preflight remaining budget $8.44 before, $5.84 after), of which
roughly $1.0 was the two controls regenerating, $1.05 the two new operators and $0.47 labeling.
Cap for the step was $3.00. The bench was run once and not rerun.

## Reference: the 3-operator Anthropic run (same setup, same seeds)

Outputs archived in `results/cache_eviction/bench-anthropic-3ops/`; log `results/bench-anthropic.log`.
Cells are mean [min, max] across the 2 repeats. Spend is the operator's own calls (cached
cost); labeling the candidates cost $0.378 and is billed to no operator.

| operator | candidates | valid rate | useful discoveries | best held-out | median held-out of family picks | median dev score (valid) | spent $ |
|---|---|---|---|---|---|---|---|
| plain | 12.0 [12.0, 12.0] | 1.00 | 2.0 [2.0, 2.0] | **0.5681** [0.5674, 0.5687] | 0.5764 | 0.5862 | 0.261 [0.259, 0.263] |
| literature_random | 10.0 [9.0, 11.0] | 1.00 | 4.0 [4.0, 4.0] | **0.5680** [0.5673, 0.5687] | 0.5893 | 0.5935 | 0.272 [0.263, 0.280] |
| blind_spot | 6.0 [6.0, 6.0] | 1.00 | 4.0 [4.0, 4.0] | 0.5877 [0.5850, 0.5904] | 0.5932 | 0.5957 | 0.259 [0.256, 0.261] |

Invalid candidates: none (0 of 56). The bench's own verdict: highest useful-discovery rate
(cached) blind_spot, 15.46 per $, ahead of literature_random (14.74) in both repeats; best
held-out score literature_random 0.5680 and plain 0.5681, with blind_spot behind at 0.5877 in
both repeats. Per repeat, blind_spot's useful families were LRU-K, LeCaR/CACHEUS, LIRS/CLOCK-Pro
and 2Q/S3-FIFO (seed 0) and LRU-K, CLOCK/SIEVE, LeCaR/CACHEUS, LIRS/CLOCK-Pro (seed 1); plain's
were LFU and 2Q/S3-FIFO in both repeats.

## Column definitions

- candidates: generated per repeat (valid or not). valid rate: candidates that ran and scored on dev.
- useful discoveries: distinct labeled families whose dev-selected pick beats LRU on held-out.
- best held-out: the held-out score of the overall dev-best candidate (the number a user would get).
- median held-out of family picks: median over both repeats of the family picks' held-out scores
  (how good a typical discovered family is, not just the best one).
- median dev score (valid): median dev score over all valid candidates of the operator, both repeats
  (typical candidate quality before selection).
- spent $: the operator's own LLM calls per repeat, cached cost.
