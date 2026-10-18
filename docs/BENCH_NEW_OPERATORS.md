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

## Results (interim: generation done for 7 of 8 operator-seed runs; held-out pending)

At the 08:48 cutoff the bench was still generating `blind_spot_directions` seed 1, after which it
labels all candidates and scores each family pick on held-out. The held-out table, useful
discoveries and the bench's own verdict will be in `results/bench-new-ops.log` and
`results/cache_eviction/bench/bench.md` when it exits; the numbers below are read from the
per-seed `run.json` files and are **dev scores only** (lower is better).

| operator | seed | candidates | valid rate | dev best | dev median (valid) | spent $ |
|---|---|---|---|---|---|---|
| plain | 0 | 12 | 1.00 | **0.5665** | 0.5872 | 0.265 |
| plain | 1 | 12 | 1.00 | **0.5675** | 0.5856 | 0.256 |
| blind_spot | 0 | 5 | 1.00 | 0.5903 | 0.5962 | 0.213 |
| blind_spot | 1 | 6 | 1.00 | 0.5869 | 0.5960 | 0.272 |
| blind_spot_hybrid | 0 | 8 | 1.00 | 0.5856 | 0.5960 | 0.253 |
| blind_spot_hybrid | 1 | 8 | 1.00 | 0.5841 | 0.5956 | 0.256 |
| blind_spot_directions | 0 | 7 | 1.00 | 0.5889 | 0.5961 | 0.256 |
| blind_spot_directions | 1 | running at cutoff (2 candidates, $0.103) | | | | |

Invalid candidates by reason: none so far (0 of 70 generated candidates across the 7 finished
runs; no evaluator timeouts despite load average above 100). Useful discoveries and held-out
columns: pending.

Per-family dev picks of the new operators (both seeds): hybrid's best candidates were
LeCaR/CACHEUS-based (0.5856) and CLOCK/SIEVE-based (0.5841); directions' best was CLOCK/SIEVE
with the "state" direction (0.5889). Within each family the hybrid and direction retries still
cluster within about 0.001 of each other on dev (for example LRU-K at 0.5958-0.5963 across all
three blind-spot operators), so the retries are still landing on near-identical programs.

### Verdict (preliminary, 2 repeats, dev only)

- Neither new operator closes the gap to plain sampling. On dev best, `blind_spot_hybrid`
  improved on `blind_spot` in both seeds (0.5856 vs 0.5903; 0.5841 vs 0.5869) and
  `blind_spot_directions` marginally in seed 0 (0.5889 vs 0.5903), but all three blind-spot
  operators sit at 0.584-0.590 while plain reaches 0.5665-0.5675. The gap of about 0.017 miss
  ratio to plain is unchanged in kind; the hybrid allowance recovers roughly a quarter of it.
- The hybrid allowance produced more candidates per dollar (8 vs 5-6) at the same budget without
  any loss of validity.
- Treat this as a direction, not a result: 2 repeats, dev scores, and the controls are fresh
  samples rather than the archived reference run. Held-out scores and useful-family counts
  from the bench's own output are needed before any of this is written up as a finding.

Spend on this step: $2.03 by 08:44 (preflight remaining budget $8.44 before, $6.42 during
labeling), projected about $2.6 after labeling; cap for the step was $3.00.

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

### Final held-out table (run completed 08:45 Pacific; supersedes "held-out pending" above)

Setup: cache_eviction, Claude Sonnet 5.5 implementer via the Anthropic API, $0.30 per operator per
repeat, 2 repeats. Held-out miss ratio, lower is better; values are mean [min, max] across repeats.
LRU scores 0.6747 on held-out.

| operator | candidates | distinct families beating LRU | discoveries per $ (cached) | best held-out |
|---|---|---|---|---|
| plain | 12 | 2 | 7.68 | **0.5672** [0.5666, 0.5678] |
| blind_spot | 5.5 | 3.5 | 14.41 | 0.5887 [0.5867, 0.5908] |
| blind_spot_hybrid | 8 | **4.5** | **17.67** | 0.5843 [0.5838, 0.5848] |
| blind_spot_directions | 7 | 4.0 | 15.09 | 0.5862 [0.5850, 0.5874] |

**Verdict (preliminary, 2 repeats).**
- **Hybrid:** the hybrid-allowed template helps blind-spot steering. It produces the most
  distinct working families per dollar (17.67, against 7.68 for plain) and improves blind_spot's
  best score from 0.5887 to 0.5843.
- **Directions:** per-retry directions help a little.
- **Against plain:** neither closes the gap to plain's best program (0.5672). On this workload,
  steering toward overlooked families buys variety, not a better best program.
