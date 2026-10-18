# Bench forensics: why blind_spot / literature_* lose to plain on cache_eviction

Offline analysis of files on disk (no model calls), 2026-10-18. Scripts under
`scratchpad/forensics/` (`per_operator.py`, `traces.py`, `code_features.py`, `selection.py`).
Scores are dev miss ratios unless marked held-out; lower is better. Dev LRU = 0.6752, held-out LRU = 0.6747.

Two data layers exist and disagree in size, so each table names its source:

- **Frozen verdict run** (`results/cache_eviction/bench/bench.json`, `candidates.jsonl`, `bench.md`, 06:54, implementer
  `google/gemini-4-argon`, trace `results/traces/bench3-20261018/calls.jsonl`, all 36 implementer calls cache hits).
  This is the run behind the published verdict: plain 5 / literature_random 7 / blind_spot 6 candidates.
- **Later draws** (`results/cache_eviction/bench/seed0/*/run.json`, `seed1/*/run.json`; the 03:46-04:00 ablation
  operators from phase2, and plain / literature_random / blind_spot rewritten at 08:05-08:12 by the
  `bench-anthropic` run, `anthropic/claude-sonnet-5.5`). A bench was still writing these files during the analysis,
  so their counts are a snapshot.

## 1. Per operator

Frozen verdict run (`bench/candidates.jsonl`):

| operator | n | valid | invalid reasons | dup code | dup dev score | LRU-equivalent | median dev | best dev | best held-out | $/valid (cached) | $/valid (cold) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | 5 | 4 (0.80) | 1 IndentationError | 0 | 1 | **2** (0.6752 = LRU) | **0.6568** | **0.5666** | 0.5667 | 0.033 | 0.033 |
| literature_random | 7 | 7 (1.00) | - | 0 | 0 | 1 | 0.5935 | 0.5844 | 0.5842 | 0.014 | 0.059 |
| blind_spot | 6 | 5 (0.83) | 1 IndentationError | 0 | 0 | 0 | 0.5942 | 0.5842 | 0.5829 | 0.022 | **0.713** |

Later draws (`bench/seed*/*/run.json`, snapshot):

| operator (seed) | n | valid | invalid reasons | dup dev score | median dev | best dev | median code chars |
|---|---|---|---|---|---|---|---|
| plain (s0, sonnet) | 12 | 12 | - | 6 | 0.5860 | 0.5687 | 1672 |
| plain (s1, sonnet) | 7 | 6 | 1 "no code in reply" | 1 | 0.5800 | 0.5745 | 1726 |
| literature_random (s0, sonnet) | 9 | 9 | - | 2 | 0.5935 | 0.5687 | 1880 |
| blind_spot (s0, sonnet) | 6 | 6 | - | 0 | 0.5949 | 0.5856 | 2717 |
| blind_spot_contrast (gemini) | 6 | 6 | - | 0 | 0.5935 | 0.5793 | 6173 |
| blind_spot_contrast_random | 6 | 6 | - | 0 | 0.5966 | 0.5832 | 5699 |
| blind_spot_feedback | 5 | 5 | - | 0 | 0.5967 | 0.5883 | 6701 |
| blind_spot_feedback_sham | 6 | 6 | - | 0 | 0.6008 | 0.5896 | 6063 |
| literature_all | 8 | 7 | 1 IndentationError | 1 | 0.5966 | 0.5845 | 4506 |
| adhd_frames | 4 | 4 | - | 0 | 0.6229 | 0.5758 | 5857 |

Invalid reasons across every bench record on disk: 3 IndentationError, 1 "no code in reply", 0 timeouts,
0 crashes at runtime. In the frozen run both IndentationErrors are implementer formatting slips, not truncation:
`plain-s0-001` has a 9-space indent at line 104 of 105 (`finish_reason=stop`, 0 code fences left in the code);
`blind_spot-s0-002` has an "unexpected indent" at line 9 of 197 (a docstring continuation), `finish_reason=stop`.
`finish_reason=length` never occurs in `bench3` (0/36 implementer calls); it occurs once in `bench3b` (plain, after the
budget shrank `max_tokens` to 14873) and 12/100 times in the herd-map samples (`samples.jsonl`).

## 2. Timeouts: none in the benchmark

- Evaluator limit: `tasks/cache_eviction/evaluator.py` `TIMEOUT_S = 60` for 6 simulations (3 traces x 2 sizes,
  ~50k requests each); `diverge/reality.py` `Evaluator(timeout=600)` wraps it.
- Every candidate record carries `dev_detail.runtime_s`. Across all 10 run.json files the maximum is **3.73 s**
  (blind_spot_feedback/002); blind_spot's max is 3.06 s, literature_random's 3.47 s, plain's 3.35 s. The 08:05-08:12
  files were produced while the machine was loaded and still sit 16x under the limit.
- `grep -i timeout` over `bench/*.json`, `candidates.jsonl` and the three trace files: 0 hits in error fields
  (the two hits in `phase2` are inside generated prose about latency).
- The only evaluator timeouts anywhere are 6 herd-map samples in `results/cache_eviction/samples.jsonl`, all
  `meta-llama/llama-4-maverick` (ids 007, 009, 011, 016, 017, 019; 350-524 completion tokens, i.e. tiny programs that
  scan the cache on every evict). They affect the consensus map's effective n, not the operator benchmark, and
  would have been invalid on an idle machine too (O(n) evict over 50k requests is the failure, not load).

So the "timeouts invalidated the overlooked operators" hypothesis is rejected by the data on disk.

## 3. Family fit: the blind spots are the weaker families for this workload

Workload (`tasks/cache_eviction/evaluator.py`): stationary Zipf(0.9) over 5,000 keys, a 1,000-key sequential scan
every 10,000 requests, and a trace where 50% of requests are one-hit keys; cache = 1% and 10% of unique keys.

`blind_spots.json` picks 4 families with herd share 0.00-0.01 (LRU-K, LeCaR/CACHEUS, LIRS/CLOCK-Pro,
CLOCK/SIEVE). The herd (`herd_map.json`) concentrates 75% on W-TinyLFU (33%), 2Q/S3-FIFO (24%), LFU (18%).

Dev score by family, pooled over every implementation on disk (herd samples via `samples.jsonl`+`labels.jsonl`;
template implementations via all `run.json` + `candidates.jsonl` by `intended_family`):

| family | herd share | herd samples (n, median, best) | template implementations (approx. n, median, best) |
|---|---|---|---|
| W-TinyLFU (consensus) | 0.33 | 30, **0.5772**, 0.5730 | 1, 0.6752 (degenerated to LRU) |
| 2Q / S3-FIFO (consensus) | 0.24 | 24, 0.5889, 0.5856 | 4, 0.5857, 0.5856 |
| LFU (consensus) | 0.18 | 11, 0.5936, 0.5713 | 5, 0.5844, **0.5687** |
| ARC | 0.07 | 6, 0.5938, 0.5748 | 5, 0.5935, 0.5882 (0.5935 four times) |
| CLOCK / SIEVE (blind spot) | 0.01 | 1, 0.6635 | 6, 0.5903, 0.5856 |
| LeCaR / CACHEUS (blind spot) | 0.00 | - | 11, 0.5942, 0.5793 |
| LRU-K (blind spot) | 0.00 | - | 11, 0.5966, 0.5883 (0.5966 eight times) |
| LIRS / CLOCK-Pro (blind spot) | 0.00 | - | 13 (1 invalid), 0.5966, 0.5948 |

Reading: no blind-spot family has ever produced a dev score below 0.579, while the consensus family W-TinyLFU
sits at 0.577 median from plain sampling and the winning plain draw (an aged-LFU heap plus a FIFO tier for
first-seen keys, labeled LFU) reaches 0.5666. The workload rewards exactly what the herd already does: frequency
with aging for Zipf, a probation tier for one-hit keys, and scan isolation. LRU-K and LIRS classify by reuse
distance, which is uninformative when half the keys never recur; LeCaR mixes LRU and LFU experts and inherits
LRU's scan damage. Absence from the consensus was treated as an opportunity signal; here it is a quality signal.
Related: `blind_spots.json` lists LeCaR/LIRS/LRU-K at share 0.00 with `share_upper_95 = 0.0295` from 100 samples,
so the map cannot even distinguish "overlooked" from "rare".

## 4. Prompt effects: the "faithfully within the family" template

`diverge/reality.py` `IMPLEMENT_TEMPLATE` ends with "Implement this approach faithfully and make it as strong as
you can within the family." Every blind_spot / literature_* candidate on disk used it (`faithful` present in 100% of
their prompts, 0% of plain's; `code_features.py`).

| group | n | median chars | ghost/history | probation or window queue | scan / one-hit handling | adaptive |
|---|---|---|---|---|---|---|
| plain (bench3 + sonnet draws) | 23 | 1672 | 1.00 | 1.00 | 0.00 | 0.33-0.67 |
| herd samples (5 models) | 87 | 1578-5241 | 0.00-1.00 | 0.38-1.00 | 0.00-1.00 | 0.00-0.65 |
| blind_spot (sonnet) | 6 | 2717 | 0.67 | 0.50 | 0.33 | 0.50 |
| blind_spot_* gemini ablations | 29 | 5699-6701 | 0.83-1.00 | 0.80-1.00 | 0.67-1.00 | 0.33-0.80 |
| literature_random (sonnet) | 9 | 1880 | 0.89 | 0.33 | 0.22 | 0.44 |

Two effects, both tied to files:

- **Resampling one prompt returns one program.** The template plus temperature 1.0 gives near-identical
  programs on retry: LRU-K scores 0.5966 in 8 of 11 attempts and the exact value 0.59661606 appears 4 times
  (bench3 `blind_spot-s0-000`, blind_spot_contrast/000, blind_spot_contrast_random/004, blind_spot_feedback_sham/000); ARC 0.5935118
  appears 4 times across literature_random/001, /005 and literature_all/005. Round-robin retries on a blind spot
  therefore buy no new draw (see `blind_spot_directions.py` docstring: "several attempts scored exactly alike").
- **The template blocks the hybrid that wins.** Plain's winner (`candidates.jsonl` `plain-s0-004`, 1489 chars,
  53 lines) is two mechanisms: an exponentially decayed frequency heap plus a FIFO tier that evicts frequency-1
  keys first. Under the faithful template, LeCaR (blind_spot's best, `blind_spot-s0-001`, 7389 chars) bounds its
  history and caps counts to stay textbook. Code length does not predict score (corr(len, score) = +0.15 over
  68 valid bench candidates; longer is slightly worse), so the template's extra bulk is not buying quality.

Also notable: three valid candidates (`plain-s0-000`, `plain-s0-003`, `literature_random-s0-006`) and
`literature_all/000`, `/007` score exactly dev LRU (0.6752227): "W-TinyLFU" and "aged LFU" programs whose admission
or window logic degenerates to LRU. They count as valid and as distinct families but add nothing; the current
metrics do not flag them.

## 5. Selection and metric: a max over four draws, and a threshold everyone clears

- **The headline is one tail draw.** In the frozen run plain's four valid dev scores are 0.5666, 0.6383, 0.6752,
  0.6752 (median 0.6568, the worst median of the three operators); blind_spot's five are 0.5842-0.5966 (median
  0.5942, sd 0.0047). `bench.md` reports only the best held-out score, so plain "wins" with 0.5667 while being the
  least reliable operator. Bootstrapped expected best-of-k from the pooled draws (`selection.py`, dev):

  | operator (pool) | n | median | E[best-of-1] | E[best-of-3] | E[best-of-5] |
  |---|---|---|---|---|---|
  | plain (sonnet, 24 draws) | 24 | 0.5862 | 0.5853 | 0.5815 | 0.5788 |
  | bench3 plain (gemini) | 4 | 0.6568 | 0.6396 | 0.5845 | - |
  | bench3 blind_spot | 5 | 0.5942 | 0.5922 | 0.5870 | 0.5842 |
  | bench3 literature_random | 7 | 0.5935 | 0.6024 | 0.5865 | 0.5849 |
  | blind_spot (sonnet) | 6 | 0.5949 | 0.5952 | 0.5896 | 0.5869 |

  With one repeat (`config.repeats = 1`) and 4-7 valid draws per operator, the published ordering is within
  the noise of a single draw. Held-out moves nothing: dev-to-held-out gaps are -0.0013 to +0.0005 across all
  selected candidates, so the "held-out" column is the dev max re-measured.
- **useful_discoveries is a family count, not a quality metric.** It counts families whose held-out score beats
  LRU 0.6747 (`diverge/bench.py` `run_metrics`, `evaluator.beats`). 17 of 18 valid candidates in the frozen run
  beat LRU; the exceptions are the LRU-equivalent programs. On this metric blind_spot (3) and literature_random
  (4) beat plain (2), but the verdict ranks by discoveries per dollar, and blind_spot's cold-start denominator
  includes the $3.455 herd map (`bench.md` Notes), giving 0.84/$ against literature_random's 9.75/$ and plain's
  15.27/$. The map cost is charged to one repeat of one operator, so blind_spot cannot win cold-start at any
  budget under $3.

## Top 3 causes, ranked by evidence

1. **Family fit (strongest).** Every blind-spot family implemented on disk (41 programs) lands at dev
   0.579-0.606; the consensus W-TinyLFU / aged-LFU+probation programs land at 0.567-0.577. The blind spots are
   intrinsically weaker on a stationary Zipf + scan + one-hit workload, and the selection rule (share = 0 among
   100 samples, upper 95% bound 0.03) cannot separate "overlooked" from "rightly ignored".
2. **Selection noise and the metric.** One repeat, 4-7 draws, a max reported as the headline, and a usefulness
   threshold (beats LRU) that 17/18 valid candidates clear. Plain has the worst median (0.657) and the best max
   (0.567) in the same run; the cold-start denominator charges a $3.46 map to a $0.11 operator.
3. **The faithful template.** It removes the hybrid room that produced plain's winner, and resampling it at
   T=1.0 returns the same program (LRU-K at 0.5966 in 8 of 11 attempts, four identical ARC scores), so retries are wasted.
   Timeouts, truncation and crashes are not causes: 0 timeouts, 0 `finish_reason=length`, 3 indentation slips
   and 1 empty reply across 72 bench candidates.

## Replacements

| change | where | what the data says |
|---|---|---|
| Hybrid template as the default arm for family operators | `diverge/operators/blind_spot_hybrid.py` (exists), `reality.py` `HYBRID_TEMPLATE`; add `literature_random_hybrid` as the matched control | the only programs under 0.575 are hybrids (plain-s0-004, plain s0/s1 draws); keep `blind_spot` as the purity control and report purity |
| A different direction per retry | `diverge/operators/blind_spot_directions.py` (exists; `hybrid_jobs` combines both levers) | LRU-K at 0.5966 in 8/11 attempts and 4x identical ARC scores show round-robin resampling is a no-op |
| Filter blind spots by workload fit before spending | `diverge/blind_spots` selection: require one probe implementation per family (one call) and drop families whose probe is more than 0.01 behind the herd's best family median, or weight by `share_upper_95` | blind-spot medians 0.590-0.597 vs W-TinyLFU 0.577; none of the four has ever beaten 0.579 |
| Report medians, best-of-k curves and paired seeds, not one max | `diverge/bench.py` `report`, `verdict`, `aggregate`: add median valid dev, bootstrapped E[best-of-k] for k=1..n, share of candidates within 0.005 of the run best; require `repeats >= 3` before a verdict | plain median 0.657 vs blind_spot 0.594 in the same run the verdict calls plain's win |
| Raise the usefulness bar and flag degenerate programs | `bench.py` `run_metrics`: count a family as useful only if it beats the herd's best family median (0.577 here) or the run's best plain draw; mark candidates whose score equals a baseline to 1e-6 as `degenerate` | 17/18 valid candidates beat LRU; 5 candidates on disk score exactly LRU and are counted as valid discoveries |
| Charge the map fairly | `bench.py` `setup_cost`: amortise the herd map over the number of operators and repeats that use it, or report cold-start only when the map was actually built in the run (`new_spend_usd` was $0 here) | cold-start turns 27.1 discoveries/$ into 0.84/$ for a map that cost $0 new this run |
| Evaluator timeout handling: no change needed for this task | `tasks/cache_eviction/evaluator.py` | max runtime 3.7 s of 60 s under heavy load; record `runtime_s` in the bench table so this check is one grep next time |
