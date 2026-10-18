# E1 results: detector validity (bin packing)

## E1b (supersedes the $0 verdict below): full LLM coverage, three small extractor models

**Verdict: behavior beats the LLM signal.** The $0 run had cached LLM fields for only 11
programs, so it was inconclusive. I reran it with paid extractor calls for all 140 programs, using
each of three small models, for $0.31 in total. The command is
`scripts/e1_detector_validity.py --paid 1.0 --extractor <model>`.

Primary target, all 9,730 pairs (2,074 positive):

| extractor | covered | cost | behavior AUC | LLM strategy AUC | code AUC | behavior − LLM [95% CI] |
|---|---|---|---|---|---|---|
| anthropic/claude-haiku-5.5 | 140/140 | $0.088 | 0.661 | 0.503 | 0.527 | **+0.158 [+0.099, +0.222]** |
| deepseek/deepseek-v4-pro-0813 | 140/140 | $0.110 | 0.661 | 0.526 | 0.527 | **+0.135 [+0.072, +0.204]** |
| meta-llama/llama-4-maverick | 138/140 | $0.108 | 0.653 | 0.502 | 0.526 | **+0.152 [+0.084, +0.239]** |

With every extractor model, the LLM-description strategy similarity is at chance (0.50–0.53) for
predicting the same failure set. Behavior is about 0.15 higher, and every CI excludes 0.

On the secondary target (failure sets relative to best fit), no signal beats chance with any
extractor. Behavior − LLM is −0.012, −0.001 and −0.007, and every CI includes 0. All caveats below
still apply. Raw data: `results/e1/results-<model>.json`.

## Original E1 ($0, cache-only)

**Verdict: behavior does not beat the LLM signal.** On the 55 pairs where both programs have
cached LLM-extracted fields (11 programs), AUC(behavior) − AUC(LLM strategy similarity) = +0.07,
with a 95% bootstrap CI of [−0.07, +0.32]. That CI includes 0. The comparison is badly
underpowered (11 programs, 10 positive pairs), so it is **inconclusive**, not evidence that the
two signals are equal.

**Code similarity:** across all 9,730 pairs, behavior beats code similarity. AUC 0.66 vs 0.53;
difference +0.14, CI [+0.07, +0.19], which excludes 0. Code similarity is close to chance.
On the 55 LLM-covered pairs, code similarity scores 0.74 vs 0.71 for behavior, with very wide CIs.

**Behavior vs AST-only strategy similarity (all pairs):** behavior 0.66 vs 0.50; difference +0.16,
CI [+0.09, +0.24]. This comparison is extra. It does not replace the LLM comparison, because the
LLM-covered set reached 55 pairs, above the 30-pair floor.

**Pre-registered E1 success criterion:** behavior AUC is at least 0.10 above the embedding AUC
(docs/RESEARCH-loop-escape.md). No embedding signal was computed here. Behavior is 0.14 above
code similarity and 0.16 above AST strategy similarity, but the absolute behavior AUC is modest (0.66).

## Setup

- **Programs:** 162 collected, 140 unique after deduplicating by the sha256 of the code. Sources:
  - 95 valid herd samples;
  - 12 fixture trajectory attempts;
  - 13 `demo/*.py`;
  - 13 `escape/**/*.py`;
  - 29 `paraphrase/*.py`.
  
  All 140 programs were valid on dev with the real sandboxed evaluator, and all had valid probe
  fingerprints. None were dropped.
- **Target:** `same_failure = jaccard(failure_set(i), failure_set(j)) >= 0.8`, with k=2 of the 5 dev
  instances (seeds 0–4). With k=2, this means both programs have exactly the same two worst seeds.
- **Signals:**
  - behavior = 1 − `fingerprint.distance` over the 200 probe states;
  - code = `strategy.code_similarity`;
  - LLM strategy = `strategy.similarity(...)["overall"]` with `extract=llm_extractor(Herd(budget_usd=0.0))`;
  - AST strategy = the same with `extract=None`, which uses only the inputs and state parts.
  
  In both strategy fingerprints `behavior=False`, so the strategy signals do not include behavior.
- **Metrics:**
  - Mann–Whitney AUC, with ties counted as 0.5.
  - Program-level bootstrap: 300 reps, resampling programs with replacement and keeping all
    pairs of distinct draws. CIs are 2.5/97.5 percentiles.
  - Paired differences use the same resamples.
- **Cost:** `herd.llm._post` was replaced with a function that raises, and the run cap was $0.
  Of the 140 extracts, 129 were blocked as uncached and 11 were cache hits. The script asserts
  `herd.ledger.new_spend == 0`, and the run reported $0.00.
- **Runtime:** 44 s for the full run on 10 CPUs: evaluation 35 s, fingerprints 6 s, bootstrap a few seconds.

## Primary target: same top-2 worst dev instances

| pair set | n programs | n pairs | positives | base rate | behavior | code | LLM strategy | AST strategy |
|---|---|---|---|---|---|---|---|---|
| all pairs | 140 | 9,730 | 2,074 | 0.213 | **0.662** [0.595, 0.734] | 0.527 [0.488, 0.577] | 0.522* [0.489, 0.569] | 0.500 [0.477, 0.530] |
| LLM-covered pairs | 11 | 55 | 10 | 0.182 | 0.714 [0.206, 1.00] | 0.744 [0.264, 1.00] | 0.644 [0.172, 0.959] | 0.694 [0.233, 1.00] |

\* On all pairs, "LLM strategy" is really AST-only for the 129 programs that have no cached extract.
It is a mixed signal, shown only for completeness.

Paired differences (bootstrap 95% CI):

| difference | pair set | value | CI | excludes 0? |
|---|---|---|---|---|
| behavior − LLM strategy | LLM-covered (55 pairs) | +0.070 | [−0.073, +0.324] | no |
| behavior − code | all (9,730) | +0.135 | [+0.072, +0.194] | yes |
| behavior − AST strategy | all (9,730) | +0.162 | [+0.091, +0.240] | yes |

## Secondary target (clearly secondary): failure sets relative to best_fit

Each instance gets a gap: the program's excess % minus best_fit's excess % on the same seed.
The failure set is the k=2 seeds with the largest gaps, and the same Jaccard ≥ 0.8 rule applies.

| pair set | n programs | n pairs | base rate | behavior | code | LLM strategy | AST strategy |
|---|---|---|---|---|---|---|---|
| all pairs | 140 | 9,730 | 0.237 | 0.541 [0.497, 0.599] | 0.521 [0.487, 0.564] | — | 0.542 [0.493, 0.594] |
| LLM-covered pairs | 11 | 55 | 0.509 | 0.671 [0.323, 1.00] | 0.433 [0.269, 0.765] | 0.739 [0.243, 1.00] | 0.792 [0.384, 1.00] |

- behavior − AST strategy, all pairs: −0.001, CI [−0.052, +0.062].
- behavior − LLM strategy, covered pairs: −0.069, CI [−0.319, +0.185], from 294 of 300 reps
  (6 reps had only one class).

On the secondary target, no signal is clearly above chance.

## Coverage

Cached LLM extractor replies exist for only 11 of the 140 programs. Ten of them are trajectory
fixture attempts and one is a herd sample. The 11 mechanism labels are near-synonyms:
"value-function best fit", "distribution-aware best fit", "stochastic dynamic programming" and
similar. The LLM-covered set is therefore a narrow slice of the population: late
strategy-loop attempts that are mostly variations of one idea. It cannot represent the
100+ herd samples.

## Caveats

- **Pairs are not independent.** Each program appears in up to 139 pairs. The program-level
  bootstrap accounts for this roughly, but with 11 programs the percentile CIs are coarse and
  unstable.
- **The target is coarse.** It uses 5 dev instances and k=2, so there are only 10 possible
  failure sets. One set, {seed:2, seed:3}, covers 55 of 140 programs. Part of "same failure set"
  is therefore instance difficulty shared by most heuristics, not strategy.
- **The relative target behaves differently.** Its most common set is {seed:1, seed:4} (59 of 140
  programs), and no signal predicts it.
- **Some differences are tiny.** Excess-bin differences between seeds are often 1–2 bins out of
  about 2,000, so the worst-2 ranking for near-best-fit programs is close to noise.
- **No embedding baseline.** No embedding baseline was computed, so this report does not test the
  pre-registered E1 comparison (behavior vs embedding).
- **Probe coverage.** The behavior signal uses single-decision probe states from best-fit
  trajectories. It does not capture stateful or long-horizon effects.

Raw numbers and per-program failure sets are in `results/e1/results.json`, which is gitignored.
To reproduce: `~/.local/bin/uv run python scripts/e1_detector_validity.py`.
