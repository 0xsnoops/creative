# Measured results (as of 2026-10-18)

Only measured numbers are recorded here. The raw data (`results/`, `cache/`, traces) is
gitignored and kept locally. Protocols: `docs/PREREGISTRATION.md` (operator benchmark) and
`docs/PREREGISTRATION-overlook.md` (paraphrase test and overlook).

Total spent through HERD so far: **$11.84** (cap $28):
- $0.50 smoke test
- $6.16 for the two N=20 herd maps
- $0.44 lost to a benchmark bug, since fixed
- about $1.5 for the first, partial benchmark repeat
- $3.07 for the paraphrase test and overlook
- $0.62 for the live strategy-loop runs

The outline-only model review cost $0.69 separately.

## Herd maps (N=20 per model, 5 models)

| task | valid samples | top 3 families | share held by the top 3 |
|---|---|---|---|
| online_bin_packing | 95 / 100 | Tight-fit-or-spacious gap avoidance; Best Fit; Sum-of-Squares gap-count balancing | 62% |
| cache_eviction | 76 / 100 | W-TinyLFU (sketch frequency duel); 2Q / S3-FIFO (probationary queue + ghosts); LFU | 75% |

## Is the herd real? Partly a prompt effect

Method:
- 3 meaning-preserving paraphrases of each task prompt. Automatic checks confirmed identical
  code blocks and numbers, and I read every paraphrase.
- Each sampled on Claude Sonnet 5.5 and GPT-6.1 Sol, 5 samples per paraphrase per model.
- Labeled with each map's own taxonomy.

Comparison: total variation distance (TVD) between 10-sample groups.

| task | resampling the original (mean / max TVD) | paraphrase vs original (mean TVD) | pre-registered verdict |
|---|---|---|---|
| online_bin_packing | 0.25 / 0.40 | **0.51** | partly a prompt effect |
| cache_eviction | 0.07 / 0.10 | **0.35** | partly a prompt effect |

On cache eviction, rewording alone moved W-TinyLFU from 45% to 67% of samples and 2Q/S3-FIFO
from 50% to 17%. One of the three paraphrases stayed close to the original (TVD 0.1–0.2) and two
moved a lot (0.4–0.5). With groups of 10 samples, only large shifts can be detected. Herd claims
should be read as "for this wording".

## Overlook (`herd overlook <task>`)

### online_bin_packing (95 valid samples, $0.17)

**Inputs the herd uses.** Static analysis covered all 95 samples.

| input | share of samples using it |
|---|---|
| **n_open_bins** (how many bins are offered) | **5%, overlooked** |
| bin_order (bin age) | 36% |
| history (state across calls) | 40% |
| new_bin_slot (special-casing the new bin) | 60% |
| item_distribution (the stated Weibull) | 69% |
| bin_capacity | 82% |
| item, capacity_values | 100% |

**Shared assumptions.** Of the 8 proposed assumptions, only 1 was held by every one of the 12
checked samples: ties are resolved by argmax order, and bin age is not used as a signal. It is
classed hard because the prompt states the tie rule.

**Component grid.**
- Literature combinations no herd sample occupies: Next Fit, First Fit, Worst Fit, Harmonic
  size classes, and evolved composite scoring.
- Coherent value pairs never seen together, for example "inventory potential change" scoring
  with no persistent state, and an item-size-dependent new-bin rule with expected-waste scoring.

**Minority report.** Families produced by only one model:
- Worst Fit and First Fit (Llama only)
- distribution-learning packing (Anthropic only)
- evolved composite scoring and a piecewise band score (DeepSeek only)

**Latent tail.** Verified approaches outside the herd, listed with vs without the herd shown:

| model | with herd shown | without |
|---|---|---|
| Anthropic | 6 | 5 |
| OpenAI | 5 | 5 |
| Google | 7 | 4 |
| Meta | 2 | 4 |
| DeepSeek | 3 | 4 |

Only 2 of 5 models named more, so there is **no evidence** that showing the herd unlocks hidden
knowledge. The pre-registered bar was 4 of 5.

### cache_eviction (76 valid samples, $0.17)

**Inputs the herd uses.** Static analysis decided the clear cases; the extractor model resolved
the ambiguous ones (workload_mix always, time and frequency often).

| input | share |
|---|---|
| workload_mix (logic aimed at scans, one-hit keys, skew) | 12% |
| key_value (ordering or arithmetic on keys) | 42% |
| time | 54% |
| capacity | 80% |
| eviction_history (ghosts) | 82% |
| occupancy | 86% |
| frequency | 97% |

No input is under the 10% "overlooked" line. workload_mix is close.

**Shared assumptions**, held by all 12 checked samples. 7 of the 14 proposed passed.
- Incidental, so these may be broken:
  - keys are opaque labels, and no solution uses key values or ordering to predict reuse;
  - scan resistance comes from first-seen vs re-referenced status, never from detecting
    sequential keys;
  - recency and frequency are counted in requests, mostly with exact per-key counters rather
    than bounded sketches;
  - no lookahead.
- Hard: every miss inserts the key, so the interface allows no admission refusal; each instance
  serves one trace; the policy mirrors the simulator's cache.

**Component grid.**
- Literature combinations no herd sample occupies: plain LFU, CLOCK / reference bits, LRU-K,
  ARC, and LIRS.
- Coherent unseen pairs, for example small-queue ghost promotion combined with periodic counter
  halving.

**Minority report.**
- Plain LRU appears only from Llama.
- CLOCK-style lazy promotion appears only from DeepSeek.
- W-TinyLFU, a top herd family, is produced almost entirely by Google and OpenAI. "Produced by
  few models" is not the same as "rare".

**Latent tail.**

| model | with herd shown | without |
|---|---|---|
| Anthropic | 4 | 4 |
| OpenAI | 3 | 3 |
| Google | 6 | 2 |
| Meta | 3 | 2 |
| DeepSeek | 5 | 3 |

3 of 5 models named more, so again **no evidence** by the pre-registered bar of 4 of 5.

## Operator benchmark: stopped partway, no verdict

Only repeat 1 of 3 for the 9 cache-eviction operators ran. That is dev scores only, with no
labeling and no held-out scoring.

| best dev miss ratio | plain | adhd_frames | blind_spot_contrast | blind_spot_contrast_random | literature_all | literature_random | blind_spot | blind_spot_feedback | blind_spot_feedback_sham |
|---|---|---|---|---|---|---|---|---|---|
| | 0.5666 | 0.5758 | 0.5793 | 0.5832 | 0.5845 | 0.5844 | 0.5842 | 0.5883 | 0.5896 |

LRU scores 0.6752 and FIFO 0.7097. With one repeat of 4–8 candidates each, this does not separate
the operators. Resume with `scripts/run_benches.py`, which replays completed calls from the
cache at no cost.

### Three-operator run to held-out (1 repeat, preliminary)

This is the core comparison: plain sampling, random literature, and literature chosen because the
herd overlooked it. Each operator got $0.30 and 1 repeat (seed 0). Candidates were selected on dev,
and each family's pick was then scored once on held-out. The dev calls were replayed from the
cache, so the new spend was about $0.

| operator | candidates | valid | useful families (beat LRU on held-out) | best held-out miss ratio |
|---|---|---|---|---|
| plain | 5 | 80% | 2 | **0.5667** |
| literature_random | 7 | 100% | 4 | 0.5842 |
| blind_spot | 6 | 83% | 3 | 0.5829 |

Held-out baselines: LRU 0.6747, FIFO 0.7088. Lower is better.

On held-out, **plain had the best score**. blind_spot did not beat random literature or plain on
best score. Random literature produced the most distinct families that beat LRU. This is one
repeat, so it is no verdict either way.

## Live strategy-loop runs (preliminary, 1 run each)

Scores are % excess bins, as dev / held-out; lower is better.

| trajectory | HERD verdict | strategy vs code similarity | best attempt | fresh_start |
|---|---|---|---|---|
| open revisions, 3.66 → 0.30 | keep | 0.30 vs 0.21 | 0.299 / 0.258 | 0.279 / 0.248 |
| "keep the approach, tune it", 3.66 → 0.36 | keep | 0.29 vs 0.42 | 0.359 / 0.537 | 1.376 / 1.409 |
| 4 more tuning revisions, stalled 0.66 → 1.85 (`herd demo strategy_loop`) | repair, not a loop (below the 0.7 threshold) | 0.45 vs 0.64 | 0.658 / 0.953 | 0.628 / 0.834 |

Baselines: best fit 4.15 / 3.98, first fit 4.54 / 4.31.

HERD never returned `escape` in these runs, so no escape candidate was generated.

On the stalled window, the lever HERD would have tried was `n_open_bins`. It is used by 5 of 95
herd samples: 2 from DeepSeek and 1 each from Gemini, Llama and GPT-6.1. I implemented it anyway as
an ablation, at the same budget as fresh_start. It scored 1.096 / 0.973 and lost to fresh_start.

These runs are no evidence either way about escape vs fresh_start at scale. They show one weakness:
strategy similarity partly depends on how the extractor model describes each attempt, so tuning
revisions of the same code can measure as different strategies.

Fixtures: `fixtures/trajectories/` and `fixtures/runs/`, as `strategy_loop`,
`strategy_loop_tune` and `strategy_loop_unconstrained`. Reproduce with
`scripts/strategy_loop_run.py --mode open|tune|plateau` and `scripts/forced_escape.py`; cached
calls replay at no cost. Replay with `herd demo <fixture>`.

## Mini rescue test (preliminary, n = 4 stuck states)

`scripts/rescue_test.py`; raw data in `results/rescue/` (gitignored).

**Setup**
- **Agent model:** Llama 4 Maverick. It is the cheapest sampler, and its best herd samples are
  exactly best fit.
- **Trajectories:** it wrote 6 trajectories of 6 attempts with dev-score feedback. 4 were
  naturally stuck: no dev improvement of more than 1% over the last 3 attempts.
- **Arms:** each stuck state got four arms, all using the same model, at most 3 dev-scored
  candidates per arm, and a $0.40 cap per arm.
- **Rescue:** the arm's dev-selected candidate beats the stuck best on held-out.

| stuck state (held-out of the stuck best) | fresh_start | try_different | random_lever | herd_escape v2 |
|---|---|---|---|---|
| 0 (148.12) | 148.12 no | **3.98 yes** | 6.88 yes | 4.02 yes |
| 1 (3.98, best fit) | 12.97 no | 3.98 no | 4.89 no | 12.97 no |
| 3 (3.98, best fit) | 3.98 no | 3.98 no | no admitted candidate | no admitted candidate |
| 4 (3.96) | 12.97 no | 3.96 no | 10.53 no | 148.12 no |
| **rescued** | 0/4 | 1/4 | 1/4 | 1/4 |

**What this shows:** herd_escape v2 did not beat the simple "try a fundamentally different
approach" arm. The only rescue came in state 0, which was stuck far from best fit, and three arms
rescued it there; try_different did best. No arm beat a plateau at best fit.

**Cost:** herd_escape cost $0.010 in total and was the most expensive arm, but every arm was cheap.

**Lever:** HERD chose a family lever, "Worst Fit / Almost-Any-Fit variants", which is a weak
direction for escaping best fit.

**Admission check:** it rejected 17 escape and random-lever candidates for not differing enough
from the stuck attempts or not depending on the lever.

**Bug fixed before these numbers:** the first run of this test exposed a bug in the sketch filter.
Family and paraphrased levers were always dropped. It was fixed in commit 6ba6309, and the table
above comes from the rerun.

## Not yet measured

- An `escape` verdict on a live trajectory. It never triggered in the three runs above.
- Whether escape beats "continue", "be creative" or a fresh model at an equal budget. This is the
  agent-recovery benchmark.
