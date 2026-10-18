# Measured results (as of 2026-10-18)

Only measured numbers are recorded here. The raw data (`results/`, `cache/`, traces) is
gitignored and kept locally. Protocols: `docs/PREREGISTRATION.md` (operator benchmark) and
`docs/PREREGISTRATION-overlook.md` (paraphrase test and overlook).

Total spent through DIVERGE so far: **$11.84** (cap $28):
- $0.50 smoke test
- $6.16 for the two N=20 consensus maps
- $0.44 lost to a benchmark bug, since fixed
- about $1.5 for the first, partial benchmark repeat
- $3.07 for the paraphrase test and overlook
- $0.62 for the live strategy-loop runs

The outline-only model review cost $0.69 separately.

## Consensus maps (N=20 per model, 5 models)

| task | valid samples | top 3 families | share held by the top 3 |
|---|---|---|---|
| online_bin_packing | 95 / 100 | Tight-fit-or-spacious gap avoidance; Best Fit; Sum-of-Squares gap-count balancing | 62% |
| cache_eviction | 76 / 100 | W-TinyLFU (sketch frequency duel); 2Q / S3-FIFO (probationary queue + ghosts); LFU | 75% |

## Is the consensus real? Partly a prompt effect

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
moved a lot (0.4–0.5). With groups of 10 samples, only large shifts can be detected. Consensus claims
should be read as "for this wording".

## Overlook (`python -m diverge overlook <task>`)

### online_bin_packing (95 valid samples, $0.17)

**Inputs the consensus uses.** Static analysis covered all 95 samples.

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
- Literature combinations no consensus-map sample occupies: Next Fit, First Fit, Worst Fit, Harmonic
  size classes, and evolved composite scoring.
- Coherent value pairs never seen together, for example "inventory potential change" scoring
  with no persistent state, and an item-size-dependent new-bin rule with expected-waste scoring.

**Minority report.** Families produced by only one model:
- Worst Fit and First Fit (Llama only)
- distribution-learning packing (Anthropic only)
- evolved composite scoring and a piecewise band score (DeepSeek only)

**Latent tail.** Verified approaches outside the consensus, listed with vs without the consensus shown:

| model | with consensus shown | without |
|---|---|---|
| Anthropic | 6 | 5 |
| OpenAI | 5 | 5 |
| Google | 7 | 4 |
| Meta | 2 | 4 |
| DeepSeek | 3 | 4 |

Only 2 of 5 models named more, so there is **no evidence** that showing the consensus unlocks hidden
knowledge. The pre-registered bar was 4 of 5.

### cache_eviction (76 valid samples, $0.17)

**Inputs the consensus uses.** Static analysis decided the clear cases; the extractor model resolved
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
- Literature combinations no consensus-map sample occupies: plain LFU, CLOCK / reference bits, LRU-K,
  ARC, and LIRS.
- Coherent unseen pairs, for example small-queue ghost promotion combined with periodic counter
  halving.

**Minority report.**
- Plain LRU appears only from Llama.
- CLOCK-style lazy promotion appears only from DeepSeek.
- W-TinyLFU, a top consensus family, is produced almost entirely by Google and OpenAI. "Produced by
  few models" is not the same as "rare".

**Latent tail.**

| model | with consensus shown | without |
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
consensus overlooked it. Each operator got $0.30 and 1 repeat (seed 0). Candidates were selected on dev,
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

| trajectory | DIVERGE verdict | strategy vs code similarity | best attempt | fresh_start |
|---|---|---|---|---|
| open revisions, 3.66 → 0.30 | keep | 0.30 vs 0.21 | 0.299 / 0.258 | 0.279 / 0.248 |
| "keep the approach, tune it", 3.66 → 0.36 | keep | 0.29 vs 0.42 | 0.359 / 0.537 | 1.376 / 1.409 |
| 4 more tuning revisions, stalled 0.66 → 1.85 (`diverge demo strategy_loop`) | repair, not a loop (below the 0.7 threshold) | 0.45 vs 0.64 | 0.658 / 0.953 | 0.628 / 0.834 |

Baselines: best fit 4.15 / 3.98, first fit 4.54 / 4.31.

DIVERGE never returned `escape` in these runs, so no escape candidate was generated.

On the stalled window, the lever DIVERGE would have tried was `n_open_bins`. It is used by 5 of 95
consensus-map samples: 2 from DeepSeek and 1 each from Gemini, Llama and GPT-6.1. I implemented it anyway as
an ablation, at the same budget as fresh_start. It scored 1.096 / 0.973 and lost to fresh_start.

These runs are no evidence either way about escape vs fresh_start at scale. They show one weakness:
strategy similarity partly depends on how the extractor model describes each attempt, so tuning
revisions of the same code can measure as different strategies.

Fixtures: `fixtures/trajectories/` and `fixtures/runs/`, as `strategy_loop`,
`strategy_loop_tune` and `strategy_loop_unconstrained`. Reproduce with
`scripts/strategy_loop_run.py --mode open|tune|plateau` and `scripts/forced_escape.py`; cached
calls replay at no cost. Replay with `python -m diverge demo <fixture>`.

## Mini rescue test (preliminary, n = 4 stuck states)

`scripts/rescue_test.py`; raw data in `results/rescue/` (gitignored).

**Setup**
- **Agent model:** Llama 4 Maverick. It is the cheapest sampler, and its best consensus-map samples are
  exactly best fit.
- **Trajectories:** it wrote 6 trajectories of 6 attempts with dev-score feedback. 4 were
  naturally stuck: no dev improvement of more than 1% over the last 3 attempts.
- **Arms:** each stuck state got four arms, all using the same model, at most 3 dev-scored
  candidates per arm, and a $0.40 cap per arm.
- **Rescue:** the arm's dev-selected candidate beats the stuck best on held-out.

| stuck state (held-out of the stuck best) | fresh_start | try_different | random_lever | diverge_escape v2 |
|---|---|---|---|---|
| 0 (148.12) | 148.12 no | **3.98 yes** | 6.88 yes | 4.02 yes |
| 1 (3.98, best fit) | 12.97 no | 3.98 no | 4.89 no | 12.97 no |
| 3 (3.98, best fit) | 3.98 no | 3.98 no | no admitted candidate | no admitted candidate |
| 4 (3.96) | 12.97 no | 3.96 no | 10.53 no | 148.12 no |
| **rescued** | 0/4 | 1/4 | 1/4 | 1/4 |

**What this shows:** diverge_escape v2 did not beat the simple "try a fundamentally different
approach" arm. The only rescue came in state 0, which was stuck far from best fit, and three arms
rescued it there; try_different did best. No arm beat a plateau at best fit.

**Cost:** diverge_escape cost $0.010 in total and was the most expensive arm, but every arm was cheap.

**Lever:** DIVERGE chose a family lever, "Worst Fit / Almost-Any-Fit variants", which is a weak
direction for escaping best fit.

**Admission check:** it rejected 17 escape and random-lever candidates for not differing enough
from the stuck attempts or not depending on the lever.

**Bug fixed before these numbers:** the first run of this test exposed a bug in the sketch filter.
Family and paraphrased levers were always dropped. It was fixed in commit 6ba6309, and the table
above comes from the rerun.

## Rescue test round 2 (pre-registered in docs/PREREGISTRATION-final.md; preliminary, n = 4)

**Setup**
- **Stuck states:** the same 4 as round 1.
- **Arms:** every arm got fresh seeds and used Claude Haiku 5.5 as the implementer, through the
  Anthropic API, per the amendment.
- **Scoring:** selection was on dev. Held-out was rescored one program at a time
  (`scripts/rescore_rescue.py`), because the machine was overloaded during the run and one program
  timed out. The candidates and dev picks are unchanged.
- **Rescue:** the arm's held-out score beats the stuck best's held-out score.

| state (stuck best held-out) | fresh_start | try_different | random_lever | escape v2 | escape v3 | escape v4 (exploratory) |
|---|---|---|---|---|---|---|
| 0 (148.12) | 3.98 yes | 3.98 yes | 7.54 yes | 3.97 yes | 3.98 yes | **1.66 yes** |
| 1 (3.98, best fit) | 3.98 no | 3.75 yes | **1.21 yes** | 8.88 no | 2.52 yes | no admitted candidate |
| 3 (3.98, best fit) | 4.16 no | 3.98 no | **1.06 yes** | 7.49 no | no admitted candidate | no admitted candidate |
| 4 (3.96) | 3.76 yes | 3.98 no | 6.53 no | 6.46 no | 4.01 no | **1.11 yes** |
| **rescued** | 2/4 | 2/4 | **3/4** | 1/4 | 2/4 | 2/4 |

Cost per arm was $0.003 to $0.08 per state; v3 and v4 cost the most.

**Pre-registered rule:** v3 becomes the default only if it rescues more states than both
`try_different` and `random_lever`. It rescued 2, against 2 and 3, so **v2 stays the default and
v3 stays opt-in**. v4 was added after the pre-registration and is exploratory. It also stays opt-in.

**What the data show**
- **Pipeline vs plain instruction:** arms that go through DIVERGE's escape pipeline (fresh
  context, failure summary, sketches, behavioral admission check) did break best-fit plateaus.
  random_lever reached 1.21 and 1.06, and v3 reached 2.52. Round 1 had none, and "try something
  different" reached 3.75 once.
- **The lever choice:** the arm that rescued most used a random lever (the control). So the
  pipeline helps, but DIVERGE's choice of the overlooked lever has not been shown to matter.
- **v4's quality:** when v4 produced a candidate, it was the best of any arm (1.66 and 1.11).
  But it produced no admitted candidate on 2 of 4 states.
- **Sample size:** n = 4. One state changes any count.

## Three-operator benchmark on the Anthropic API (budget fix applied; preliminary, 2 repeats)

**Setup**
- Task: cache_eviction, held-out miss ratio, lower is better.
- Implementer: Claude Sonnet 5.5. Budget: $0.30 per operator per repeat.
- Every operator now uses its whole budget.

| operator | candidates per repeat | best held-out | distinct families beating LRU | discoveries per $ (cached) |
|---|---|---|---|---|
| plain | 12 | 0.5681 | 2 | 7.67 |
| literature_random | 9-11 | **0.5680** | 4 | 14.74 |
| blind_spot | 6 | 0.5877 | 4 | **15.46** |

- **Best final program:** blind_spot does not produce a better one. Plain and random literature
  tie.
- **Variety:** blind_spot and random literature find twice as many distinct working families as
  plain per repeat.

docs/BENCH_FORENSICS.md explains why. The blind-spot families (LRU-K, LeCaR, LIRS, CLOCK) are weak
for this workload, and "best of a few draws" hides that plain has the worst median.
docs/BENCH_NEW_OPERATORS.md has the hybrid and per-retry-direction operators, if that run finished
before the freeze.

## Not yet measured

- An `escape` verdict on a live trajectory. It never triggered in the three runs above.
- Whether escape beats "continue", "be creative" or a fresh model at an equal budget. This is the
  agent-recovery benchmark.

## New blind-spot operators vs plain, Anthropic, 2 repeats (interim, dev only)

Full write-up: `docs/BENCH_NEW_OPERATORS.md`. Same setup as the 3-operator Anthropic run
(implementer `anthropic/claude-sonnet-5.5`, labeler `anthropic/claude-opus-5.5`, $0.30 per
operator per repeat, seeds 0-1, cache_eviction). `blind_spot_hybrid` allows hybrids of the
blind-spot family with other mechanisms; `blind_spot_directions` gives each retry of a family a
different design direction. At the cutoff the bench had finished generation for 7 of 8
operator-seed runs; labeling and held-out scoring were still running (final table:
`results/bench-new-ops.log`). Dev miss ratio, lower is better; all candidates were valid.

| operator | candidates (s0, s1) | dev best (s0, s1) | dev median (s0, s1) | spent $ (s0, s1) |
|---|---|---|---|---|
| plain | 12, 12 | **0.5665, 0.5675** | 0.5872, 0.5856 | 0.265, 0.256 |
| blind_spot | 5, 6 | 0.5903, 0.5869 | 0.5962, 0.5960 | 0.213, 0.272 |
| blind_spot_hybrid | 8, 8 | 0.5856, 0.5841 | 0.5960, 0.5956 | 0.253, 0.256 |
| blind_spot_directions | 7, running | 0.5889, - | 0.5961, - | 0.256, - |

Reference, the 3-operator Anthropic run (held-out, 2 repeats): plain 0.5681, literature_random
0.5680, blind_spot 0.5877; blind_spot had the most useful families (4 per repeat vs plain's 2).

**Verdict (preliminary):** the hybrid allowance improves blind_spot's dev best in both seeds by
0.003-0.005 and the directions lever at most marginally, but neither closes the ~0.017 gap to
plain sampling. Caveats: dev scores only; 2 repeats; plain and blind_spot did not replay from the
cache as intended, so the controls here are fresh samples at the same seeds. Spend about $2.0 by
the cutoff, projected about $2.6 with labeling.

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
