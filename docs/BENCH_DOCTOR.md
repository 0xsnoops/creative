# Bench doctor: why the blind-spot operators lose on cache_eviction

A panel of 18 models from 18 providers reviewed the cache_eviction operator benchmark (2026-10-18).
The question: why do the blind-spot and literature operators score worse than plain sampling, and
how can the generation pipeline be changed so they produce stronger solutions, without tuning on
held-out data, changing the evaluator or its data, or peeking at results?

- Script: `scripts/bench_doctor.py` (`packet`, `panel`, `cluster`, `report`)
- Outputs (gitignored): `results/bench_doctor/packet.md`, `replies.jsonl`, `clusters.json`,
  `plausibility.json`, `synthesis.json`
- Trace: `results/traces/bench-doctor-20261018/calls.jsonl`

## Bottom line

1. **The benchmark ran every operator at about a third of its budget.** This was verified in code
   and run files, and only Opus found the actual cause. `LLM.chat` reserves the worst case before
   each implementer call: 16,000 tokens × $10/M × 1.25 ≈ **$0.20**. A real call costs about
   **$0.02**. Under a $0.30 cap, `Ledger._check` refuses new calls once about $0.10 is spent, and
   `run_jobs` then stops. All 9 operators stopped between $0.099 and $0.135, with 4–8 candidates
   each (the `max_candidates` limit is 12).
2. **The headline comes from one lucky draw.** Plain's valid dev scores are 0.5666, 0.6383,
   0.6752 and 0.6752, a median of 0.657. Blind_spot's are 0.5842–0.5966, a median of 0.594.
   Blind_spot is the more reliable operator. Plain wins because of one tail draw.
3. **The blind-spot families start behind on this workload, and the template stops them from
   catching up.** The traces are stationary Zipf traffic, plus a scan over the same key range
   every 10k requests, plus 50% one-hit keys. Persistent frequency history plus a probation tier
   for first-seen keys fits this almost perfectly, and that is what plain/004 is. The blind-spot
   families are mostly recency- or reuse-based. `IMPLEMENT_TEMPLATE` ("Implement this approach
   faithfully … within the family") also steers the implementer to textbook versions. For example,
   blind_spot/001 bounds its history at `3*capacity` and caps frequency at 15 (lines 39 and 120).
   plain/004 beats it on all 6 dev runs.
4. **Retries mostly reproduce the same program.** LRU-K and LIRS land on exactly 0.5966 eight
   times across operators. Five W-TinyLFU or LFU attempts land on exactly 0.6752, which is dev LRU.
   Round-robin resampling of the same prompt adds almost no diversity.
5. **Dev selection is not the problem.** For the 10 picks scored on both splits, |dev − held-out|
   is at most 0.0013. Gemini's claim of dev overfitting is contradicted by this.

Taken together, the measured gap (0.5667 vs 0.5829) is not evidence against the thesis yet. The
pipeline also gives blind spots no route to the hybrid that wins. The thesis may also simply not
hold on this workload; several reviewers (Opus, Sonnet, nemotron, stepfun, kimi) said so.

## Panel

Each model received the same packet of about 31.5k tokens and one of three lenses:
(a) prompt and generation design, (b) selection, budget and evaluation mechanics, (c) what ADHD,
FunSearch, AlphaEvolve/OpenEvolve and ShinkaEvolve do better. The packet held the operators,
reality.py, the bench.py excerpts, literature/consensus-map (`consensus.py`) signatures, the task prompt, the evaluator,
blind spots and the consensus map, a per-candidate table for all 9 operators, per-trace dev scores, the
best plain and blind_spot code, and ADHD's README, how-it-works, engine.ts, frames.ts and types.ts
(clone @7b91ce6, read as text only). Calls used `Session(budget_usd=8.0).llm.chat`, temperature 0.7.

| model | lens | cost $ | status |
|---|---|---|---|
| qwen/qwen3.8-flash | a | 0.0224 | partial: cut off 3 times on length, repaired; 4 fixes, no ADHD lessons |
| mistralai/mistral-small-2603 | b | 0.0065 | ok, weak (see below) |
| meta-llama/llama-4-maverick | c | 0.0110 | ok, thin (2 fixes) |
| deepseek/deepseek-v4.1-flash | a | 0.0337 | **failed**: length 3 times, 155 chars of content; replaced |
| z-ai/glm-5.3 | b | 0.1440 | **failed**: empty content 3 times (reasoning used the whole cap); replaced |
| nvidia/nemotron-3-super-120b-a12b | c | 0.0038 | ok |
| amazon/nova-2-lite-v1 | a | 0.0124 | ok |
| cohere/command-a-plus | b | 0.0374 | ok, one factual error |
| ibm-granite/granite-4.2-8b | c | 0.0099 | ok on 3rd attempt, thin |
| minimax/minimax-m3 | a | 0.0315 | ok, detailed |
| moonshotai/kimi-k2.5 | b | 0.0293 | ok |
| tencent/hy3 | c | 0.0149 | ok |
| openai/gpt-6-luna-pro | a | 0.0137 | ok, careful |
| google/gemini-3.5-flash-lite | b | 0.0146 | ok, contains out-of-scope fixes |
| bytedance-seed/seed-2.0-lite | c | 0.0147 | ok |
| xiaomi/mimo-v2.6-flash (replacement) | a | 0.0061 | ok |
| stepfun/step-3.7-flash (replacement) | b | 0.0489 | partial: repaired truncation, no experiment field |
| anthropic/claude-haiku-5.5 ×3 | a,b,c | 0.0434 | ok (lens c needed 3 attempts) |
| anthropic/claude-sonnet-5.5 ×3 | a,b,c | 0.8683 | ok (a, b needed a 12k retry) |
| anthropic/claude-opus-5.5 ×3 | a,b,c | 2.6097 | ok (all needed retries; c needed 16k) |
| clustering (claude-sonnet-5.5 ×3) | — | 0.2348 | — |
| **total** | | **$4.211** | under the $8.00 cap |

Cheap models cost $0.455 in total, including the two failures. **Deviation from the spec:**
`max_tokens=6000` counts hidden reasoning. At that cap, Sonnet and Opus spent most of the tokens
thinking and their JSON was cut off before `fixes`. Incomplete replies were re-asked with 12k and
then 16k under new cache keys (`bench-doctor:{lens}:retry`, `:retry2`), and the most complete
reply was kept. The retries cost $2.16.

**Weak or wrong replies:**
- mistral and cohere both state that LRU-K's 0.5966 is "identical to LRU". It is not: dev LRU is
  0.6752.
- mistral proposes adding checks inside the evaluator, giving only blind_spot a $0.60 budget, and
  giving only blind_spot a stronger implementer. All three are out of scope or unfair.
- gemini proposes averaging over more dev seeds, which changes the evaluator's data, and ensemble
  selection. It also claims dev overfitting, which the data contradicts.
- nemotron, cohere and stepfun propose steering generation toward the plain winner's trick or
  toward high-consensus-share families. That is peeking, or it defeats the test.
- Sonnet attributes the underspend to `max_candidates`. The real cause is the reservation.
- llama and granite are generic.

## Method

Clustering: one Sonnet call per section (temperature 0) grouped all items into themes. Support is
the number of **distinct models** behind a theme, out of 18. A model that made 3 calls counts once.
Plausibility was assigned by the synthesizer after checking each claim against the run files, on
this scale:

| score | meaning |
|---|---|
| 0 | cheating, peeking, or contradicted by the data |
| 1 | weak or off-target |
| 2 | plausible but not verified |
| 3 | verified, or mechanically certain |

Rank = support × plausibility. The notes behind each score are in
`results/bench_doctor/plausibility.json`.

## Top diagnoses

| rank | diagnosis | support | plaus. | evidence |
|---|---|---|---|---|
| 1 | Degenerate or converged candidates, no dedup | 14 | 3 | 0.5966 ×8 (LRU-K/LIRS), 0.6752 = dev LRU ×5 |
| 2 | Budget use and allocation: ~40% of the cap spent, round-robin gives 1–2 draws per family | 13 | 3 | Worst-case reservation of $0.20 per call vs a $0.30 cap (root cause from Opus b/c; verified) |
| 3 | The "faithful within the family" template blocks hybrids | 12 | 3 | `reality.py IMPLEMENT_TEMPLATE`; the loosest prompts gave the best non-plain scores (adhd_frames/000 0.5758, blind_spot_contrast/001 0.5793) |
| 4 | Blind-spot families are poorly matched to the workload; absence from the consensus is not a quality signal | 11 | 3 | Stationary Zipf, repeated scans, 50% one-hit keys; blind_spot/001 truncates history |
| 5 | Selection noise: a max over 4–8 draws, one repeat | 10 | 3 | Plain median 0.657 vs blind_spot 0.594 |
| 6 | Contrast and feedback do not beat their controls | 11 | 2 | 0.5793 vs 0.5832 and 0.5883 vs 0.5896 (n=1, unreadable). The contrast list also tells the model to avoid persistent frequency, the winning ingredient |
| 7 | Family descriptions are too abstract and carry parameters that break at small caches | 9 | 2 | LIRS's "~1% cold resident share" is about 0 slots at capacity 47 |
| 8 | Complex families are hard to implement zero-shot | 6 | 2 | LeCaR 205 lines vs a 53-line winner, but LeCaR was still blind_spot's best |

Lower ranked: metric validity (6 models), no evolutionary loop (5), IndentationError with no
repair (4), adhd_frames evidence (3).

## Top 8 fixes, with code targets

| # | fix | where | support | plaus. |
|---|---|---|---|---|
| 1 | **Allow adaptation and hybrids around a family core.** Replace "Implement this approach faithfully and make it as strong as you can within the family" with "Use this family's decision rule as the core; you may add standard components (history of evicted keys, a probation queue for first-seen keys, aging, admission) if they help on the workload described above." Ship it as a **new arm**, keep the old template as a control, and report purity. | `diverge/reality.py` `IMPLEMENT_TEMPLATE`, `implementation_prompt` (new flag or operator) | 10 | 3 |
| 2 | **Syntax gate and one repair call.** Run `textwrap.dedent`, tab normalization and `compile()` on extracted code. On a SyntaxError, make one billed repair call with the parser error. Apply to every operator. | `diverge/sample.py` `extract_code`; `diverge/operators/__init__.py` `OperatorContext.job` | 7 | 3 |
| 3 | **Give each retry a different variant.** On visit r>0 to a family, append one of a fixed, family-agnostic list of design axes (history: none / bounded / unbounded with decay; treat first-seen keys specially; isolate scans; simplest O(1) variant), or rotate the named sub-algorithm (LRU-2 vs LRU-K, LIRS vs CLOCK-Pro, SIEVE vs CLOCK). | `diverge/operators/blind_spot.py` `jobs`, `literature_random.py`, `literature_all.py`; a helper in `reality.py` | 6 | 3 |
| 4 | **Evolutionary refine stage** (FunSearch/OpenEvolve style). After a seeding wave, show the top 2 dev parents (code plus per-trace dev miss ratios via `blind_spot_feedback.gaps`) plus one "inspiration" program from another family, and ask for an improved program. Families act as islands. Needs **matched `plain_evolve` controls**. | new `diverge/operators/evolve.py`; `registry()` in `diverge/operators/__init__.py` | 6 | 3 |
| 5 | **Spend the budget.** Reserve on the expected cost (the running mean of this run's completed calls, with the worst case only for the first call). Retry a refused singleton once after in-flight calls settle instead of breaking. Log unspent budget. Also report an equal-N mode (for example, 16 implementer calls). | `diverge/llm.py` `LLM.chat` / `Ledger.reserve`; `diverge/operators/__init__.py` `run_jobs` (`if len(wave) == 1: break`) | 6 | 3 |
| 6 | **Cross-family hybrid operator.** Build the prompt as "Start from base family `<top consensus family>`; add this component from `<blind-spot family>`", using only the consensus map and the taxonomy (no run results). Keep pure blind_spot as a control. | new `diverge/operators/blind_spot_hybrid.py` | 5 | 3 |
| 7 (tie) | **Successive halving across families.** Round 0: one draw per family. Then give the remaining draws to the top half by dev score. Stop a family after two tied scores (within 1e-4) or once it is 0.02 behind. Use the `on_done` hook and the `None` barrier, as `blind_spot_feedback` does. | `diverge/operators/blind_spot.py`, `literature_*.py` `jobs` | 7 | 2 |
| 7 (tie) | **Richer dev feedback.** Show all 6 per-run dev gaps against the run's **best program**, not against LRU (which every candidate already beats). Include the run-best code as an inspiration. Drop "keeping the family's mechanism" in favor of "keep it as the core". Keep the sham control. | `diverge/operators/blind_spot_feedback.py` `note_for`, `NOTE_VALID`, `_jobs` | 7 | 2 |
| 7 (tie) | **Family mechanism specs** (state / on_hit / on_miss / evict / scaling at tiny capacity), generated by the taxonomy builder. Drop benchmark claims such as "outperforms ARC". | `diverge/literature.py` `build_taxonomy` (TAXONOMY_PROMPT) | 7 | 2 |

Next, outside the top 8: a behavioral dedup and degeneracy gate. It would hash per-trace dev
vectors and run the existing `evaluator.probe()` on tiny fixed synthetic traces to flag
LRU-equivalent code (4 models, plausibility 3; pairs naturally with fix 3). Also multi-seed
distributional reporting: a bootstrap of the expected best-of-k, medians, and at least 5 paired
seeds in `diverge/bench.py` `run_metrics`/`aggregate` (3 models, plausibility 3). This is required
before any verdict.

**Rejected:**
- Injecting plain/004's mechanism (persistent frequency plus a one-hit FIFO tier) into prompts or
  code (3 models). This is peeking at this run's winner.
- Weighting families by consensus share (part of a 5-model cluster). It defeats the blind-spot test.
- More or different dev traces, and ensembles. These change the evaluator's data or the artifact.
- A stronger implementer or a bigger budget for blind_spot only. Unfair.

## ADHD lessons

ADHD @7b91ce6 runs reframe → diverge → **score/critic → cluster → deepen**. DIVERGE's `adhd_frames`
copies only the diverge step, as its own docstring says.

| lesson | support | plaus. |
|---|---|---|
| **No critic gate.** ADHD scores every idea for novelty, viability and fit (0.35/0.4/0.25), flags traps and prunes before investing. DIVERGE implements ideas in generation order. 3 of the 4 adhd_frames candidates were the same "sacrificial lamb FIFO" idea. | 15 | 3 |
| **No deepen or refine step.** ADHD expands the top-K into sketches with a named risk and a first step. DIVERGE hands the implementer one sentence or one line of mechanism. | 9 | 3 |
| **No anchor stripping.** ADHD's REFRAME separates real constraints from incidental ones. DIVERGE's "faithfully" instruction does the opposite for blind spots: it anchors on literature parameters (bounded ghosts, a 1% cold share). | 13 | 2 |
| **Generator and critic separated by construction.** DIVERGE keeps the divergent prompt but drops the convergent half. Without the viability gate, "ban the first 3 ideas" pushes away from frequency counting, which is near-optimal here. | 11 | 2 |
| **Recombination.** `deepenIdea` receives sibling ideas to build hybrids. DIVERGE never combines families. The best programs (plain/004, adhd_frames/000) are hybrids. | 5 | 3 |
| **Coverage.** Under a tiny budget, DIVERGE's lazy position-major loop built only the position-0 ideas from 4 frames. The wild frame was never implemented. | 4 | 2 |
| **Caveat.** ADHD's own evidence is LLM-judged breadth and novelty on 6 open-ended problems, not executable scores. Borrow the critic and deepen stages, but ablate them against DIVERGE's real evaluator. | 2 | 3 |

## Dissenting views

- **Is the thesis wrong here, or is the pipeline?** Opus, Sonnet c, nemotron, kimi and stepfun
  argue that the blind-spot families are structurally weaker on stationary Zipf, so pipeline fixes
  may only close the gap by turning blind spots into hybrids built on a consensus core. That would weaken
  attribution. Others (qwen, luna, mimo) put the blame mostly on the template and allocation.
- **Contrast.** Opus and Sonnet want it dropped or made combinational ("you may reuse these;
  must also include X"), because it currently forbids the winning ingredient. Amazon wants it
  targeted at weak dev cases instead.
- **Feedback.** Opus c reads the single own-feedback step (LRU-K 0.5967 → 0.5883, the only LRU-K
  draw to break the 0.5966 cluster) as a signal. Sonnet b calls feedback against sham unreadable
  at n=1. Both are right: the effect is suggestive and untested.
- **adhd_frames.** Sonnet notes it reached the best non-plain score from 4 candidates. Opus b notes
  its best idea is essentially 2Q/S3-FIFO, a consensus family, so "divergence" re-converged on the consensus.

## Anti-overfitting protocol for the follow-up experiment

1. **Pre-register before running.** Freeze the operator code, prompt texts, design-axis list,
   seeds, k and metrics, and record the commit hash. Nothing new is derived from this run's winning
   code. The design axes come only from the public task prompt.
2. **Fix the accounting first** (fix 5), then rerun the existing 9 operators unchanged at 5 or
   more paired generation seeds. This tells us whether plain's lead survives beyond one draw before
   anything else changes.
3. **Use dev only for every decision.** Selection, allocation, feedback, dedup and evolution all
   see dev scores and dev per-run details only. Held-out is read **once**, for each seed's
   pre-registered dev pick, at the end. No iteration after reading it.
4. **Give every new mechanism a matched control.** Pair evolve with plain_evolve, the variant
   prompts with plain variants, and the hybrid operator with pure blind_spot. Keep contrast_random,
   the sham feedback and the old template. Apply the syntax repair to all operators.
5. **Report distributions, not a max.** Report the per-seed dev-selected best, bootstrap expected
   best-of-k for k=1..N, the median valid dev score, the share beating dev LRU, the degenerate rate,
   spend against the cap, and the dev → held-out gap (currently ≤ 0.0013) as an overfitting
   monitor. The verdict is "inconclusive" unless the ordering holds across seeds.
6. **Track attribution.** For hybrids and evolved programs, report purity (the labeled family vs
   the intended one) and which family's component survives in the final pick.
7. **Replicate** on a second registered task (bin packing) before claiming the thesis holds or
   fails.
