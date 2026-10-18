# Overlooked-ideas phase: protocol (written before any live call of this phase)

Branch `overlooked-ideas`, started from `model-review-changes` at `309f740`. Spend before this
phase: $8.14, all in DIVERGE's cache.

**Budget for this phase: $20.** DIVERGE's own global cap goes to $28.00, so the code itself refuses
to spend past it. I stop and ask before this phase goes past $20. Every call is traced to
`results/traces/overlook-20261018/`.

**Models:**
- Samplers: the existing five.
- Labeler: `anthropic/claude-opus-5.5`, used for taxonomy labels, so they stay comparable to
  the maps.
- New extractor: `anthropic/claude-haiku-5.5`, for reading consensus-map code in Step 1. Cheap; it never
  generates solutions.
- Implementer for Step 2: `google/gemini-4-argon`.

**Consensus data:** the existing N=20 maps, 100 samples per task (95 valid for bin packing, 76 for
cache eviction).

## Step 0: paraphrase test (estimate about $1.8)

- **Paraphrases.** The labeler writes 3 meaning-preserving paraphrases of each task's
  `prompt.md`. Each must keep the interface, the code signature, every number, and the scoring
  rule. I read all six before sampling; any paraphrase that changes the task is rewritten once
  and the rewrite is recorded.
- **Samplers.** `anthropic/claude-sonnet-5.5` and `openai/gpt-6.1-sol`: different companies,
  and the cheapest well-behaved pair in the map run. 5 samples per paraphrase per model, at
  temperature 1.0, with the same max_tokens as the map.
- **Labels.** Labeled with the map's existing taxonomy and the same labeler.
- **Resampling baseline, at no cost.** Those two models' existing 20 original-prompt samples
  per task are split by sample index into 4 groups of 5 (indices 0-4, 5-9, 10-14, 15-19). One
  "original group" is the 2 models x 5 samples = 10 samples.
- **Statistic.** Total variation distance (TVD) between family-share distributions of
  10-sample groups. Unlabeled samples are excluded. All `emergent:*` labels, on both sides, are
  counted as one "emergent" bucket, because emergent names are merged per labeling run and
  would not line up across runs.
  - R = the 6 pairwise TVDs among the 4 original groups (resampling noise).
  - P = the 12 TVDs between each of the 3 paraphrase groups and each of the 4 original groups.
- **Decision rule, fixed now:**
  - If mean(P) <= max(R), the consensus is **robust to wording**: paraphrasing moves it no more than
    resampling does.
  - If mean(P) > max(R), the consensus is **partly a prompt effect**. The README will then say so.
  - With 10 samples per group this can only detect large effects, and the report will say that.
- The brief stated the rule the other way round; this reading is the logically correct one.

## Step 1: overlook.json per task (estimate about $2.5)

1. **Decision-input audit.** I list the inputs available at decision time by hand from
   `prompt.md` and the evaluator's call signature, and fix that list in code before looking at
   any sample code. Each valid sample gets a static AST analysis: does the input flow into the
   return value? An input "flows" if a backward slice from the return or `evict()` value reaches
   it, through assignments, loops and calls. Inputs that can't be decided statically are marked
   "ambiguous". The extractor resolves them in batches of up to 8 samples per call. Inputs are
   ranked by usage share; used by 10% or less of valid samples = **overlooked lever**.
2. **Shared assumptions.**
   - The extractor reads up to 12 valid samples from the top-3 families, chosen deterministically
     (first by sample ID within each family), and proposes assumptions they share.
   - A second extractor pass checks each assumption against each of those samples (yes/no plus
     evidence). Only assumptions held by **every** checked sample count as "shared".
   - Each shared assumption is classed **hard** only if it cites prompt text that is found
     **verbatim** in `prompt.md`. Otherwise it is **incidental**. A quote that doesn't match the
     prompt exactly can't make an assumption hard.
3. **Component grid.**
   - Fixed components: for bin packing, scoring term, tie-break, state kept, and new-bin rule.
     For caching: admission, eviction score, state kept, and aging.
   - The extractor assigns each valid sample a short value per component. One canonicalization
     call merges synonyms into a small vocabulary per component. The literature taxonomy
     families are decomposed the same way.
   - **Gap cells:**
     (a) combinations that literature families occupy and no consensus-map sample does;
     (b) pairs of component values that each appear in at least 2 samples, are never seen
     together, and are judged coherent by one extractor call that sees only the values.
   - Evidence is recorded for every cell.
4. **Minority report.** From the existing labels and the component values: families or
   component values produced by only 1-2 of the 5 sampler models and absent from the other 3-4.
   No API cost.
5. **Latent tail.**
   - Each of the 5 sampler models answers 2 prompts per task (20 calls): (A) shown the consensus
     top-3 families with a strict no-repeat instruction, asking which OTHER approaches it knows
     for this exact problem; (B) the same question without the list (the control).
   - Answers are matched to the taxonomy by the extractor. Answers that match no taxonomy family
     are checked with at most 3 Exa searches per task. Only items with a matching taxonomy
     family, a retrieved source, or a precise mechanism (state plus decision rule) count as
     verified.
   - Reported: verified non-consensus approaches per model under A vs B.
   - "Showing the consensus unlocks knowledge" requires A > B for at least 4 of the 5 models;
     otherwise the result is "no evidence".

Every overlook.json item carries evidence: sample IDs, models, code lines, sources, quotes.
**Checkpoint:** I show overlook.json for both tasks and stop before any Step 2 code or calls.

## Step 2 (fixed now; parameters may be amended once, before any Step 2 live call)

- **Operators and controls** (same implementer, base template, evaluator and budget per
  operator; one overlook item per prompt):
  - unused_signal vs unused_signal_control (a random input the consensus already uses).
  - assumption_break vs assumption_break_control (a random assumption from non-consensus families).
  - component_gap vs component_gap_control (a random occupied cell).
  - minority_report vs minority_report_control (a random majority mechanism).
  - blind_spot+latent.
  - References: plain, blind_spot, literature_all, adhd_frames.
- **Protocol:** 3 repeats, dev selection, held-out reporting, equal budget per operator.
- **Primary metric:** held-out best score.
- **Secondary metrics:** faithful valid candidates per dollar (faithfulness from AST
  input-usage checks, plus fingerprints for bin packing); behaviorally distinct valid
  candidates; useful discoveries (held-out beats the best baseline).
- **"Works"** only if the operator beats its own control AND plain on held-out best score in
  every repeat. Otherwise the result is "inconclusive" or "loss", reported plainly.
- **Budget formula:** remaining = $20 - Step 0/1 actual spend - $1 reserve; online_bin_packing
  first, cache_eviction only if each operator would still get at least $0.30. Operators run in
  parallel. Each keeps its own run budget, so this does not change fairness.
