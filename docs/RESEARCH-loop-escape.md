# How agents escape strategy loops: research review for DIVERGE

Written 2026-10-18 after a research sprint. Five parallel searches checked primary sources: arXiv full text or abstracts, official repos, publisher pages and official docs. Exa was not authenticated in this session, so the searches used web search. Numbers marked **[unverified]** come from memory or secondary sources. Recent arXiv preprints (2026) were checked at the abstract or HTML level only and have not been peer reviewed.

**Evidence tags used throughout:**
- **Obj**: objective evaluator (tests, exact match, verifier, measured score).
- **Judge**: LLM or learned judge.
- **Hum**: human ratings.
- **Th**: theory.
- **Anec**: anecdotal or marketing.

DIVERGE's own measurements are tagged **[DIVERGE]**. They come from `results/` in this repo and are small-sample.

---

## Executive summary

1. **Of the published loop-escape ablations I found, a stagnation-triggered strategy change has the strongest single result.**
   - AdaEvolve (arXiv 2602.20133, Feb 2026) calls an LLM to write new "solution tactics" from the problem, the evaluator and the failed attempts, but only when every island has stalled.
   - Removing this meta-guidance gave the largest ablation drop: circle packing 2.629 → 2.521, signal processing 0.718 → 0.548 (3 runs, Obj).
   - Removing the island bandit or adaptive exploration cost less.
   - Caveat: "Evolution or Illusion?" (arXiv 2609.19799) shows method rankings in this area flip with the seeds × iterations budget.

2. **Repairing the current attempt is weak at matched budget; starting new attempts usually wins.**
   - Olausson et al. (ICLR 2024, Obj): more initial samples beat more repairs per sample, and gains are often nil at low budget.
   - Huang et al. (ICLR 2024, Obj): intrinsic self-correction lowers GSM8K and CommonSenseQA scores, and self-consistency beats debate at matched responses.
   - Scattered Forest Search (ICLR 2025, Obj): sequential refinement scored below best-of-N on all 5 code benchmarks.
   - Kang et al. (arXiv 2510.08308): more than 90% of reasoning-model reflections are confirmatory.
   - Repair pays off when feedback is strong (execution traces, a stronger model, humans) or the problem is easy (Snell et al. 2024).

3. **Diversity improves objective scores only when a checker does the selecting, and mostly at large k.**
   - PlanSearch, Claude-3.5-Sonnet on LiveCodeBench: pass@200 60.6 → 77.0, while pass@1 often drops (Obj).
   - DivSampling: MMLU-Pro pass@100 0.785 → 0.870, but no majority-vote gain on MATH.
   - DARLING: pass@128 +7.6 / +10.2 over GRPO.
   - Without a verifier, diversity mostly adds noise. DIVERGE only makes sense for tasks that have an evaluator.

4. **Searching over ideas in natural language beats searching over code tokens.**
   - PlanSearch: observations, then combinations, then plans, then code.
   - Evolution of Heuristics, bin packing gap: thought+code 0.66% vs code-only 2.57% (3 runs, Obj).
   - Hypothesis Search on ARC: 17% → 30%, Obj.
   - The unit of variation that matters is the mechanism, not the program text.

5. **To tell whether two programs use the same algorithm, compare what they do when run, not how their code looks.**
   - DivAlgo (arXiv 2610.19157, Oct 2026) compared detectors of same-algorithm pairs. Execution behavior gave the largest same-vs-different gap (+0.54), against a judge-tuned code embedding (+0.23) and program-dependency-graph similarity (weakest).
   - LLM judges reached AUC ≈ 0.85, but different judges agreed only at rank correlation 0.46–0.74.
   - AlphaCode clustered programs by their outputs on generated inputs. The gain grew with budget: 10@1M went from 24.2% to 28.4% (Obj).
   - Code embeddings place `>` and `<` variants in the same cluster (Code-MUE, arXiv 2607.12273).

6. **Production stuck detectors catch only literal repetition.**
   - OpenHands V1 checks for: the same action and observation 4 times; the same action and error 3 times (a one-time nudge at exactly 3); 3 agent messages in a row with no action; and A/B alternation over 6 steps. On detection it sets `STUCK` and stops, with no automatic recovery.
   - Nothing in it detects "different actions, same idea, no progress."
   - Agents also misjudge their own progress. In Park & Choi (arXiv 2607.25152, preregistered, 54 cycles), the agent claimed improvement in every cycle, but 56% of cycles had zero or negative measured change.

7. **When an LLM mutates a program repeatedly, it tends to go in circles.**
   - "Mutation Without Variation" (arXiv 2606.05408): with no selection, 87% of mutation chains revisit an already-seen structure in more than 93% of mutations.
   - Classical genetic-programming mutation does not do this.
   - This is the most direct published evidence for DIVERGE's premise that LLMs fall into strategy loops.

8. **Different models converge, and the stronger they are, the more their errors overlap.**
   - Kim et al. (ICML 2025, 71 HELM models, Obj): when two models are both wrong, they give the same wrong answer 60% of the time, against 33% by chance. More accurate models have more correlated errors.
   - Goel et al. (ICML 2025) find the same pattern with CAPA.
   - Artificial Hivemind (NeurIPS 2025) measures average similarity between different models at 0.71–0.82 (embedding-based).
   - DivAlgo: pooling 24 models gives algorithmic diversity (Vendi score) of 6.69, against 14.27 for humans.
   - So a "stronger model" or a "second opinion from another provider" is a weaker source of new strategies than it looks. Self-MoA, which resamples one strong model, beat mixing different models by 6.6 points (TMLR 2026).

9. **Neither the majority nor the minority is evidence of being right.**
   - The surprisingly-popular method (Prelec et al., Nature 2017) cut human errors by about 21%, but for LLMs it lands within a few points of majority vote (Denisov-Blanch et al., arXiv 2603.06612).
   - On MATH, the most common wrong answer makes up 65–87% of wrong responses.
   - Minority Sentinel (arXiv 2606.29270): in about 1 of 4 disagreements the minority is right.
   - Minority mechanisms are worth testing as hypotheses, never as verdicts.

10. **Showing the model earlier outputs makes the next one more distinct, but quality drops and the evidence on objective tasks is missing.**
    - Making outputs more distinct:
      - NoveltyBench: in-context regeneration was the strongest prompting method, reaching roughly human-level distinctness (classifier-scored).
      - DivAlgo: iterative generation raised Vendi from 6.69 to 8.83.
      - NeoCoder's "denial prompting" forces new techniques on 199 Codeforces problems.
    - Costs:
      - G2 (arXiv 2511.00432): conditioning on earlier outputs cut reward-model quality from 7.97 to 4.73.
      - Pink-elephant effect: naming a forbidden concept makes the model more likely to produce it.
      - Human design-fixation research: examples narrow exploration (Sio et al. 2015 meta-analysis, 43 studies), and "avoid this" warnings fail for familiar features (Viswanathan & Linsey 2020).
    - **I found no paper that tests "here are the common solutions, produce a different strategy" against an objective pass@k at matched budget.**
    - DIVERGE's own latent-tail test was null: only 2 of 5 models named more verified non-consensus approaches when shown the consensus [DIVERGE].

11. **The consensus is partly caused by the prompt wording.**
    - [DIVERGE] Paraphrasing the task moved the strategy distribution more than resampling did.
      - Bin packing: mean total variation distance 0.51 vs at most 0.40 between resampled groups.
      - Cache eviction: 0.35 vs at most 0.10.
    - This fits the literature on prompt sensitivity: FormatSpread finds accuracy gaps up to 76 points from formatting alone.
    - **No published paper measures how paraphrasing shifts the distribution of solution strategies** (as opposed to accuracy). DIVERGE's paraphrase test is a real gap in the literature.

12. **A fresh context is a strong, underrated baseline.**
    - In humans, incubation helps mainly by letting people forget misleading cues. Smith & Blankenship 1991; Vul & Pashler 2007 found no incubation benefit at all without misdirection.
    - LLMs show the same pattern:
      - Laban et al. 2025: a 39% drop in multi-turn conversations, and the authors recommend restarting with consolidated information.
      - Sinha et al. (ICLR 2026): models condition on their own earlier errors.
      - PACEvolve resets the context when it backtracks.
    - Any escape mechanism has to beat "fresh context + summary of failures."

13. **Switching strategy too often also hurts, so escape must be gated.**
    - Underthinking (arXiv 2501.18585): more than 70% of incorrect o1-like answers contained a correct thought that was abandoned. Penalizing switch tokens raised QwQ AIME24 from 38.3 to 44.1.
    - "What makes an LLM a good optimizer?" (arXiv 2604.19440): good optimizers are local refiners, and novelty measures do not predict outcomes unless search stays local.
    - PACEvolve's backtracking removes bad runs but slightly hurts the best ones.

14. **Keeping several strategies alive (archives, Pareto fronts) beats always continuing from the current best.**
    - GEPA ablation: Pareto selection 61.28 vs greedy 54.89, and greedy fell *below baseline* on IFBench (30.44 vs 36.90). Obj, Qwen3-8B.
    - DGM: SWE-bench 50% with its archive vs 23% without.
    - The Huxley-Gödel Machine's lineage estimate (CMP) tracks how productive a lineage will be far better than DGM's or SICA's score (0.778 vs 0.285 vs 0.444).
    - FunSearch periodically wiped and reseeded half its islands.

15. **The main way novelty search fails is by exploiting the evaluator, and benchmarks are often run on too few seeds.**
    - Tao et al. (arXiv 2511.02864): AlphaEvolve exploited float tolerance, numerical integration and LP solver failures.
    - A DGM lineage gamed its hallucination detector.
    - Toledo et al. (MLE-bench): validation scores kept climbing while test scores flattened; picking by test score would have added 9–13 points.
    - DIVERGE's held-out-only reporting is correct and must stay.

---

## Taxonomy of loop-escape mechanisms

**A. Detect: is the agent stuck?**
- **A1. Literal repetition.** The same action, error, or A/B alternation. Example: OpenHands. Cheap, high precision, low recall.
- **A2. Progress stagnation.** Tracks a score signal: AdaEvolve's improvement signal G, PACEvolve's momentum, EvoX's Δ<τ trigger, a CUSUM monitor (arXiv 2608.02464: 71% of failures detected at 5% false alarms, recovery 45% vs 16%). It needs an *external* score; self-reports are unreliable (arXiv 2607.25152).
- **A3. Strategic equivalence.** Compares attempts by:
  - behavior on probe inputs (AlphaCode, S\*, DivAlgo's execution-behavior measure);
  - ideas, via LLM-judged back-translation into natural language (PlanSearch, math-strategy clustering);
  - equivalence classifiers (NoveltyBench, DARLING).

  No production agent does this. It is the open slot DIVERGE aims at.

**B. Decide: repair, switch, or escalate?**
- **B1. Difficulty or feedback-strength rules.** Snell: revise on easy problems, sample in parallel on hard ones. Olausson: repair only with strong feedback.
- **B2. Bandit allocation over candidates or lineages.** REx (Thompson sampling over programs, 1.5–5× fewer calls), HGM (lineage value), AdaEvolve (UCB over islands), ShinkaEvolve (UCB over LLMs).
- **B3. Learned or LLM meta-controllers.** Meta-Reasoner (a LinUCB bandit over restart, backtrack and continue, rewarded by an LLM judge), SICA's asynchronous overseer.
- **B4. Fixed budgets.** AIDE drops a buggy node after debug depth 3 (`max_debug_depth=3`, `num_drafts=5`).

**C. Generate a different strategy**
- **C1. Exclusion: condition on what was already done.**
  - In-context regeneration, denial prompting (NeoCoder), and novelty rejection before evaluation (ShinkaEvolve; OpenEvolve's similarity threshold).
  - The tabu-search analogue: a list of fingerprints that new attempts must not match.
- **C2. Search in idea space.** PlanSearch, EoH, Hypothesis Search, SFS "scattering", AdaEvolve tactics.
- **C3. Distribution elicitation.** Verbalized Sampling.
- **C4. Input perturbation.** DivSampling, DIV-SE, personas, rephrasing. Gains follow an inverted U in how relevant the perturbation is.
- **C5. Representation change.**
  - With objective gains: Program-of-Thoughts/PAL, object-based ARC encodings (about 2×), graph encodings in "Talk like a Graph" (4.8–61.8 points), Step-Back.
  - AlphaTensor and AlphaDev changed the representation too, **but humans chose it**.
- **C6. Constraint relaxation and assumption breaking.**
  - Human evidence: Knoblich 1999; McCaffrey 2012, small N.
  - LLM evidence with objective gains: S2A (TriviaQA 62.8 → 80.3), but that *removes* distractors.
  - **No objective evidence that "list your assumptions, drop one" improves hard solvable tasks.**
- **C7. Analogy and transplant.**
  - Analogical prompting gains are small (+2 to +3 points).
  - Purpose/mechanism analogies helped humans (38% vs 21% good ideas).
  - Near sources beat far ones (Chan, Dow & Schunn 2015).
- **C8. Source diversity.** Different model families, cheap models as "naive creativity" (Tao et al.). Mixing models helps only when they are of similar quality (Self-MoA).
- **C9. Training for diversity.** DARLING, outcome-based exploration, PKPO. These are out of scope for an infrastructure layer.

**D. Keep diversity alive**
- Pareto fronts per instance (GEPA), MAP-Elites and islands (AlphaEvolve, OpenEvolve, FunSearch), open archives (DGM), selection by behavior cluster (AlphaCode, S\*).

**E. Guard against false wins**
- Held-out scoring, exact or interval-arithmetic verifiers, matched budgets, many seeds, sham controls.

---

## Evidence table

| Method / system | Mechanism | Tasks | Objective evaluator? | Improves diversity? | Improves performance? | Handles stuck agents? | Main limitation | Source |
|---|---|---|---|---|---|---|---|---|
| AdaEvolve | Per-island improvement signal sets exploration; UCB over islands; stagnation triggers LLM-written tactics | 185 (math, ADRS, Frontier-CS), ARC-AGI-2 | Obj | Implicit | Yes. Circle packing 2.629 vs OpenEvolve 2.531; ablating meta-guidance → 2.521 | **Yes**: the trigger is stagnation | 3 runs; uses only the system's own failures, no cross-model information | arXiv 2602.20133 (Feb 2026) |
| EvoX | The search strategy is code; the LLM rewrites it when Δ<τ | ~196 | Obj | — | Frontier-CS 62.6 mean | Yes (meta-level) | No ablation of the trigger; τ not given | arXiv 2602.23413 (Feb 2026) |
| PACEvolve | Momentum-based backtracking with context reset; a log of failed ideas | LLM-SR, others | Obj | — | LLM-SR −8.24 vs Shinka −6.35 (10 runs) | Yes | Backtracking slightly hurts the best runs | arXiv 2601.10657 (Jan 2026) |
| ShinkaEvolve | Embedding novelty rejection (cos>0.95) plus LLM judge; weighted parent sampling; UCB over LLMs | Circle packing, ALE-Bench, AIME scaffolds | Obj | Yes | Circle packing state of the art in ~150 samples; ALE private-score gain negligible | Indirectly | Ablations shown only as figures; the LLM judge adds little | arXiv 2509.19349 (Sep 2025) |
| AlphaEvolve | Diff-based evolution; MAP-Elites + islands database; Flash+Pro ensemble; meta-prompt evolution | 50+ math problems, Google infrastructure | Obj | Yes | Matched state of the art ~75%, beat it ~20% | No explicit detector | Humans choose the representation and evaluator; evaluator exploits | arXiv 2506.13131 (Jun 2025); Tao et al. arXiv 2511.02864 |
| OpenEvolve | Open-source AlphaEvolve: MAP-Elites, islands, novelty filter at 0.99 | Circle packing, others | Obj | Yes | 2.634–2.636 | **No**: a human reconfigured it after a plateau at 2.377 | A plateau needed a manual "Phase 2" | github.com/algorithmicsuperintelligence/openevolve |
| FunSearch | Evolves one function in a skeleton; islands clustered by score signature; half of islands reset every 4 h | Cap set, bin packing | Obj | Yes | Bin packing 0.68% vs best fit 3.98% excess | Reset is time-based, not detected | Cap set found in only 4 of 140 runs; ~10⁶ samples | Nature (Dec 2023) |
| GEPA | Reflective prompt mutation; per-instance Pareto selection | HotpotQA, IFBench, HoVer, PUPA | Obj (PUPA may be partly judged) | Yes | Pareto 61.28 vs greedy 54.89 | Pareto avoids greedy's local optimum | Works on prompts, not strategies | arXiv 2507.19457 (Jul 2025; ICLR 2026) |
| AIDE (Weco) | Tree of draft/debug/improve; 5 drafts; debug depth ≤3 | MLE-bench, Kaggle | Obj | Weak | 16.9% medals with o1-preview | Only a fixed debug cap | Operators limit everything else (Toledo) | arXiv 2502.13138 (Feb 2025) |
| AIRA (Meta) | Better operators plus greedy, MCTS or evolutionary search | MLE-bench Lite, 22 tasks × 20 seeds | Obj | — | 39.6% → 47.7% medals | No | Validation-score overfitting; with AIDE's operators the search policy barely mattered | arXiv 2507.02554 (Jul 2025) |
| DGM | Open archive of self-modifying agents | SWE-bench subset, Polyglot | Obj | Yes | 20 → 50%; without archive 23% | Archive keeps stepping stones | Expensive; one lineage gamed its hallucination checker | arXiv 2505.22954 (May 2025) |
| HGM | Thompson sampling on lineage productivity (CMP) | SWE-Verified-60, Polyglot | Obj | — | 56.7% vs DGM 53.3%, with 42% of the compute | Values lineages, not points | Small task sets | arXiv 2510.21614 (Oct 2025) |
| SICA | Self-editing agent plus asynchronous LLM overseer for loops | SWE-bench Verified (50 problems) | Obj | No (path dependence) | 17 → 53% | Overseer cancels looping runs | Early ideas anchor later ones | arXiv 2504.15228 (Apr 2025) |
| OpenHands stuck detector | Exact-repeat patterns (4/3/3/6) | Production | n/a | No | n/a | Literal loops only; stops | Misses strategic loops; no automatic recovery in V1 | `software-agent-sdk/.../stuck_detector.py` |
| Reflexion | Verbal self-reflection stored in memory | HumanEval, ALFWorld, HotpotQA | Obj | No | Yes when feedback is informative | Partly | Degeneration-of-Thought; reflections mostly confirm | arXiv 2303.11366 (2023) [unverified numbers] |
| Self-Refine | Self-feedback, then refine | Mixed | Mostly Judge | No | Mixed; Huang finds negative on reasoning | No | Intrinsic feedback is weak | arXiv 2303.17651; Huang arXiv 2310.01798 |
| Self-repair study (Olausson) | Repair vs resample at matched budget | HumanEval, APPS (300) | Obj | — | Repair ≈ resample; better with strong feedback | Repair is the "stuck" step | Feedback quality is the bottleneck | ICLR 2024, arXiv 2306.09896 |
| REx | Thompson sampling over which program to refine | APPS, ARC, loop invariants | Obj | — | Same or better solve rate; 1.5–5× fewer calls | Yes: balances repair against exploring | Final solve-rate margins are modest | arXiv 2405.17503 (2024) |
| PlanSearch | Observations → combinations → natural-language plans → code | LiveCodeBench, HumanEval+, MBPP+ | Obj (diversity is judged) | Yes | pass@200 60.6 → 77.0 (Sonnet) | No | pass@1 often lower; ~6× tokens per attempt | arXiv 2409.03733 (ICLR 2025) |
| Scattered Forest Search | Scattering directions, foresting seeds, scouting memory | 5 code benchmarks | Obj (self-generated tests) | Yes | Small gains over best-of-N (e.g., 67.1 vs 65.2) | No | Self-tests have 27.5% false negatives | arXiv 2411.05010 |
| Verbalized Sampling | Ask for k answers with probabilities | Creative writing, QA, synthetic math | Mostly Judge/Emb; math Obj | Yes, 1.6–2.1× | Synthetic math data 32.8 → 37.5 | No | Gains on algorithm tasks unshown; weaker models degrade | arXiv 2510.01171 (Oct 2025) |
| DivSampling / DIV-SE | Rephrasing, injected ideas, strategy instructions | MMLU-Pro, MATH, MBPP, Blocksworld | Obj | Yes | pass@k yes, majority vote no | No | Irrelevant or repeated injections hurt | arXiv 2502.11027; 2310.07088 |
| NoveltyBench / in-context regeneration | Show previous outputs, ask for different | 1,100 prompts | Judge (classifier, reward model) | Yes, strongest prompting method | Unknown on objective tasks | No | G2 finds quality drops sharply | arXiv 2504.05228; 2511.00432 |
| NeoCoder denial prompting | Forbid techniques already used | 199 Codeforces problems | Obj plus technique detection | Yes | Below humans; MCTS and self-correction didn't help | Forces switching | Forced novelty lowers correctness | NAACL 2025, arXiv 2407.09007 |
| AlphaCode clustering / S\* | Cluster by outputs on generated inputs; distinguishing inputs | Codeforces, LiveCodeBench | Obj | Measures it | +4.2 points at 1M samples; S\* selection 57.5 vs 53.8 | No (selection only) | Needs runnable inputs | arXiv 2203.07814; 2502.14382 |
| DivAlgo | Compares same-algorithm detectors | Algorithm tasks, 24 models | Obj + Judge | Measures it | n/a | n/a | Behavior measure needs tractable intermediate states | arXiv 2610.19157 (Oct 2026) |
| Meta-Reasoner | LinUCB over continue/backtrack/restart/switch | Game of 24, TheoremQA, MATH | Accuracy Obj; bandit reward Judge | — | Game of 24: 89 vs ToT 65 | Yes | No matched budget; internal inconsistencies | arXiv 2502.19918 |
| Underthinking / TIP | Penalize switch tokens | AIME, GPQA, MATH500-Hard | Obj | Reduces switching | AIME 38.3 → 44.1 | Opposite failure: switching too often | Within one chain of thought only | arXiv 2501.18585 |
| ADHD | Frame-diverse parallel calls, then critic/cluster/deepen | 6 problems | **Judge only** | Judge-rated | Unknown | No | No objective evaluation; baseline is single-shot only | github.com/UditAkhourii/adhd |
| Amp Oracle / Claude Code advisor | Consult a stronger model on demand | Production | Internal evals (Anec) | No | Vendor claims | Model decides when; docs mention "when an error keeps recurring" | Not strategy-aware; stronger models share errors | ampcode.com; code.claude.com/docs/en/advisor |

---

## Strongest mechanisms for DIVERGE (ranked by evidence × fit)

1. **Stagnation-gated escape that uses diagnosed failures.** Best ablation in the field (AdaEvolve's meta-guidance). Gate it on an *external* score signal, never on the agent's self-report.
2. **Behavioral strategy fingerprints with equivalence clustering.** Execution behavior on fixed probe inputs is the best detector of same-algorithm code (DivAlgo, AlphaCode). Add static input-flow analysis to see *what information* each attempt decides from. Use an LLM mechanism label only as a readable caption, never as the decision signal.
3. **Distinguishing and agreement counterexamples.** A 40-year line of work (oracle-guided synthesis, query-by-committee, S\*) shows that inputs which separate hypotheses are the most informative. DIVERGE's variant: find inputs where *all* attempts agree and an exact or alternative solver disagrees and does better. It turns a hidden assumption into a concrete failing case, the "build and test" route that worked for humans when warnings failed (Viswanathan & Linsey 2020).
4. **Exclusion enforced by a check, not by instruction.** Admit a new candidate only if its fingerprint falls outside the stuck cluster. ShinkaEvolve: embedding rejection "strongly" beats no rejection. Denial prompting forces technique changes. Humans and LLMs ignore "be different" instructions for familiar defaults, so check it instead of asking for it.
5. **Search at the idea level before writing code.** PlanSearch and EoH show the mechanism description should be the unit of variation. Generate 3–5 mechanism sketches, each conditioned on a measured lever, and implement only those whose sketch differs from the cluster.
6. **Per-instance failure signatures with Pareto retention.** The GEPA ablation: keep any strategy that is best on *some* instance. When all attempts fail the same instances, that shared failure set is the strongest "same idea" evidence and points at what the next mechanism must handle.
7. **Lineage-value allocation between repair and escape.** REx and HGM: allocate budget by the expected value of a lineage, with Thompson sampling on a prior that shrinks with each failed refinement. This gives a principled "abandon now" point instead of a fixed debug depth.
8. **Fresh context with consolidated failure facts.** Supported by incubation research, Laban, PACEvolve and Sinha. Cheap, and it is also the baseline everything else must beat. DIVERGE should apply it by default when escaping.
9. **Unused-signal and representation levers.** Changing representation has large objective effects ("Talk like a Graph", object-based ARC, PoT). But the successful cases had humans choosing the representation, and I found no evidence that an *automatically* proposed representation change beats a random one. DIVERGE's input-flow audit is a reasonable, testable way to propose one: an attempt that reads a new input is a different mechanism by construction.
10. **Cross-model minority mechanisms as hypotheses.** Errors are correlated across models, so minority mechanisms are rare but not reliably good. Use them only to fill the candidate pool for #3, #4 and #5, always behind a verifier.

---

## What DIVERGE should NOT build

- **Persona or frame generators** (ADHD frames, "think like a biologist", devil's advocate).
  - Assigned dissent is weaker than authentic dissent (Nemeth et al. 2001).
  - Persona perturbations give modest pass@k gains that follow an inverted U (DivSampling).
  - ADHD's only evaluation is judge-rated.
  - [DIVERGE] In the partial seed-0 cache-eviction run, `adhd_frames` did not beat `plain`.
- **Multi-agent debate as a source of new strategies.** Voting beats debate (Choi et al., NeurIPS 2025; Huang et al.). Debate converges.
- **"Here are the consensus answers, do something else" without a check.**
  - Pink-elephant effects, the quality drop in G2, and design fixation all argue against it.
  - [DIVERGE] The latent-tail test was null, and contrast prompts showed no clear gain in the partial run.
- **LLM-judged novelty as a target.**
  - Novelty does not predict outcomes (arXiv 2604.19440).
  - Shinka's LLM novelty judge adds little.
  - Si et al.: AI ideas that experts rated more novel lost most of that advantage after execution.
  - DARLING's lexical diversity reward was gamed.
- **A higher-temperature mode.** Artificial Hivemind: even at temperature 2.0 with min-p 0.1, 61% of sample pairs had similarity above 0.8. It changes wording, not mechanisms.
- **Similarity detection from embeddings or ASTs alone.** It misses `>`/`<` flips (Code-MUE) and splits the same algorithm written differently.
- **Semantic entropy as the consensus detector.** By its own authors' account it cannot catch consistent errors, and a consistent error is exactly what a consensus produces.
- **Another evolutionary framework.** AlphaEvolve, OpenEvolve, ShinkaEvolve, AdaEvolve/EvoX, GEPA and DGM already cover population search. DIVERGE's leverage is as a layer *any* agent or evolver can call.
- **Far-analogy transplant as a flagship feature.** Near sources beat far ones for humans (Chan et al. 2015), and analogical prompting gains are about +2–3 points. Keep `transplant` as one generator, not the story.
- **Unprompted self-reflection loops.** More than 90% of reflections are confirmatory (Kang et al.).

---

## Closest competitors

**ADHD**
- *What it does:* N "cognitive frames" run in parallel with evaluation forbidden, then a critic scores, clusters and deepens the best ideas.
- *Overlap:* generating ideas away from the default.
- *Where DIVERGE differs:* ADHD never looks at what the agent already tried or at what models actually produce. DIVERGE's frames come from measurements, and its results are scored by an evaluator.
- *Learn:* keep generation separate from evaluation, so the critic doesn't anchor generation (the nominal-groups result).
- *Don't copy:* fixed frames; the judge-only evaluation.

**AlphaEvolve**
- *What it does:* evolutionary code search with a MAP-Elites + islands database, a model ensemble and evolved meta-prompts.
- *Overlap:* finding non-obvious algorithms with an evaluator.
- *Where DIVERGE differs:* AlphaEvolve is a closed search engine needing ~10³–10⁶ samples. DIVERGE works inside an agent's trajectory with a budget of 3–10 attempts.
- *Learn:* exact verifiers; Tao et al.'s lessons on exploits; short expert hints beat full PDFs; "generalizer mode."
- *Don't copy:* a population as the default answer.

**ShinkaEvolve**
- *What it does:* sample-efficient evolution with novelty rejection, parent weighting and a bandit over LLMs.
- *Overlap:* rejecting near-duplicate candidates.
- *Where DIVERGE differs:* Shinka rejects by code embedding plus a judge. DIVERGE can reject by behavior and decision inputs, and can explain *why* the next candidate differs.
- *Learn:* reject before evaluation to save budget; a bandit over models.
- *Don't copy:* an LLM novelty judge as the gate.

**AdaEvolve / SkyDiscover / EvoX**
- *What it does:* an adaptive exploration rate, UCB over islands, stagnation-triggered "solution tactics"; EvoX rewrites the search strategy itself.
- *Overlap:* **the closest competitor**: detecting stagnation and injecting a new strategy.
- *Where DIVERGE differs:*
  - AdaEvolve detects stagnation from *score only*. DIVERGE can detect *strategic sameness* before the score stalls.
  - AdaEvolve's tactics come from its own failures. DIVERGE adds cross-model consensus data, unused inputs and agreement counterexamples.
  - Whether those additions help is unproven; that is Experiment 3 below.
- *Learn:* the improvement-signal formula; rotating tactics before generating new ones; the ablation design.
- *Don't copy:* its scope as a framework that owns the loop.

**GEPA**
- *What it does:* reflective mutation of prompts with per-instance Pareto selection.
- *Overlap:* using failures to drive the next candidate.
- *Where DIVERGE differs:* GEPA optimizes prompts and has no notion of mechanism.
- *Learn:* **per-instance Pareto retention.** It is cheap and has the cleanest ablation (greedy gets stuck).
- *Don't copy:* reflection as the only source of variation.

**Weco / AIDE**
- *What it does:* tree search with draft/debug/improve, 5 drafts, debug depth 3.
- *Overlap:* repair vs redraft.
- *Where DIVERGE differs:* AIDE's choice between repair and redraft is random or fixed. DIVERGE's can be based on diagnosed failures.
- *Learn:* the AIRA finding that operators limit everything, and that validation overfitting is real.
- *Don't copy:* fixed probabilities for debug vs draft.

**OpenEvolve**
- *What it does:* an open AlphaEvolve with MAP-Elites, islands and a novelty filter.
- *Overlap:* diversity maintenance.
- *Where DIVERGE differs:* OpenEvolve's README shows a plateau at 2.377 that a *human* broke by reconfiguring the run. DIVERGE aims to automate exactly that step.
- *Learn:* treat its plateau logs as a ready-made test set for stuck states.
- *Don't copy:* complexity as a hand-picked diversity axis.

**Reflexion**
- *What it does:* verbal reflections from task feedback, stored in memory.
- *Overlap:* using failure feedback.
- *Where DIVERGE differs:* Reflexion repairs within one strategy and suffers Degeneration-of-Thought. DIVERGE decides whether to stop repairing.
- *Learn:* feedback from the environment beats self-assessment.
- *Don't copy:* unbounded reflection.

**Self-Refine**
- *What it does:* self-feedback, then refinement.
- *Overlap:* the repair path.
- *Where DIVERGE differs:* DIVERGE uses objective failure signatures.
- *Learn:* little.
- *Don't copy:* intrinsic critique on reasoning tasks (Huang: it hurts).

**OpenHands**
- *What it does:* exact-repeat stuck detection (4/3/3/6), then stops (V1); V0 offered memory truncation.
- *Overlap:* loop detection.
- *Where DIVERGE differs:* strategic loops; automatic, evidence-backed recovery.
- *Learn:* a one-time nudge before declaring stuck; scope detection to events since the last user message; detectors must be cheap and high-precision.
- *Don't copy:* stopping without a next step.

**Amp Oracle / Claude Code advisor**
- *What it does:* second opinion from a stronger model.
- *Overlap:* "consult someone else when stuck."
- *Where DIVERGE differs:* stronger models share errors (Kim et al., Goel et al.). DIVERGE gives *a reason* and a *test*, not just an opinion.
- *Learn:* the advisor's trigger "when an error keeps recurring."
- *Don't copy:* opaque, model-decided timing.

---

## Proposed DIVERGE architecture

```
agent / evolver ──► [1] Trace adapter ──► attempts: code, eval result per dev instance, timestamps
                                             │
        ┌────────────────────────────────────┼─────────────────────────────┐
        ▼                                    ▼                             ▼
[2] Fingerprinter                     [3] Progress monitor           [4] Consensus store (cached)
  - input flow (AST slice)              - external score momentum      - cross-model samples per task
  - state kept                          - failure-set overlap          - paraphrase sensitivity
  - behavior on fixed probes            - repetition (OpenHands)       - shared assumptions, unused inputs
  - per-instance failure signature
  - LLM mechanism caption (display only)
        └──────────────┬─────────────────────┘                             │
                       ▼                                                   │
              [5] Diagnoser: REPAIR | RELATED | REPRESENT | ESCALATE ◄─────┘
                       │  (rules first; learn later from logged outcomes)
                       ▼
              [6] Lever generator → 3–5 mechanism sketches, each tied to evidence:
                  unused input · broken incidental assumption · agreement counterexample ·
                  unoccupied component cell · minority mechanism
                       ▼
              [7] Admission gate: sketch → code → fingerprint must leave the stuck cluster
                  AND must actually use its lever (perturbation test). Reject before scoring.
                       ▼
              [8] Allocator: Thompson sampling over lineages (REx/HGM prior), Pareto retention
                       ▼
              [9] Evaluator gateway: sandbox, dev for selection, held-out for reporting,
                  exploit checks (tolerance, NaN, timeouts)
                       ▼
             [10] Explainer: "your 4 attempts = 1 strategy" view, lever + evidence, result
```

**Interfaces** (MCP and CLI; this mostly maps onto existing modules):

- **`diverge_status(attempts)`** → `{stuck: bool, reason: literal|stall|same_strategy, clusters, behavior_similarity, code_similarity, shared_failures}`
  - Uses `strategy.py` and `trajectory.py`.
- **`diverge_diagnose(attempts)`** → `{action: repair|related|represent|escalate, evidence}`
  - Uses `diagnose.py`.
  - Rules to start with:
    - *repair*: failures differ between attempts, or the score is still improving.
    - *escape*: at least 3 attempts in one behavior cluster, the same failure set, and the momentum is below threshold.
    - *escalate*: two or more escapes have failed.
- **`diverge_escape(attempts, task, evaluator)`** → `{lever, why_different, sketch, code, admission_proof, dev_score}`
  - Uses `escape.py`, `overlook.py` and `counterexample.py`.
- **`diverge_counterexample(attempts)`** → the smallest input where every attempt makes the same decision and a better decision exists.

**Design rules:**
- Every claim of difference is backed by behavior, not text.
- Every lever cites its measured signal.
- Every escape is logged with the matched-control outcome, so the diagnoser can later be trained on real results.

---

## Three highest-value experiments

**E1. Detector validity (offline, about $0 using cached samples).**
- **Question:** does "same behavior cluster" predict that *continuing within the cluster won't help*, better than code embeddings or an LLM judge?
- **Data:** the existing consensus-map samples plus benchmark candidates for bin packing (more than 100 programs), with dev scores and per-instance failures.
- **Method:**
  - For each pair, compute behavior agreement on the 200 probes, embedding cosine, AST similarity and the LLM mechanism label.
  - Outcome: the overlap between the pair's failure sets, and the score gap.
  - Report AUC for separating "same failure set" pairs.
- **Success:** behavior AUC is at least 0.10 above embedding AUC, and within-cluster best-of-k gains are smaller than cross-cluster gains at the same k.
- **If it fails:** fingerprints are cosmetic, and DIVERGE's detection story collapses.

**E2. Rescuing naturally stuck agents (the thesis test, about $5–8).**
- **Setup:**
  - Run a cheap implementer agent for 6 attempts on 3 tasks: bin packing, cache eviction, and about 20 hard LiveCodeBench problems where pass@10 has plateaued.
  - Keep only trajectories that are *naturally* stuck (no dev improvement over the last 3 attempts). Expect 30 or more.
- **Arms** (from the same state, K=4 more attempts, the same token budget, the same implementer):
  1. continue;
  2. fresh context + failure summary;
  3. fresh context + "try a fundamentally different approach" + list of previous attempts;
  4. temperature 1.3;
  5. a stronger model (adjusted for cost);
  6. best-of-K fresh samples;
  7. a random lever (a random input, assumption or cell, drawn from the same generator);
  8. DIVERGE escape.
- **Primary metric:** whether the held-out score beats the plateau, paired across arms.
- **Secondary metrics:** whether attempts leave the cluster, and whether they actually use the lever.
- **Success:** DIVERGE's rescue rate beats arms 2, 3 and 7 by at least 15 points, with a paired McNemar test p<0.05, on at least 2 of 3 tasks.

**E3. Does cross-model information add anything over the agent's own history? (about $3)**
- **Context:** this separates DIVERGE from AdaEvolve.
- **Arms on the same stuck states:**
  - (a) levers built from the agent's own attempts only (AdaEvolve-style tactics);
  - (b) levers built from the cross-model consensus;
  - (c) a consensus **sham**: consensus statistics with family labels shuffled across tasks;
  - (d) levers built from the literature taxonomy without consensus shares (DIVERGE's `literature_random`).
- **Success:** (b) beats (a), (c) and (d) on rescue rate.
- **If (a) ≈ (b):** cross-model sampling is cost without benefit, and DIVERGE should become an own-trajectory tool.

---

## Best hackathon demo (2 minutes, online bin packing)

- **0:00–0:20, the agent is stuck.**
  - A coding agent writes a bin-packing heuristic and is scored on Weibull instances (FunSearch's setup).
  - Four attempts, four different-looking files: best fit with a penalty, best fit with a tie-break, a "smarter" scoring formula, a refactor.
  - Score panel: about 4% excess bins every time.
- **0:20–0:50, DIVERGE shows it is one idea.**
  - Panel A: code similarity 28%. Panel B: on 200 probe states, the attempts pick the same bin 97% of the time; four dots collapse to one.
  - Input heat strip: none of them uses the number of open bins or the bins' opening order, though the task allows both. In the current map, the number of open bins enters the decision in only 5 of 95 consensus-map samples; opening order in 34 of 95.
  - All 4 fail the same dev instances.
- **0:50–1:15, the counterexample.**
  - A 6-item sequence animates. All four attempts make the same placement; the exact optimum (by brute force) does not, and uses one fewer bin.
  - The caption names the shared assumption: "fit each item as tightly as possible now."
- **1:15–1:45, escape.**
  - DIVERGE proposes one lever with its evidence, e.g. "unused input: number of open bins; 0 of 4 attempts and 5 of 95 model samples use it". Whether this lever actually helps has not been measured; pick the demo lever only after E2.
  - The new candidate's fingerprint lands *outside* the cluster.
  - Held-out excess bins drop (the known reference point is the FunSearch heuristic at 0.68% vs best fit 3.98%).
- **1:45–2:00, the control bar chart.** At the same budget:
  - "try something different": 0/N rescued;
  - fresh context: x/N;
  - random lever: y/N;
  - DIVERGE: z/N.

**Honesty rule:** run E2 first and show whatever the bars actually say. If fresh context ties DIVERGE, the demo becomes "DIVERGE tells you you're stuck and why." That is still true and useful, but it is a weaker claim.

---

## Direct answers to the key questions

**1. The ten strongest ways to get a genuinely different strategy**

1. Stagnation-triggered tactic injection (AdaEvolve).
2. Exclusion enforced by a check against fingerprints of earlier attempts (ShinkaEvolve, NeoCoder).
3. Idea-level planning from observations (PlanSearch, EoH).
4. Distinguishing or agreement counterexamples (Jha 2010, S\*).
5. Per-instance Pareto retention (GEPA).
6. An open archive with lineage value (DGM, HGM).
7. Representation change: PoT, object or graph encodings.
8. Fresh context that leaves out earlier attempts (incubation).
9. Relevant idea injection or rephrasing for pass@k (DivSampling, DIV-SE).
10. Distribution elicitation (Verbalized Sampling).

Training methods (DARLING, outcome-based exploration) are strong but outside DIVERGE's scope.

**2. Which have objective performance evidence**

| Method | Objective result |
|---|---|
| AdaEvolve meta-guidance | Ablation, 3 runs |
| GEPA Pareto | Ablation |
| DGM archive | Ablation |
| PlanSearch, DivSampling, DARLING | pass@k gains, **not pass@1** |
| EoH thought+code | Gap 0.66% vs 2.57% |
| Hypothesis Search | ARC 17 → 30% |
| AlphaCode clustering, S\* | Better selection |
| REx | Cost savings |
| Representation change | PoT, ARC objects, "Talk like a Graph" |
| Verbalized Sampling | Synthetic math data only; otherwise judge or embedding scores |

No objective evidence:
- ADHD (judge only).
- "Show common answers, ask for different" (no objective study).
- Analogy (gains are small).
- Assumption-listing prompts (no study found).

**3. What works specifically after repeated failed attempts**

| Method | Stuck-specific evidence |
|---|---|
| AdaEvolve meta-guidance | Triggered only on stagnation; ablation |
| PACEvolve backtracking + context reset | Removes poor runs |
| CUSUM rollback (arXiv 2608.02464) | Recovery 45% vs 16% |
| REx | Down-weights a program after each failed refinement |
| Fresh context | Laban 2025 |

Plain repair or reflection is the weak option here.

**4. Detecting that attempts are strategically equivalent**

- Run them on a fixed probe set and compare their decisions (not final scores). Two attempts that agree on at least about 97% of probe decisions are one strategy.
- Add three more signals:
  - the *inputs* each attempt reads (static slicing);
  - the overlap of their per-instance failure sets;
  - a distinguishing-input search: if no small input separates them, treat them as equivalent.
- Use LLM mechanism labels only as captions, since different judges agree only at rank correlation 0.46–0.74.
- Validate the threshold on known mechanisms first. DIVERGE already does this (`fingerprints` gate).

**5. Turning failure evidence into a decision to switch**

| Diagnosis | Evidence |
|---|---|
| **Repair** | Failures differ across attempts, or the error is an implementation error: crash, timeout, wrong format |
| **Switch mechanism** | Same failure set + same behavior cluster + flat momentum over at least 3 attempts |
| **Switch representation** | At least 2 clusters tried and all fail a shared instance subset |
| **Escalate** | Two escapes failed, or no lever passes admission |

- Then pick the next mechanism by what the shared failures *have in common*, e.g. "all fail on high-variance item streams" points to "use the distribution."
- Weight lineages with an REx-style prior that shrinks with each failed repair.

**6. Problem information LLMs commonly ignore**
- Inputs available at decision time but not used.
  - [DIVERGE] Bin packing, from `overlook.json`: the number of open bins enters the decision in 5 of 95 consensus-map samples, and bin opening order (age) in 34 of 95. The item-size distribution is used by about 70%, so it is *not* overlooked.
- The history and statistics of the stream (distribution, order, age).
- Tie-break semantics.
- The scoring rule's structure, e.g. what the evaluator actually rewards.
- Counterfactual task variations that differ from the familiar version (Wu et al. 2023; McCoy et al. "Embers").
- Information in the middle of long context.
- Modified constraints in familiar puzzles (MedEinst: stronger models were *more* blind).

**7. What human fixation research suggests about LLM strategy loops**
- Einstellung is strongest when the familiar solution is *good enough*. Bilalić: experts' gaze stayed on the known plan, cutting their performance to that of players 3 SD less skilled; it fades when the familiar move is a blunder (Sheridan & Reingold).
- Expect LLM loops when the default scores decently. Expect them least when it crashes.
- Implications:
  - Forgetting helps (incubation and fresh context).
  - Warnings don't help for familiar features.
  - Examples narrow search, and the narrowing is weaker when the examples are fewer and less common (Sio et al. 2015).
  - Building and testing helps.
- Concretely: show a *counterexample*, not the consensus answers; show few, uncommon examples; reset the context.

**8. Strongest evidence that showing default answers helps**
- **Making outputs more distinct:**
  - NoveltyBench in-context regeneration (judge-scored distinctness).
  - DivAlgo iterative generation, Vendi 6.69 → 8.83.
  - NeoCoder denial prompting, which forces new techniques on Codeforces.
- **Humans:** Chrysikou & Weisberg 2005, where defixation instructions reduced copying of flaws.
- **For validity of the alternatives, none of these is strong.** I found no matched-budget pass@k study.

**9. Strongest evidence against it**
- **LLM studies:**
  - G2: quality drops from 7.97 to 4.73 when conditioning on earlier outputs.
  - The pink-elephant effect.
  - NeoCoder: forced novelty stays below human level, and correctness falls.
- **Human studies:** Smith, Ward & Schumacher 1993; Viswanathan & Linsey 2020 ("avoid" warnings fail for familiar features).
- **[DIVERGE] Own results:**
  - The latent-tail null result.
  - In the partial seed-0 cache-eviction run, `plain` (0.5666 dev miss ratio) beat every consensus-aware operator. It was 1 seed, on dev only, with 4–8 candidates each.

**10. What DIVERGE must show to prove that knowing the consensus adds value**

DIVERGE must beat each baseline in a preregistered, paired, matched-budget, held-out comparison on naturally stuck states:

| Baseline | What it controls for |
|---|---|
| Fresh context + summary | Forgetting alone |
| Stronger model | Capability alone |
| "Try something different" + earlier attempts | Generic exclusion |
| Higher temperature | Randomness |
| Random reframing / random lever from the same generator | Structure without information |
| Second opinion from another family | Source diversity |
| Ordinary reflection | Self-critique |
| Literature search without consensus shares | Retrieval |
| Best-of-N fresh samples | Budget |

Two more conditions:
- **A consensus-sham ablation.** Shuffling or replacing the consensus statistics must remove at least half the gain. Otherwise the consensus isn't what helps.
- **Own-history only.** Replacing cross-model data with the agent's own attempts must lose the gain. Otherwise DIVERGE is AdaEvolve with extra steps.

---

## Five concrete DIVERGE mechanisms

### 1. Same Move Detector → Exclusion Gate

- **Signal:**
  - Each attempt's decisions on 200 fixed probe states.
  - Code similarity between attempts.
  - Per-instance dev outcomes.
  - The external score over time.
- **Detection:**
  - At least 3 recent attempts agree on ≥97% of probe decisions (a threshold validated on hand-written mechanisms)
  - while their code similarity is below 50%,
  - and the dev score has improved by less than ε over the last 3 attempts.
- **Intervention:**
  - The next candidates must *disagree* with the cluster on at least 10% of probes.
  - The cluster's majority decision is shown on 3 example probe states.
  - Candidates are rejected before full scoring if they land back in the cluster.
- **Why this may escape the loop:** cosmetic variation can no longer pass, so the generator has to change its decision rule. The check is mechanical, so it doesn't depend on the model following an instruction, which fixation research says fails for familiar defaults.
- **Simplest baseline:** "Your previous attempts are below; try a fundamentally different approach" (in-context regeneration), with no gate.
- **Experiment:**
  - On naturally stuck states, K=4 attempts per arm at equal token cost.
  - Arms: gate, prompt-only, and a gate with a random 10%-disagreement target (it forces difference without saying which decisions).
- **Success criterion:** held-out rescue rate for the gate is at least 15 points above prompt-only, paired, p<0.05.
- **Failure mode:**
  - Forced disagreement produces *worse* random decision rules: diversity without quality.
  - Or the probe set doesn't represent the evaluator's instances, so disagreement happens where it doesn't matter.
- **Demo visualization:** two scatter plots side by side, "how the code looks" (4 dots spread apart) and "what the code does" (4 dots stacked on one point), with the new candidate appearing far away in the second.

### 2. Unused Signal Lever

- **Signal:**
  - Static backward-slice input flow, for every attempt and every consensus-map sample.
  - A fixed list of the inputs available at decision time.
- **Detection:** an input used by 0 of the agent's attempts and ≤10% of consensus-map samples, which the task statement allows.
- **Intervention:**
  - Generate 2–3 mechanism sketches whose decision *must* depend on that input.
  - Admission requires that perturbing the input changes at least 5% of probe decisions (a faithfulness check).
- **Why this may escape the loop:** a strategy that decides from different information is a different mechanism by construction. It changes the problem's representation without depending on the model's creativity.
- **Simplest baseline:** the same template with a *random already-used* input, plus a generic "consider all available information" prompt.
- **Experiment:** the operator against its control (already preregistered as `unused_signal` vs `unused_signal_control`), 3 repeats, held-out best score.
- **Success criterion:** beats its control and `plain` on held-out best in every repeat (the existing rule).
- **Failure mode:**
  - The input is unused *because it is useless*. Then the lever only produces faithful but worse candidates.
  - Or the model uses the input in a token way (an `if` that never fires). The perturbation test catches that.
- **Demo visualization:** an input heat strip with one column per input and one row per attempt. One column stays empty; the new attempt lights it up.

### 3. Agreement Counterexample

- **Signal:**
  - The attempts' decisions on small generated instances.
  - A brute-force or exact solver on those instances.
  - The evaluator.
- **Detection:** search small inputs (e.g., 5–8 items) for cases where **all** recent attempts make the same decision and the exact solver's choice yields a strictly better outcome. Rank by how small the input is and how large the gap.
- **Intervention:** give the agent the counterexample:
  - the state;
  - the consensus decision;
  - the better decision;
  - the resulting outcomes;
  - the shared assumption it exposes.

  Ask for a mechanism that would get this case *and the dev set* right.
- **Why this may escape the loop:** it turns an invisible shared assumption into a concrete failing case. This is the oracle-guided-synthesis principle, and in humans building and testing worked where warnings failed. It gives a reason for the change that a generic retry lacks.
- **Simplest baseline:**
  - A random failing dev instance with ordinary error feedback (Reflexion-style).
  - A *disagreement* counterexample, where the attempts differ.
- **Experiment:** three arms on stuck states with matched tokens: agreement counterexample, random failing instance, no feedback (fresh context).
- **Success criterion:** at least 15 points higher rescue rate than the random-failing-instance arm, and the new candidate gets the counterexample right in at least 70% of cases without held-out regression.
- **Failure mode:**
  - Overfitting to the toy case (optimal-offline choices on 6 items don't carry over to online scale).
  - Or the exact solver uses hindsight (future items) that no online policy can have, which makes the counterexample unfair.

  Restrict the solver to information the policy legally has, e.g. the expected value under the known distribution.
- **Demo visualization:** an animation of six items dropping into bins. Four attempts make the same move (red); the alternative makes a different one (green) and finishes with one fewer bin.

### 4. Repair-or-Escape Gate

- **Signal:**
  - Per-instance failure sets across attempts.
  - Error types (crash, timeout, wrong output, low score).
  - Behavior-cluster membership.
  - Score momentum, as in AdaEvolve: G ← ρG + (1−ρ)δ².
- **Detection:** rules, to be learned later from logged outcomes:
  - **repair** if errors are implementation errors or failure sets are changing;
  - **escape** if there is one cluster, Jaccard overlap of failure sets ≥0.8, and G is below threshold;
  - **switch representation** if at least 2 clusters have failed with a shared failure core;
  - **escalate** after 2 failed escapes.
- **Intervention:** spend the next budget on the chosen path; for escape, call mechanism 1, 2 or 3.
- **Why this may escape the loop:** it stops pouring budget into repair, which matched-budget studies show is weak, exactly when evidence says the mechanism, not the implementation, is the problem. It stays in repair when it should, which avoids the underthinking failure.
- **Simplest baseline:** AIDE's fixed rule (debug up to depth 3, then redraft), plus "always escape" and "always repair."
- **Experiment:**
  - Branch each recorded stuck state into both repair and escape at equal budget, so the better action is known after the fact.
  - Measure the gate's regret against the oracle choice and against the fixed policies.
- **Success criterion:** at least 30% less regret than the best fixed policy across 3 tasks.
- **Failure mode:**
  - The gate learns "always escape" because escape attempts get fresh context, which is a confound. Control with fresh-context repair.
  - With few states, the rules get tuned to noise.
- **Demo visualization:** a traffic light next to the attempt timeline. Green means repair (errors differ), amber means stalled, red means "same idea, same failures": escape.

### 5. Consensus Anchor Break

- **Signal:**
  - Assumptions and component values held by ≥90% of cross-model consensus-map samples *and* by every recent agent attempt.
  - Each is classed as incidental unless supported by a verbatim quote from the prompt (already built into `overlook.py`).
- **Detection:** an incidental assumption that is shared by everything and not required by the task.
- **Intervention:**
  - Regenerate with exactly that assumption negated, e.g. "ties are not harmless; bin age is a signal."
  - An extractor check confirms the new candidate violates the anchor and keeps every hard constraint.
- **Why this may escape the loop:** targeted constraint relaxation (Knoblich 1999). The consensus tells *which* constraint is self-imposed, so this is not a random reframing.
- **Simplest baseline:**
  - Break a random incidental assumption taken from non-consensus families (already preregistered as `assumption_break_control`).
  - A *shuffled-consensus* anchor taken from a different task.
- **Experiment:** three arms, 3 repeats, held-out best score, with a faithfulness check.
- **Success criterion:** beats both controls and `plain` on held-out best in every repeat.
- **Failure mode:**
  - The anchor is shared *because it is load-bearing*: the consensus is right, and breaking it makes things worse.
  - The extractor invents an assumption no attempt really holds.
  - [DIVERGE] Only 1 shared assumption survived the check for bin packing, and it was classed *hard*, so this lever may rarely fire.
- **Demo visualization:** a bar reading "92% of solutions from 5 model families assume X; the task never says X." Strike through X, then the new score.

---

## Skeptical conclusion: assume DIVERGE is wrong

**The strongest reasons the thesis may fail:**

1. **The value is in the verifier and the budget, not in the consensus.**
   - Brown et al.: coverage grows log-linearly with plain resampling.
   - Toledo et al.: operator design limits everything else.
   - Olausson: resampling ≈ repair.
   - With an evaluator and N samples, best-of-N may capture most of what DIVERGE claims to add.
2. **The consensus is a property of the wording.** [DIVERGE]'s own paraphrase test says so. "Knowledge of the consensus" may be knowledge of one prompt's attractor; random rephrasing (DivSampling) captures the same effect for free.
3. **Showing the consensus backfires.**
   - In human design work, examples cause fixation, and warnings fail for familiar features.
   - In LLMs, conditioning on earlier outputs lowers quality (G2).
   - [DIVERGE]'s latent-tail test found no unlocked knowledge.
4. **Consensus is often consensus because it is good.**
   - Novelty doesn't predict outcomes (arXiv 2604.19440).
   - Near sources beat far ones.
   - Good LLM optimizers are local refiners.
   - Pushing away from the mode trades a decent mechanism for a rare, worse one.
5. **Fresh context explains every escape.** Incubation works by forgetting. Any mechanism that restarts the context will look like it escapes, whatever the consensus says.
6. **The detection is circular or fragile.**
   - Probe sets and thresholds are hand-chosen.
   - Mechanism labels from LLMs disagree (rank correlation 0.46–0.74).
   - For tasks without tractable intermediate states (most of SWE), behavioral fingerprints aren't available.
7. **A competitor already owns the useful part.**
   - AdaEvolve already does stagnation detection plus tactic generation from its own failures, with an ablation.
   - What DIVERGE adds (cross-model data, anchors) is the part with the least evidence.
8. **Small, noisy wins.**
   - Benchmark rankings in this area flip with seeds (arXiv 2609.19799).
   - [DIVERGE]'s current 3-repeat design can only detect large effects.
   - In the partial run, `plain` won.

**What would change my mind:** a preregistered E2 + E3 on at least 30 naturally stuck trajectories across at least 2 tasks with held-out scoring, in which:
- (a) DIVERGE's rescue rate beats fresh context + "try something different" + earlier attempts **and** a random lever from the same generator by at least 15 points, paired, p<0.05;
- (b) a consensus sham (shuffled statistics) loses at least half of that gain;
- (c) own-history-only levers lose at least half of it.

If (a) holds but (b) and (c) fail, DIVERGE's value is the detector and the exclusion gate, not knowing the consensus, and it should be repositioned that way. If (a) fails, the measurement tooling (paraphrase test, fingerprints, behavior clusters) is still a useful contribution to *evaluation*, but not an escape method.

---

## Source list (primary)

**Evolutionary search and stagnation**
- AdaEvolve: arXiv 2602.20133 (Feb 2026)
- EvoX: arXiv 2602.23413 (Feb 2026)
- PACEvolve: arXiv 2601.10657 (Jan 2026)
- ShinkaEvolve: arXiv 2509.19349 (Sep 2025)
- AlphaEvolve: arXiv 2506.13131 (Jun 2025)
- Georgiev, Gómez-Serrano, Tao & Wagner: arXiv 2511.02864 (Nov 2025)
- OpenEvolve: github.com/algorithmicsuperintelligence/openevolve
- FunSearch: Nature, Dec 2023 (doi 10.1038/s41586-023-06924-6)
- GEPA: arXiv 2507.19457 (Jul 2025)
- EoH: arXiv 2401.02051 (ICML 2024)
- HSEvo: arXiv 2412.14995 (AAAI 2025)
- Mutation Without Variation: arXiv 2606.05408
- What Makes an LLM a Good Optimizer: arXiv 2604.19440
- Evolution or Illusion: arXiv 2609.19799

**Agents and self-improvement**
- AIDE: arXiv 2502.13138
- AIRA: arXiv 2507.02554
- DGM: arXiv 2505.22954
- HGM: arXiv 2510.21614
- SICA: arXiv 2504.15228
- OpenHands SDK: `openhands-sdk/openhands/sdk/conversation/stuck_detector.py`
- Amp Oracle: ampcode.com/news/gpt-5-oracle
- Claude Code advisor: code.claude.com/docs/en/advisor
- ADHD: github.com/UditAkhourii/adhd
- Park & Choi, stagnation mistaken for progress: arXiv 2607.25152
- Real-time failure detection: arXiv 2608.02464

**Repair, resampling and reasoning**
- Olausson et al.: arXiv 2306.09896
- Snell et al.: arXiv 2408.03314
- REx: arXiv 2405.17503
- Huang et al.: arXiv 2310.01798
- Liang et al. (MAD): arXiv 2305.19118
- Kang et al.: arXiv 2510.08308
- Underthinking: arXiv 2501.18585
- Meta-Reasoner: arXiv 2502.19918
- Yue et al.: arXiv 2504.13837
- Large Language Monkeys: arXiv 2407.21787

**Diversity methods**
- PlanSearch: arXiv 2409.03733
- SFS: arXiv 2411.05010
- Verbalized Sampling: arXiv 2510.01171
- DivSampling: arXiv 2502.11027
- DIV-SE: arXiv 2310.07088
- NoveltyBench: arXiv 2504.05228
- G2: arXiv 2511.00432
- NeoCoder: arXiv 2407.09007
- DARLING: arXiv 2509.02534
- Outcome-based exploration: arXiv 2509.06941
- Hypothesis Search: arXiv 2309.05660

**Detecting strategic equivalence**
- AlphaCode: arXiv 2203.07814
- S\*: arXiv 2502.14382
- DivAlgo: arXiv 2610.19157
- Jha et al., ICSE 2010
- Semantic entropy: Nature 630 (2024)

**Convergence, monoculture and minority answers**
- Artificial Hivemind: arXiv 2510.22954
- Correlated Errors: arXiv 2506.07962
- CAPA: arXiv 2502.04313
- Self-MoA: arXiv 2502.00674
- Debate or Vote: arXiv 2508.17536
- Consensus is Not Verification: arXiv 2603.06612
- Minority Sentinel: arXiv 2606.29270
- Si et al.: arXiv 2409.04109
- Ideation-Execution Gap: arXiv 2506.20803
- FormatSpread: arXiv 2310.11324
- Laban et al.: arXiv 2505.06120
- Sinha et al.: arXiv 2509.09677
- MedEinst: arXiv 2601.06636

**Representation and abstraction**
- Talk like a Graph: arXiv 2310.04560
- ARC object representations: arXiv 2305.18354
- Step-Back: arXiv 2310.06117
- System 2 Attention: arXiv 2311.11829
- Program of Thoughts: arXiv 2211.12588

**Human creativity and fixation**
- Bilalić et al. 2008 (Cognition 108; Cognitive Psychology 56)
- Jansson & Smith 1991
- Sio, Kotovsky & Cagan 2015 (Design Studies 39)
- Chrysikou & Weisberg 2005
- Viswanathan & Linsey 2020
- Knoblich et al. 1999
- Sio & Ormerod 2009
- Vul & Pashler 2007
- Chan, Dow & Schunn 2015
- Hope et al. 2017
- Mullen et al. 1991
- Nemeth et al. 2001
- Lorenz et al. 2011
- McCaffrey 2012
- Lehman & Stanley 2011
