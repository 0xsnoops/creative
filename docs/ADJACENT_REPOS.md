# Adjacent repos: what we learned and what we took (2026-10-18)

This was a read-only review. Every repo was cloned with `--depth 1` outside our repo and read as
text. Nothing was run, installed or imported, and no code or prompt text was copied. The changes
below are our own implementations of ideas.

**Disclosure.** Before this review finished, the panel in `docs/BENCH_DOCTOR.md` had already
nominated escape v3 (critic + deepen + anchor strip), the readable output and free-text
`diverge_check`. Work on all three had started. The repo review confirmed them and added one
change, P2.

## Chosen changes (at most 4)

| # | change | kind | why | where |
|---|---|---|---|---|
| P1 | **escape v3**: anchor-stripped failure summary (each kept constraint quoted from the task); 5 sketches in isolated calls, each on a different lever; a critic on a different model (scores, one strength, one trap, is_trap / restates_stuck); deepen the top 2; implement through the existing admission check | performance, opt-in mode, default stays v2 | ADHD's generator/critic separation and deepen-top-K; 15 of 18 panel models named the missing critic gate | `diverge/escape.py` `escape_v3` |
| P2 | **Feed admission rejections back into v3's resample**: the resample prompt states why the last candidate was rejected (probe disagreement, lever sensitivity) | performance, inside v3 only | Shinka resamples blindly after a novelty rejection, and so did we. In rescue-test state 3 nothing was admitted | `diverge/escape.py` (v3 resample) |
| U1 | **Readable markdown block** above the JSON in `diverge_escape` / `diverge_check`: verdict, shortlist, traps, next test, with every number taken from the result object | product | Agents and people read the text first; the first line stays `VERDICT:` | `diverge/contract.py`, `diverge/server.py` |
| U2 | **Free-text `diverge_check`**: without an evaluator it returns the herd comparison (only if a map is cached) plus 3–5 alternatives, each with a strength, a trap and how to test it, all HYPOTHESIZED | product | It makes the pre-implementation check useful without an evaluator | `diverge/server.py` |

## Per repo

### UditAkhourii/adhd (@7b91ce6)
- **Generator/critic separation, trap notes, anchor stripping, deepening the top ideas, clustering by angle, recombining siblings while deepening.** We took the critic, trap, anchor-strip and deepen ideas (P1).
- **Our differences:** we keep constraints only when they are verbatim quotes, which is stricter than ADHD's reframing, and our trap gate is an explicit boolean.
- **Rejected for now: clustering by angle and recombining siblings.** Both are plausible and cheap, but they came too late for this freeze. They are the next candidates.

### CHATS-lab/verbalized-sampling (@d042a7b)
- **The idea:** ask for k responses with stated probabilities and sample from the tails.
- **Two problems confirmed:**
  - **A hidden ~200-word default.** `verbalize()` never passes `target_words`, so every system prompt silently gets "approximately 200 words".
  - **A tau/tails contradiction.** The prompt asks for probabilities below 0.10, the filter keeps items at or above tau (default 0.12), and the relaxation step then keeps the most typical items. The filter therefore selects against the tail.
- **Rejected for v3:** the change would take about 2 hours and alter sketch parsing. Our existing `verbalized` operator never imports VS and has no length cap. If we adopt tails later, the stated probability should only order candidates (lowest first), never act as a minimum.

### SakanaAI/ShinkaEvolve and algorithmicsuperintelligence/openevolve
- **Novelty check.** Shinka rejects near-duplicates before evaluation with embeddings and an LLM judge. Its judge counts hyperparameter changes as novel, so it would admit exactly the "different code, same strategy" programs DIVERGE exists to catch. OpenEvolve's check runs after evaluation. Our behavioral admission check (`diverge/admission.py`) is stronger, so we kept it and took only the "say why it was rejected" idea (P2).
- **Failure artifacts in the next prompt** (stderr, failure stage, improved/regressed labels). A one-repair-on-crash step is a candidate for later.
- **Packaging:** Shinka's setup/convert/run/inspect skills, and OpenEvolve's host-CLI backend that needs no API key. Both would help adoption. An evaluator setup helper is the top adoption follow-up. Both would take 3–6 hours, so they are not in this freeze.
- **Rejected on scope:** islands, MAP-Elites, archives and populations.

### noahshinn/reflexion and WecoAI/aideml
- **Reflexion:** turns each failure into a free-text lesson and keeps the last 3, but has no "switch approach" rule.
- **AIDE:** abandons a branch only after 3 failed debug steps on broken code. A working branch that has stopped improving is never abandoned.
- **Not adopted now:** a stateless "escape after K stalled attempts" rule. It would flip the recorded demo fixtures and would not change the rescue test. It is the best follow-up for the agent-recovery benchmark.

### gepa-ai/gepa, plus DGM and AI Scientist v2 (concept level)
- **GEPA:** per-instance Pareto selection and accept-only-on-improvement.
- **DGM:** diagnoses one concrete failed instance.
- **AI Scientist v2:** debug nodes.
- **Not adopted now:**
  - Pareto choice among at most 3 candidates has little effect.
  - Our contract already says when DIVERGE's own candidate did not beat the before score.
  - Per-instance gaps in the summary change cache keys.
- **Rejected on scope:** archives and parent selection.

### jeremyguez/novel-associations-agent and the packaging sweep
See the appendix (Research E) if it finished in time.

## Positioning
ADHD generates from different frames. Verbalized Sampling samples the tails. Reflexion learns
from the last failure. Shinka and OpenEvolve search more candidates. DIVERGE checks whether all of
those candidates are still the same strategy, and where the neglected direction is.
