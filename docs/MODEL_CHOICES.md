# Model choices

## Implementer and critic models

*Researched 2026-10-18. Prices come from `cache/openrouter_models.json` (fetched 2026-10-18) and the Anthropic model table (cached 2026-10-06). Token and cost figures are measured from DIVERGE's own `cache/llm.jsonl` (per-call means over code-writing prompts).*

### Evidence base and its limits
- **No public benchmark covers these exact model versions.** The official LiveCodeBench board has not changed since 2025-08-01 (release v6, top entry o4-mini). Aggregator figures for 2026 models are vendor-reported, with mixed setups. One example: a 2026-09-10 aggregator lists DeepSeek-V4-Pro-Max at 93.5% and Fable 5 at 89.8% ([presenc.ai](https://presenc.ai/research/livecodebench-leaderboard-2026), [pricepertoken](https://pricepertoken.com/leaderboards/benchmark/livecodebench)). I found no current Aider-polyglot or SWE-bench entries for Gemini 4 Argon, Sonnet 5.5, Haiku 5.5 or GPT-6.x. Exa was unavailable (it needed re-authentication), so these searches used the built-in web search.
- **Heuristic-discovery papers use cheap backbones.** EoH-S (2025-08) ran EoH, FunSearch and ReEvo on online bin packing with DeepSeek-V3 ([arXiv 2508.03082](https://arxiv.org/pdf/2508.03082)), and an AST-operator paper (2026-04) used Qwen-Flash ([arXiv 2604.16420](https://arxiv.org/pdf/2604.16420)). ShinkaEvolve (ICLR 2026) mixes several models with a UCB1 bandit over the ensemble ([arXiv 2509.19349](https://arxiv.org/abs/2509.19349)). I found no paper that compares reasoning and non-reasoning models for automated heuristic design.
- **Small judges are weak on code correctness.** An ICSE 2026 study found that small LLMs struggle as code-correctness judges, and even the best one often failed ([arXiv 2507.16587](https://arxiv.org/pdf/2507.16587)). So the critic should judge sketches and wording, and the evaluator should judge correctness.
- **The strongest evidence is DIVERGE's own call log:**

| model | $/M in/out | calls | median / p90 output tokens | mean reasoning tokens | calls that hit max_tokens | mean $/call | temperature |
|---|---|---|---|---|---|---|---|
| google/gemini-4-argon | 2 / 10 | 152 | 1494 / 2541 | 271 | 2 | 0.020 | yes |
| anthropic/claude-sonnet-5.5 | 2 / 10 | 117 | 2174 / 3659 | 1030 | 2 | 0.033 | no (non-default value → 400) |
| anthropic/claude-haiku-5.5 | 0.10 / 0.50 | 252 | 1068 / 3725 | 756 | 5 | 0.0014 | no |
| anthropic/claude-opus-5.5 | 4 / 20 | 78 | 1062 / 4105 | 632 | 4 | 0.080 | no |
| openai/gpt-6.1-sol | 2 / 10 | 74 | 2388 / 6321 | 2796 | 0 | 0.037 | no |
| deepseek/deepseek-v4-pro-0813 | 0.66 / 1.98 | 187 | 169 / 11564 | 1633 | **12** | 0.0067 | yes |
| meta-llama/llama-4-maverick | 0.19 / 0.65 | 295 | 642 / 951 | 0 | 0 | 0.0005 | yes |
| deepseek-v4.1-flash, qwen3.8-flash, glm-5.3, step-3.7-flash | cheap | 3 each | 12000 | ~11k | **3 of 3** | — | yes |

The cheap reasoning models in the last row used their whole output cap on reasoning and returned empty or truncated replies (see BENCH_DOCTOR, `glm-5.3` "empty content 3 times"). They are where the reasoning-token blowups come from.

### Recommendations

| role | Anthropic-only mode | OpenRouter mode | settings | expected $/candidate |
|---|---|---|---|---|
| **Implementer** | `claude-sonnet-5-5` | keep `google/gemini-4-argon` | Sonnet: `output_config.effort: "medium"` (the default is `high`; thinking can't be disabled, so effort is the only lever), `max_tokens` 12000. Argon: `reasoning: {effort: "low"}` (or `reasoning.max_tokens` ≈ 3000), `max_tokens` 8000. | Sonnet ≈ $0.025–0.035; Argon ≈ $0.02 |
| Cheap implementer arm (optional) | `claude-haiku-5-5`, effort `medium` | `deepseek/deepseek-v4-pro-0813`, only with `reasoning.max_tokens` ≤ 4000 | `max_tokens` 8000 | ≈ $0.0015 / ≈ $0.007 |
| **Critic / sketch judge** | `claude-haiku-5-5` | `anthropic/claude-haiku-5.5` (`openai/gpt-6-luna` costs the same but is untested: one call used 9.8k tokens) | effort `low`, `max_tokens` 6000; ask for JSON and treat `finish_reason=length` as a failure | ≈ $0.001 |
| **Extractor** | `claude-haiku-5-5` | same | effort `low`, `max_tokens` 4000 | ≈ $0.001 |
| Labeler | keep `claude-opus-5-5` | same | effort `medium`, `max_tokens` 16000 | ≈ $0.08 per batch |

### Notes
- **The `hot` operator cannot run on Claude.** Every current Claude model rejects a non-default temperature, and `diverge/llm.py` already drops the temperature on the Anthropic path. That leaves `gemini-4-argon` (or DeepSeek or Llama) as the only valid hot implementer. `gpt-6.1-sol` takes no temperature either.
- **Identical retries.** At a fixed temperature, or with no temperature at all, a retry that resends the same prompt can come back with the same program. Retries should vary the prompt: add the attempt number, the previous error or truncation, and "different from: <previous program summary>". Not verified in code; this is the likely mechanism.
- **Truncation.** The Anthropic path currently sends neither `thinking` nor `effort`, so Sonnet 5.5 runs at effort `high`. Setting `medium` and capping `max_tokens` at 8–12k should reduce cut-off candidates. Lowering `max_tokens` from 16000 to 8000 also halves BENCH_DOCTOR's worst-case budget reservation, from $0.20 to $0.10 per call. That reservation is what cut every operator off at about a third of its $0.30 budget.
- **Llama 4 Maverick** is the cheapest and steadiest option (no reasoning, no truncation). Its weaker code quality makes it a diversity arm, not the main implementer.

### Uncertain
- How often each implementer produces *valid* code on these two evaluators. The log records cost and truncation, not evaluator validity by model, so the quality ranking (Sonnet 5.5 ≈ Argon > Haiku 5.5 > DeepSeek/Llama) is a judgment, not a measurement.
- How effort `medium`/`low` changes heuristic quality. Measure it with roughly 12 candidates per setting before adopting it.
- OpenRouter's reasoning-control mapping for Gemini 4 Argon (`effort` vs `max_tokens`) is not verified for this model.
- Every public benchmark figure above is from aggregators and vendor-reported.


## Sampler diversity

Researched 2026-10-18. Catalog ids and prices are from `cache/openrouter_models.json` (fetched 2026-10-18).
Measured costs come from `cache/llm.jsonl` sampling calls. Estimated costs assume 1k input and 3k-6k
output tokens per sample, because reasoning tokens dominate.

### What the evidence says

- **Models from different companies converge.** "Artificial Hivemind" (Jiang et al., arXiv 2510.22954,
  27 Oct 2025, NeurIPS 2025 best paper) studied 26K open-ended queries. It found that different models
  give answers more similar to each other than one model's own samples are. So cross-vendor agreement
  is the normal case, not proof that an answer is right. DIVERGE's 62-75% top-3 share fits this.
- **Sharing a company or base architecture makes models more alike.** "Correlated Errors in LLMs"
  (Kim et al., ICML 2025, arXiv 2506.07962) tested 350+ models. Where two models were both wrong, they
  gave the same wrong answer 60% of the time. Agreement rises when models share a provider or base
  architecture, and also as models get more accurate. More labs therefore add more than more models
  from one lab.
- **Distillation weakens "different lab" as a proxy.** On 2026-02-23 Anthropic said DeepSeek (>150k
  exchanges), Moonshot (>3.4M) and MiniMax (>13M) had distilled Claude, with the most volume on coding
  ([anthropic.com/news/detecting-and-preventing-distillation-attacks](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)).
  Kimi and MiniMax may therefore repeat Claude's modes. That is an inference from the claim, not a
  measurement.
- **Code specifically.** "How Diversely Can LMs Solve Problems?" (arXiv 2503.00691, Mar 2025,
  EMNLP Findings 2025) found that model code is less algorithmically diverse than human code. Mixing
  models and using temperature >1 both raised diversity, at some cost to correctness.

### Temperature, min-p, paraphrase

- **Temperature does nothing on 2 of the 5 current samplers.** The catalog lists no `temperature`
  for `openai/gpt-6.1-sol` or for any Claude from 4.7 on, and `diverge/llm.py:245` drops it on the
  Anthropic route. `sampling.temperature: 1.0` therefore only acts on Gemini, Llama and DeepSeek.
  Temperature >1 adds diversity but also broken code (2503.00691; DIVERGE's `hot` calibration).
- **Min-p is weakly supported.** Its benefit claim (Nguyen et al., ICLR 2025) was disputed in
  "Min-p, Max Exaggeration" (arXiv 2506.13681, Jun 2025). Few frontier endpoints accept `min_p`. Skip it.
- **Paraphrase ensembles work, but they measure a different thing.** DIVERGE's own test found that
  paraphrasing moves the distribution more than resampling does. Recommendation: send every model the
  same fixed set of 4 paraphrases × 5 samples. Then call a family "consensus" only if it leads under
  every wording. That separates real convergence from prompt effects.
- **Verbalized sampling** (arXiv 2510.01171, ICML 2026) reports 1.6-2.1x diversity on creative
  writing. It is untested on code. DIVERGE's `verbalized` operator exists but was not benchmarked.

### Recommended panel (7 models, N=20 each)

| id | N=20 cost | why it adds diversity |
|---|---|---|
| anthropic/claude-sonnet-5.5 | ~$0.50-0.66 (measured) | Anthropic lineage |
| openai/gpt-6.1-sol | ~$0.71 (measured) | OpenAI lineage |
| google/gemini-4-argon | ~$0.41 (measured) | Google lineage (do not also add Gemma, same lineage) |
| deepseek/deepseek-v4-pro-0813 | ~$0.13 (measured) | Chinese open-weight lineage; cheap |
| qwen/qwen3.7-plus | ~$0.08-0.16 (est.) | Alibaba lineage, strong coder; `qwen3.8-max-0902` costs ~$0.40-0.76 |
| mistralai/mistral-large-4-0 | ~$0.14-0.26 (est.) | European data and training lineage |
| x-ai/grok-4.7 | ~$0.40-0.76 (est.) | xAI, separate US lineage |

The panel costs about **$2.4-3.7 per task**, against about $1.9 for the current 5. Optional 8th:
`z-ai/glm-5.3`. Its 3 pilot calls averaged $0.048 each, so N=20 is about $1, and it reasons for a
long time. **Drop Llama 4 Maverick from consensus.** Its rare families were mostly broken or trivial
code, which inflates "blind spots". If it is kept, filter its samples by validity first. Deprioritize
Kimi K3, MiniMax M3 and Nemotron, which are distilled from other models or trained on synthetic data.
Pilot all new ids with 3 samples before N=20.

### Anthropic-only mode (`DIVERGE_PROVIDER=anthropic`)

A single-vendor panel **cannot measure cross-company consensus**. Correlated-errors evidence says
models from one provider agree more, so the shared assumptions and blind spots this mode reports are
"Claude's defaults", not "the field's defaults". Label the output that way. The best diversity it can
reach:

- Mix model tiers: `claude-haiku-5.5` (~$0.02-0.06/N=20), `claude-sonnet-5.5` and `claude-opus-5.5`
  (~$1.3-2.5). Use `claude-fable-5.1` (~$3-6) only if the budget allows.
- Vary `reasoning_effort` (low/high), since temperature is unavailable.
- Use the fixed paraphrase set, and verbalized sampling (ask for k approaches, each with a probability).
- Older models that still accept temperature (`claude-sonnet-4.5`/`4.6`, `opus-4.5`/`4.6`) add
  generational spread if the Anthropic API still serves them.
