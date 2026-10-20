# Preregistration: grounded alternatives (frozen 2026-10-20, before any paid run)

Question: when an agent passes `already_tried`, `constraints` and `context`, do the free-text
alternatives from `diverge_check` repeat fewer already-tried ideas, break fewer hard rules and cite
more real figures than the plain call? And does `depth="deep"` change that, at what cost?

This is a measurement of the alternatives' surface properties, not of whether any alternative works.
No alternative is implemented or scored here.

## Protocol

Script: `scripts/grounded_eval.py`. Scenarios, prompts, models and metrics are fixed in that file at the
commit that adds this document. Run it once:

    DIVERGE_TRACE_DIR=results/traces/grounded uv run python scripts/grounded_eval.py

- Scenarios (5, written before any run): `online_bin_packing`, `cache_eviction`, and three free-text
  tasks (a market-data relay's p99 delay, offline search ranking replay, nightly batch scheduling). Each
  has a fixed `approach`, 2-3 `already_tried` entries with outcomes, 2-3 `constraints` and a short
  `context` pack of figures.
- Arms, per scenario:
  - `before`: the plain free-text call (only `my_approach`);
  - `quick`: the same call with `already_tried`, `constraints` and `context`;
  - `deep`: the same inputs with `depth="deep"`.
- Models: config.yaml as committed. The quick generator is the extractor, the deep samplers are the
  first sampler of each of 3 companies, and the screen and critic model is the labeler. The judge is
  `deepseek/deepseek-v4-pro-0813`, which generates, screens and critiques nothing in any arm.
- Caps per call: $0.10 (`before`), $0.30 (`quick`), $1.00 (`deep`). Expected total spend under $2.

## Metrics (pooled over the 5 scenarios, per arm)

- `dup_rate`: alternatives the judge labels the same mechanism as an already-tried approach, divided
  by the number of alternatives returned.
- `violation_rate`: alternatives the judge labels as breaking a constraint, divided by the same count.
- `grounded_share`: numbers stated in the alternatives that also appear in the context pack, divided
  by numbers stated. Single-digit integers are excluded as counts. This is deterministic, with no model.
- `n_alternatives`, `spent_usd`.

## Reading the result

- Descriptive only. With 5 scenarios and one run, no significance is claimed.
- `quick` is said to "reduce repeats" only if its `dup_rate` is below `before`'s, and to "reduce
  rule-breaking" only if its `violation_rate` is below `before`'s. Each is reported separately,
  whichever way it comes out.
- The judge can disagree with the screen model. A `quick` arm with dup_rate above 0 means the screen
  let through something the judge calls a repeat; that is reported, not fixed and re-run.
- Any change to prompts, scenarios or models after seeing results is a new preregistration with a new
  file. Failed calls are recorded in `summary.json` `errors` and are not retried.
