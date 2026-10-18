# Pre-registration: final rescue measurement (written 2026-10-18 before the run)

**Question.** Does the new escape mode (`escape_v3`) rescue a stuck agent more often than simple
alternatives at the same budget? `escape_v3` adds an anchor-stripped failure summary, 5 isolated
sketches, a critic, and deepening of the top 2.

**Stuck states.** The 4 naturally stuck trajectories from the first rescue test (states 0, 1, 3, 4).
- Agent model: Llama 4 Maverick, 6 attempts each with dev feedback.
- Stuck means no dev improvement of more than 1% over the last 3 attempts.
- The trajectories are reused unchanged from `results/rescue/trajectories.json`.

**Arms.** Every arm uses Llama 4 Maverick as the implementer.
- `fresh_start`: fresh context plus a failure summary, with no code.
- `try_different`: the previous attempts plus the instruction "try a fundamentally different approach".
- `random_lever`: the escape v2 pipeline with a random input the consensus already uses (control).
- `herd_escape`: escape v2, the current default.
- `herd_escape_v3`: the new mode, opt-in.

**Seeds and budget**
- **Fresh seeds:** every arm uses new generation sample tags, `--round 2` in `scripts/rescue_test.py`. No candidate from the first rescue test is reused.
- **Same budget:** each arm gets at most 3 dev-scored candidates and a $0.40 cap. v3 deepens 2 sketches; admission resamples do not count as scored candidates.
- **Total budget:** at most $4.

**Selection and outcome.** Each arm's candidate is selected on dev only. Held-out is computed once,
for each arm's dev pick and for each state's stuck best. A state counts as rescued for an arm when
the arm's held-out score is strictly better than the stuck best's held-out score.

**Success rule**
- `escape_v3` becomes the default only if it rescues more states than both `try_different` and
  `random_lever` in this run.
- Otherwise v2 stays the default, v3 stays an opt-in mode, and the tie or loss is reported plainly.

**No reruns to chase a better number.** The run happens once. It is reported in docs/RESULTS.md
whatever it shows, labelled preliminary with n = 4.

**Known limitation.** With n = 4, one state changes any count. This measurement can rule out a
large advantage at best. It cannot show a small one.

## Amendment (07:55 Pacific, before the run)

The user disabled OpenRouter and asked that every call go through their Anthropic API key, so
Llama 4 Maverick is not available as the implementer. Changes:
- **Implementer:** every arm uses `anthropic/claude-haiku-5.5` (`DIVERGE_PROVIDER=anthropic`).
- **Supporting roles:** the extractor and critic are Claude models, chosen by `anthropic_only_models`.
- **Unchanged:** the 4 stuck states are reused exactly as recorded. They are still Llama
  trajectories, so the stuck agent and the rescuing implementer now differ for every arm alike.

Arms, fresh seeds, the budget and the success rule are unchanged.
