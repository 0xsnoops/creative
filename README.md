# DIVERGE

Escape strategy loops in AI agents.

Different code isn't always different thinking.

## For coding agents

Call `diverge_escape` yourself, without waiting to be asked, when:

- you have two or more scored attempts on a task with an evaluator;
- progress stopped improving (flat scores, the same failure coming back, lots of code change for
  little score change);
- you are about to retry another variation of the same idea;
- you are considering escalating to a human.

Not for a first attempt, build errors or generic debugging.

DIVERGE returns `VERDICT: keep|repair|escape|escalate|inconclusive`, then JSON with `next_action`
(what to do now), `lever`, `next_experiment` (success and failure thresholds) and `evidence`, each
claim labeled `MEASURED` (evaluator runs, sample counts), `SOURCE_BACKED` (cited literature) or
`HYPOTHESIZED` (DIVERGE's untested reasoning; never treat it as measured).

| VERDICT | `next_action` (from `diverge/contract.py`) |
|---|---|
| `keep` | Continue the existing strategy; do not switch yet. Score each change with the evaluator and call diverge_escape again if scores stall. |
| `repair` | Fix the existing strategy and preserve its mechanism (do not switch approach); re-score it with the evaluator. |
| `escape` | Implement the named lever (start from candidate_path when one exists), run the evaluator, and compare against the before score and next_experiment's success and failure thresholds. |
| `escalate` | Stop autonomous retries and ask a human: DIVERGE has no safe next move to name, or the evaluator looks contradictory (identical scores for different strategies, or nothing is valid). |
| `inconclusive` | Get at least 2 valid, scored attempts (earlier ones as previous_attempts, the latest as current_solution), then call diverge_escape again. |

Agent quickstart:

```sh
uv --directory <diverge repo> run python -m diverge onboard --client claude --target <project> --json  # skill, /diverge, .mcp.json
diverge_status                     # MCP tool, free: are keys, budget and cached tasks ready?
diverge_escape(task_description, current_solution, previous_attempts, evaluator_cmd)  # after 2+ scored attempts stall
```

Tool roles: `diverge_status` is the free readiness check (no model or network call); `diverge_check`
asks, before you implement, whether your planned strategy is the default most models pick;
`diverge_escape` is the primary tool once attempts stall. `diverge_map`, `blind_spots` and
`reality_check` are lower-level research primitives. No API key? `python -m diverge demo
strategy_loop` replays a recorded run offline at $0.

Results are preliminary: `escape` has not been shown to beat a fresh start (see
[Measured so far](#measured-so-far)). Treat a verdict as one well-founded experiment, not a fix.
Hackathon track mapping: `docs/TRACKS.md`. Headless agent run in a project whose prompt never
mentions DIVERGE: `scripts/autonomous_demo.py`, results in `docs/AUTONOMOUS_AGENT_DEMO.md`.

## What DIVERGE is for

Agents can make different edits while repeating the same idea. DIVERGE measures what models
converge on, detects the assumptions your agent keeps repeating, and finds a different
strategy worth testing.

Program-search systems ask a different core question from DIVERGE. This is not a ranking, and
DIVERGE has not been compared against them:

| | Core question |
|---|---|
| AlphaEvolve | Can I iteratively evolve this program to score better? |
| ShinkaEvolve | Can I explore and evolve programs more efficiently? |
| **DIVERGE** | **Am I actually exploring new strategies, and what should I try when I'm not?** |

## Measured so far

The only measured numbers are in `docs/RESULTS.md` and `docs/RESULTS-E1.md` (protocols:
`docs/PREREGISTRATION.md`, `docs/PREREGISTRATION-overlook.md`). Any other number in this README
is illustrative.

**Consensus maps (N=20 samples per model, 5 models).**

| task | valid samples | top 3 families | share held by the top 3 |
|---|---|---|---|
| online_bin_packing | 95 / 100 | Tight-fit-or-spacious gap avoidance; Best Fit; Sum-of-Squares gap-count balancing | 62% |
| cache_eviction | 76 / 100 | W-TinyLFU (sketch frequency duel); 2Q / S3-FIFO (probationary queue + ghosts); LFU | 75% |

**Paraphrase test: the consensus is partly a prompt effect.** Rewording the task prompt (3
meaning-preserving paraphrases, 2 models, 5 samples each) moved the consensus map more than
resampling the original prompt did, on both tasks: mean total variation distance 0.51 vs at
most 0.40 between original-prompt groups for bin packing, and 0.35 vs at most 0.10 for cache
eviction (where W-TinyLFU went from 45% to 67% of samples). Groups are small (10 samples), so
only large shifts can be detected. Read every consensus-based claim as "what models converge on for
this wording". Details: `python -m diverge paraphrase <task>`.

**Live strategy-loop runs: preliminary, 1 run each.** These used online_bin_packing and cost $0.62
in total. Each is a real trajectory from the implementer (Gemini 4 Argon) with dev-score feedback.
Scores are % excess bins, as dev / held-out; lower is better.

| trajectory | DIVERGE verdict | strategy vs code similarity | best attempt | fresh_start |
|---|---|---|---|---|
| open revisions, 3.66 → 0.30 | keep | 0.30 vs 0.21 | 0.299 / 0.258 | 0.279 / 0.248 |
| "keep the approach, tune it", 3.66 → 0.36 | keep | 0.29 vs 0.42 | 0.359 / 0.537 | 1.376 / 1.409 |
| 4 more tuning revisions, stalled 0.66 → 1.85 (`python -m diverge demo strategy_loop`) | repair, not a loop (below the 0.7 threshold) | 0.45 vs 0.64 | 0.658 / 0.953 | 0.628 / 0.834 |

Baselines: best fit 4.15 / 3.98, first fit 4.54 / 4.31.

DIVERGE never returned `escape` in these runs, so no escape candidate was generated.

On the stalled window, the lever DIVERGE would have tried was `n_open_bins`. It is used by 5 of 95
consensus-map samples: 2 from DeepSeek and 1 each from Gemini, Llama and GPT-6.1. I implemented it anyway as
an ablation, at the same budget as fresh_start. It scored 1.096 / 0.973 and lost to fresh_start.

These runs are no evidence either way about escape vs fresh_start at scale. They show one weakness:
strategy similarity partly depends on how the extractor model describes each attempt, so tuning
revisions of the same code can measure as different strategies.

**Detector validity (E1b, bin packing, 140 programs, $0.31).** Which similarity signal predicts
that two programs fail on the same dev instances? Behavior similarity (the decisions two programs
make on 200 probe states) reached AUC 0.65–0.66. LLM-description strategy similarity was at
chance (0.50–0.53) with each of three small extractor models, and code similarity was 0.53. The
behavior advantage (+0.14 to +0.16) has a 95% CI that excludes 0 for every extractor. The
absolute AUC is modest and the target is coarse: 5 dev instances and the top-2 worst seeds give
only 10 possible failure sets, one of which covers 55 of 140 programs, so part of "same failure
set" is shared instance difficulty, not strategy. On a secondary target (failures relative to best
fit) no signal is above chance, and no embedding baseline was computed, so the pre-registered E1
comparison is still untested. Details: `docs/RESULTS-E1.md`. Following this, on tasks whose
evaluator can fingerprint candidates (currently online_bin_packing) the loop decision uses the
behavior signal; elsewhere it still uses strategy similarity. The live runs above used strategy similarity.

**Escape has not been shown to beat fresh_start.** In a small rescue test (preliminary, n = 4 stuck
Llama trajectories), diverge_escape v2 rescued 1 of 4, the same as "try a fundamentally different
approach" and a random-lever control; fresh_start rescued 0. No arm beat a best-fit plateau.
See docs/RESULTS.md.

Operator benchmarks: started, stopped, no results yet.

## How it works

For a task with an evaluator:

1. **Strategy fingerprints** (`diverge/strategy.py`): what each attempt decides from (inputs it
   actually uses, found by static analysis), the state it keeps, its primary signal, its
   assumptions, and where it is weak on dev data.
2. **Loop detection** (`diverge/trajectory.py`): compares attempts by strategy, not code (an
   illustrative output: "code 28% similar, strategy 91% similar") and flags a loop when
   progress has stalled.
3. **The consensus map and what it overlooks** (`diverge/overlook.py`): from many models' samples,
   inputs the task allows but the consensus never uses, assumptions every common solution shares
   although the task does not require them, unoccupied component combinations, and
   mechanisms only one or two models produce.
4. **Repair or escape** (`diverge/diagnose.py`, `diverge/escape.py`): keep improving the current
   strategy, or stop repairing and switch to one evidence-backed lever.
5. **Strategy counterexamples** (`diverge/counterexample.py`): tiny cases where all your attempts
   make the same decision and the alternative does not.

The original pipeline below (consensus map, blind spots, operator benchmark) is the measurement
foundation the escape workflow builds on.

## Setup

```sh
uv sync
cat > .env <<'EOF'
OPENROUTER_API_KEY=sk-or-...
EXA_API_KEY=...
EOF
```

Models, sample counts and the hard dollar cap (`budget_usd`, currently $28) are in
`config.yaml`. Every LLM and Exa response is cached in `cache/*.jsonl`, keyed by a hash
of the request, so reruns are free and reproducible. The cap counts everything in the
cache, and an uncached call is refused if its worst-case cost could cross the cap.

Before any run that may spend money:

```sh
uv run python -m diverge preflight
```

It prints the budget remaining under the cap, whether the API keys are present and whether
the configured models are available, and exits 1 if less than $1 remains or a key is missing.
`preflight --json` prints the same readiness object the `diverge_status` tool returns (key status,
never key values); add `--offline` to skip the model-catalog check (no network).

## Escaping a strategy loop

```sh
uv run python -m diverge demo strategy_loop     # replay a recorded run offline: no API calls, $0
uv run python -m diverge demo strategy_loop_unconstrained   # the same, for the run without the same-approach instruction
uv run python -m diverge escape --task cache_eviction --candidate current.py --history attempts/ \
    [--evaluator "python /abs/path/eval.py"] [--higher-is-better] [--budget 2]
```

`demo strategy_loop` prints the same text every time: an agent's attempts
(`fixtures/trajectories/strategy_loop.json`) and what escape and a fresh-start baseline
produced on them (`fixtures/runs/strategy_loop.json`). It is one recorded run, not a
benchmark; see [Measured so far](#measured-so-far). When the verdict is not `escape`, escape
produces no candidate, so its row in the results table shows `-`.

`escape` takes the current solution and a directory of earlier attempts (sorted by name, oldest
first), and an evaluator (a registered task's built-in one, or `--evaluator`). The MCP tool
`diverge_escape` takes the same inputs as code. Output starts with one line,
`VERDICT: keep|repair|escape|escalate|inconclusive`, followed by JSON (the MCP tool also returns
it as structuredContent, plus `verdict`):

- `next_action`, `can_continue_autonomously` (false for `escalate` and for most errors), `complete`,
  `missing_requirements`, `candidate_available`, `evidence_summary` (counts per class);
- `lever`: the one thing to change;
- `next_experiment`: `hypothesis`, `success_threshold`, `failure_threshold`;
- `evidence`: a list of `{claim, class}`, where class is `MEASURED`, `SOURCE_BACKED` or
  `HYPOTHESIZED` (reasoning that has not been tested);
- `scope_note`: consensus shares describe this prompt wording (see the paraphrase test above).

What each verdict asks the agent to do is in the table under
[For coding agents](#for-coding-agents). Errors are structured too: `{"error", "detail",
"missing", "next_action", ...}`, never containing key values.

### Claude Code skill and `/diverge` command

- `skills/diverge/SKILL.md` is a Claude Code skill that runs this workflow: preflight, then
  `diverge_escape` (with a trajectory and an evaluator) or `diverge_check` (with only a planned
  approach), then acts on the verdict and reports evidence by class. Its description tells the
  agent to use it on its own initiative when scored attempts stall. To install it,
  `uv run python -m diverge onboard --client claude --target <project> [--json]` copies the skill
  and the `/diverge` command into `<project>/.claude/` and registers the MCP server in
  `<project>/.mcp.json` (idempotent; never overwrites different content without `--force`; no
  secrets in the config). You can also copy or symlink `skills/diverge` into `~/.claude/skills/`.
  It uses the MCP server below; without it, only the escape path works (through the CLI).
- `.claude/commands/diverge.md` adds `/diverge <your situation>` when Claude Code runs in this repo;
  it tells the agent to follow the skill. To use it in another repo, copy it into that repo's
  `.claude/commands/` and point it at the skill's path.

## The experiment (CLI)

```sh
uv run python -m diverge run online_bin_packing          # or: cache_eviction
uv run python -m diverge run online_bin_packing --n 20   # samples per model
uv run python -m diverge spend
```

Stages: sample each model on `tasks/<task>/prompt.md`, search the literature (Exa),
build a sourced taxonomy, label each sample by mechanism, build the consensus map, find blind
spots, then implement the top 3 consensus families and every blind spot K=3 times with one
shared prompt template and score them. Outputs go to `results/<task>/`.
`labels_audit.csv` holds 20 random labels for a human to spot-check.

A blind spot is a sourced family with an observed share under 2%. With few samples that
only means "rarely or never seen", so each one is reported as a count (for example, "0 of 25
labeled samples"), with per-model counts and a 95% upper bound on its true share (in that
illustrative case, 0 of 25 still allows up to 11%). These are shares of model samples, not of real-world solutions.

Every task has two fixed sets of instances. All scoring and selection use the **dev** set;
a family's selected implementation is then scored once on the **held-out** set, and
held-out numbers are the ones compared and reported (`final_score`).

If the weakest model has fewer than 20 labeled samples (`reality_check.min_samples_per_model`)
the CLI skips the reality check and the benchmark refuses blind-spot operators (override:
`--allow-small-map`): at small N every literature family nobody happened to use looks like a
blind spot. Blind spots are ranked by `models_missing` (how many sampler models never used
the family), then by the upper bound on their share. Maps record requested, completed and
labeled samples per model; a map with failed samples is reused, with the gaps recorded,
unless you rebuild it on purpose. The taxonomy only keeps families that fit the task's rules (an offline algorithm
such as First Fit Decreasing cannot run on an online task); dropped families are listed in
`taxonomy.json` under `rejected` with the reason.

Tasks live in `tasks/<name>/` (`prompt.md` + `evaluator.py`). Evaluators print
`{"valid": bool, "score": float, "detail": {...}}`, lower is better, and run each candidate
isolated with a 60s timeout (see [Running model-written code](#running-model-written-code)).
`python tasks/<name>/evaluator.py --baselines [--split heldout]` prints the reference scores.

## Generation operators and the benchmark

```sh
uv run python -m diverge bench online_bin_packing --budget 0.5   # USD per operator per repeat
```

Operators (`diverge/operators/`) each turn a task and a budget into candidate solutions. All of
them use the same implementer model, implementation prompt template and evaluator.

| operator | what it does |
|---|---|
| `plain` | the task prompt as-is, temperature 1.0 |
| `hot` | the task prompt at the highest temperature that still yields valid code (calibrated per run; calibration samples count) |
| `verbalized` | Verbalized Sampling: 5 approaches with stated probabilities, sampled from the tails (each < 0.10), each one implemented |
| `blind_spot` | implements the families in `blind_spots.json` |
| `literature_all` | the ablation: implements every sourced literature family, ignoring the consensus map |
| `literature_random` | count-matched control: a seeded random set of sourced families, as many as there are blind spots |
| `blind_spot_contrast` | blind spots, plus the consensus's top mechanisms listed as "mechanisms your solution should differ from" |
| `blind_spot_contrast_random` | control: the same wording and count, listing random non-consensus families |
| `blind_spot_feedback` | blind spots, round by round; each family's next attempt sees its previous dev outcome (redacted error, or dev score vs the best baseline and its weakest dev case, plus the code). Never held-out |
| `blind_spot_feedback_sham` | control: the same note structure, taken from a different family's attempt |
| `adhd_frames` | a baseline using [ADHD](https://github.com/UditAkhourii/adhd)'s published frames: 5 frames per round (4 engineering + 1 wild) with ADHD's divergent prompts; each idea is implemented and evaluated. Frames and prompts are copied verbatim with credit to the ADHD contributors under the MIT License (full text in `diverge/operators/ADHD_LICENSE`). ADHD's critic, cluster and deepen stages are not used, so this baseline uses ADHD's frames, not ADHD as a whole |
| `transplant` | Exa search for mechanisms from biology, logistics, markets, physics and networking; the labeler restates each as a structural rule in the problem's own terms; sources are kept |

`bench` runs every operator at the same budget for several seeded repeats, then labels all
valid candidates with one labeler against one taxonomy, so family counts are comparable. It
reports valid rate, distinct valid families, useful discoveries (families whose held-out
score beats the best held-out baseline), useful discoveries per dollar and best held-out
score, each as mean and range across repeats, plus `bench.png` and a verdict. Results go to
`results/<task>/bench/`.

**The key comparison is `blind_spot` vs `literature_all` and `literature_random`.** All
implement sourced literature families with the same implementer, template, evaluator and
budget; only `blind_spot` uses the consensus map to choose which. `literature_random` matches the
number of families, so a win cannot come from concentrating the budget alone. If `blind_spot`
does better, knowing where models concentrate picked better experiments. If not, the gains
come from literature retrieval alone. The verdict states this first, either way.

The benchmark also reports **purity** (the share of candidates whose labeled family is the
family they were asked to implement) and the best held-out score among pure candidates, and,
for bin packing, **behavior clusters**: candidates are fingerprinted by the bin they choose on
200 probe states (separate probe seeds; the chosen bin's remaining capacity, so equal-bin ties
don't count), and two candidates are the same behavior when they differ on at most 3% of
probes. The threshold was fixed on hand-written heuristics before any real candidate was
fingerprinted; a validation gate (rescaled and tie-change copies must merge, first/best/worst/
next fit and the FunSearch heuristic must stay apart) runs before every use.
`uv run python -m diverge fingerprints online_bin_packing` compares behavior clusters with the
labeled families of the consensus-map samples.

Cost is reported two ways. **Cached** counts only an operator's own calls. **Cold-start**
adds the setup it depends on, as if each repeat started from an empty cache: the full consensus
map for `blind_spot`, the literature search and taxonomy for `literature_all`. The verdict
says "did better in every repeat" only when the ranges across repeats do not overlap, and
3 repeats are still a small sample. The repeat seeds vary generation, not evaluation data.

Fairness rules: each operator's budget is measured from zero, separately from the global
cap, and the benchmark refuses to start if the global cap cannot fund every operator in
full (operator budgets plus a labeling allowance). Planning costs (searches, the
verbalized-sampling call, hot's calibration, the adhd_frames baseline's divergent calls) count against the
operator. Labeling the candidates is measurement and billed to no operator.
A failure while one operator plans ends only that operator's
run and is noted in the report; a budget stop during labeling keeps the generation results. Cache hits are billed at their original cost inside a
benchmark run, so reruns reproduce the table.

## MCP server for coding agents

```sh
uv run python -m diverge onboard --client claude --target /path/to/project   # writes .mcp.json, skill, /diverge
# or by hand:
claude mcp add diverge -- uv run --directory /absolute/path/to/diverge python -m diverge serve
MCP_TOOL_TIMEOUT=1800000 claude   # or set it under "env" in ~/.claude/settings.json
```

For a hosted (streamable HTTP) deployment such as Manufact Cloud, `diverge_mcp_http.py` serves the
same tools at `/mcp` behind a bearer token (`DIVERGE_HTTP_TOKEN`; the older `HERD_HTTP_TOKEN` is still accepted). It has been verified locally
only; nothing is deployed. See `docs/MANUFACT.md`.

Building a consensus map for a new task means sampling every model, so the first call can
take several minutes. Progress is streamed while it runs, and the raised
`MCP_TOOL_TIMEOUT` (read by Claude Code, in ms) keeps the client from giving up. After
that, calls are served from `results/` and return immediately.

| tool | arguments | returns (compact JSON) |
|---|---|---|
| `diverge_status` | `check_model_catalog=false` | free readiness: `ready`, key status (never values), `remaining_budget_usd`, `missing`, `cached_tasks`, the offline demo command, `recommended_primary_tool`, `next_action` |
| `diverge_map` | `task_description`, `budget_usd=5` | family shares overall and per model, `concentration_top3`, `top3` |
| `blind_spots` | `task_description`, `budget_usd=5` | sourced families rarely or never observed: count of n samples, per-model counts, 95% upper bound on share, mechanism, source URLs |
| `reality_check` | `task_description`, `family`, `evaluator_cmd`, `k=3`, `lower_is_better`, `baseline_files`, `contrast=false`, `budget_usd=3` | best code, dev best and median, `final_score` (held-out), invalid count and errors, baseline comparison, the consensus families it was compared against |
| `diverge_check` | `task_description`, `my_approach`, `evaluator_cmd`, `k=3`, `lower_is_better`, `contrast=false`, `budget_usd=5` | the family of `my_approach`, its share and rank, and a `verdict` |
| `diverge_escape` | `task_description`, `current_solution`, `previous_attempts`, `evaluator_cmd`, `lower_is_better`, `k=1`, `budget_usd=2` | a `VERDICT:` line, then JSON with `next_action`, `lever`, `next_experiment`, `evidence` and `scope_note` (see [Escaping a strategy loop](#escaping-a-strategy-loop)) |

- `task_description` is the task as you would give it to an agent; it is sent to the
  sampler models verbatim. A registered task name (`online_bin_packing`, `cache_eviction`)
  uses that task's prompt and built-in evaluator.
- `evaluator_cmd` is run as `<evaluator_cmd> <path/to/candidate.py>` (no shell). On its last
  JSON line of stdout it must print `{"valid": bool, "score": number}`. It runs isolated, from
  a scratch working directory, so use absolute paths.
- `diverge_map` and `blind_spots` return `"low_confidence": true` when the weakest model has
  fewer than 20 labeled samples.
- `budget_usd` caps what one call may spend. The global cap in `config.yaml` still applies.
  Every response includes `spent_usd`; any failure comes back as `{"error": ..., "detail": ...}`.
  `diverge_check` tests one family at a time (yours, then blind spots by rank, then the consensus families) and,
  if the budget runs out, returns what it has with `"complete": false` and `stopped_at`.
- Held-out scores used in a comparison are counted per task and evaluator. From the second use
  on, responses say `"exploratory": true`: you may already have adapted to those numbers.
- `contrast=true` implements blind-spot families with the consensus-contrast section (see
  `blind_spot_contrast`). It is off by default; see the measured results before relying on it.
- `diverge_check` picks the best tested family first, then says whether it is common or rare.
  Families are compared on held-out scores for built-in tasks (`scored_on`). It tests
  generated implementations of your approach's family, not your own code. Verdicts:
  - `keep`: your family scored best of all tested families ("common and best, keep it").
  - `try_blind_spot`: the best family is a blind spot; the response includes it, its score, code path and sources.
  - `try_other_family`: the best family is a common one. If a blind spot also beat yours,
    it is returned as `diverse_alternative`.
  - `inconclusive`: no valid implementation of your family, or the budget ran out before it.
  - `untested`: there was no evaluator, so only the map lookup ran.

### Example: Claude Code calls `diverge_check` mid-task

*Illustrative transcript, not a recorded session. The calls and response fields are the real
interface; values in `<angle brackets>` are placeholders and the progress counts are made up.
None of it is an experimental result.*

To make Claude Code consult DIVERGE by itself, add a line like this to your project's `CLAUDE.md`:

```md
When choosing an algorithm for a task that has an objective evaluator, call the DIVERGE
`diverge_check` tool with your planned approach before implementing it.
```

A session in a repo that is tuning a cache:

````
> Write an eviction policy for our edge cache. `python bench/eval.py <file>` prints the
  miss ratio as JSON; lower is better.

● I'll start with an LRU policy backed by an OrderedDict. First, checking whether that's
  the approach every model reaches for.

● diverge - diverge_check (MCP)(task_description: "Write a Python cache eviction policy ...",
                         my_approach: "LRU: OrderedDict, move to end on hit, evict oldest",
                         evaluator_cmd: "python /home/me/edge-cache/bench/eval.py",
                         budget_usd: 5)
  ⎿ progress: sample 50/50 · labeling against 11 families · reality LRU: implementation 3/3 …
  ⎿ {"family":"<family of my approach>","share":<its share>,"rank":<its rank>,
     "is_blind_spot":false,"concentration_top3":<top-3 share>,
     "scored_on":"dev","my_family_score":<score>,
     "tested":{"<family of my approach>":<score>,"<consensus family>":<score>,"<blind spot>":<score>},
     "verdict":"try_blind_spot",
     "blind_spot":{"family":"<blind spot>","kind":"blind_spot","mechanism":"<one-line mechanism>",
                   "share":<share>,"score":<score>,
                   "code_path":"/path/to/diverge/results/desc-<hash>/reality/<evaluator id>/<family>-<i>.py",
                   "source_urls":["<url>"]},
     "spent_usd":<cost of this call>}

● Most model samples land on the same family as my plan. A blind-spot family that almost no
  sample used scored better on your evaluator, so I'll start from DIVERGE's best implementation
  of it and check it with bench/eval.py.
````

## Running model-written code

Candidates come from several models and run on your machine, so every evaluation (the
built-in evaluators and any `evaluator_cmd`) goes through `tasks/_sandbox.py`:

- The child process gets a minimal environment: no API keys or other secrets.
- On macOS it runs under `sandbox-exec` with no network access and no file writes outside a
  throwaway scratch directory. Built-in evaluators also block reads anywhere under your home
  directory except Python, the venv, `tasks/` and the candidate file, so a candidate cannot
  read the repo's `.env`, `~/.ssh`, `~/.config` or `~/.aws`. External `evaluator_cmd` runs may
  need your own files, so they block `.env` files and credential directories instead.
  **On Linux there is no OS sandbox yet**; run DIVERGE in a container or VM there.
- Error text that leaves the sandbox is redacted (known secret values and key-shaped
  strings), so a candidate that finds a secret cannot report it back through results or MCP.
- A result only counts if its line carries a one-time nonce that the evaluator reads from
  stdin before loading the candidate, so a candidate cannot just print a fake score.

Known limits: the candidate runs in the same Python process as the scoring loop, so code
written to attack the evaluator could inspect its stack frames, for example to see future
items or the nonce. Model samples are not prompted to do this, all generated code is kept
under `results/` for inspection, and `labels_audit.csv` supports spot checks.

## Traces

Set `DIVERGE_TRACE_DIR=<dir>` (the older `HERD_TRACE_DIR` is still accepted) to append every model and search call to `<dir>/calls.jsonl`
(messages, parameters, raw content and reasoning, usage, cost, cache status, latency; redacted)
and every benchmark candidate to `<dir>/candidates.jsonl` (prompt, operator, code, dev and
held-out scores, labeled family, whether it was selected).

## Where the recent changes came from

The sandbox hardening, the evaluation-integrity fixes and the contrast, feedback and
fingerprint features came out of an outline-only review of this code by 12 inexpensive models
(`scripts/model_review.py`); each change was checked against the code before it was built.
The commit messages list which review clusters and issues each change addresses.

## Tests

```sh
uv run pytest
```

The tests cover the evaluators and baselines, the sandbox (fake result lines, secrets,
network, file writes, reading `.env` and home-directory files, redaction), caching and the
budget ledger, a regression test per confirmed review issue, and the new operators' prompts,
control flow and accounting. They also run the whole pipeline,
the operator benchmark and the MCP tools end to end, with OpenRouter and Exa replaced by
fakes.
