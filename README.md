# HERD

> Search engines find what exists. HERD finds what AI overlooks, and tests whether it's better.

HERD samples many AI models on a task. It then maps which families of solutions the
samples converge on, finds families from the literature that almost no sample uses
(blind spots), and implements and scores those families with an objective evaluator.

## Setup

```sh
uv sync
cat > .env <<'EOF'
OPENROUTER_API_KEY=sk-or-...
EXA_API_KEY=...
EOF
```

Models, sample counts and the hard dollar cap (`budget_usd`, currently $25) are in
`config.yaml`. Every LLM and Exa response is cached in `cache/*.jsonl`, keyed by a hash
of the request, so reruns are free and reproducible. The cap counts everything in the
cache, and an uncached call is refused if its worst-case cost could cross the cap.

## The experiment (CLI)

```sh
uv run python -m herd run online_bin_packing          # or: cache_eviction
uv run python -m herd run online_bin_packing --n 20   # samples per model
uv run python -m herd spend
```

Stages: sample each model on `tasks/<task>/prompt.md`, search the literature (Exa),
build a sourced taxonomy, label each sample by mechanism, build the herd map, find blind
spots, then implement the top 3 herd families and every blind spot K=3 times with one
shared prompt template and score them. Outputs go to `results/<task>/`.
`labels_audit.csv` holds 20 random labels for a human to spot-check.

A blind spot is a sourced family with an observed share under 2%. With few samples that
only means "rarely or never seen", so each one is reported as a count ("0 of 25 labeled
samples"), with per-model counts and a 95% upper bound on its true share (0 of 25 still
allows up to 11%). These are shares of model samples, not of real-world solutions.

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
uv run python -m herd bench online_bin_packing --budget 0.5   # USD per operator per repeat
```

Operators (`herd/operators/`) each turn a task and a budget into candidate solutions. All of
them use the same implementer model, implementation prompt template and evaluator.

| operator | what it does |
|---|---|
| `plain` | the task prompt as-is, temperature 1.0 |
| `hot` | the task prompt at the highest temperature that still yields valid code (calibrated per run; calibration samples count) |
| `verbalized` | Verbalized Sampling: 5 approaches with stated probabilities, sampled from the tails (each < 0.10), each one implemented |
| `blind_spot` | implements the families in `blind_spots.json` |
| `literature_all` | the ablation: implements every sourced literature family, ignoring the herd map |
| `literature_random` | count-matched control: a seeded random set of sourced families, as many as there are blind spots |
| `blind_spot_contrast` | blind spots, plus the herd's top mechanisms listed as "mechanisms your solution should differ from" |
| `blind_spot_contrast_random` | control: the same wording and count, listing random non-herd families |
| `blind_spot_feedback` | blind spots, round by round; each family's next attempt sees its previous dev outcome (redacted error, or dev score vs the best baseline and its weakest dev case, plus the code). Never held-out |
| `blind_spot_feedback_sham` | control: the same note structure, taken from a different family's attempt |
| `adhd_frames` | [ADHD](https://github.com/UditAkhourii/adhd)'s 5 frames per round (4 engineering + 1 wild) with its divergent prompts; each idea is implemented and evaluated. Frames and prompts copied verbatim (MIT, see `herd/operators/ADHD_LICENSE`); ADHD's critic/cluster/deepen stages are not used |
| `transplant` | Exa search for mechanisms from biology, logistics, markets, physics and networking; the labeler restates each as a structural rule in the problem's own terms; sources are kept |

`bench` runs every operator at the same budget for several seeded repeats, then labels all
valid candidates with one labeler against one taxonomy, so family counts are comparable. It
reports valid rate, distinct valid families, useful discoveries (families whose held-out
score beats the best held-out baseline), useful discoveries per dollar and best held-out
score, each as mean and range across repeats, plus `bench.png` and a verdict. Results go to
`results/<task>/bench/`.

**The key comparison is `blind_spot` vs `literature_all` and `literature_random`.** All
implement sourced literature families with the same implementer, template, evaluator and
budget; only `blind_spot` uses the herd map to choose which. `literature_random` matches the
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
`uv run python -m herd fingerprints online_bin_packing` compares behavior clusters with the
labeled families of the herd-map samples.

Cost is reported two ways. **Cached** counts only an operator's own calls. **Cold-start**
adds the setup it depends on, as if each repeat started from an empty cache: the full herd
map for `blind_spot`, the literature search and taxonomy for `literature_all`. The verdict
says "did better in every repeat" only when the ranges across repeats do not overlap, and
3 repeats are still a small sample. The repeat seeds vary generation, not evaluation data.

Fairness rules: each operator's budget is measured from zero, separately from the global
cap, and the benchmark refuses to start if the global cap cannot fund every operator in
full (operator budgets plus a labeling allowance). Planning costs (searches, the
verbalized-sampling call, hot's calibration, ADHD's divergent calls) count against the
operator. Labeling the candidates is measurement and billed to no operator.
A failure while one operator plans ends only that operator's
run and is noted in the report; a budget stop during labeling keeps the generation results. Cache hits are billed at their original cost inside a
benchmark run, so reruns reproduce the table.

## MCP server for coding agents

```sh
claude mcp add herd -- uv run --directory /absolute/path/to/herd python -m herd serve
MCP_TOOL_TIMEOUT=1800000 claude   # or set it under "env" in ~/.claude/settings.json
```

Building a herd map for a new task means sampling every model, so the first call can
take several minutes. Progress is streamed while it runs, and the raised
`MCP_TOOL_TIMEOUT` (read by Claude Code, in ms) keeps the client from giving up. After
that, calls are served from `results/` and return immediately.

| tool | arguments | returns (compact JSON) |
|---|---|---|
| `herd_map` | `task_description`, `budget_usd=5` | family shares overall and per model, `concentration_top3`, `top3` |
| `blind_spots` | `task_description`, `budget_usd=5` | sourced families rarely or never observed: count of n samples, per-model counts, 95% upper bound on share, mechanism, source URLs |
| `reality_check` | `task_description`, `family`, `evaluator_cmd`, `k=3`, `lower_is_better`, `baseline_files`, `contrast=false`, `budget_usd=3` | best code, dev best and median, `final_score` (held-out), invalid count and errors, baseline comparison, the herd families it was compared against |
| `herd_check` | `task_description`, `my_approach`, `evaluator_cmd`, `k=3`, `lower_is_better`, `contrast=false`, `budget_usd=5` | the family of `my_approach`, its share and rank, and a `verdict` |

- `task_description` is the task as you would give it to an agent; it is sent to the
  sampler models verbatim. A registered task name (`online_bin_packing`, `cache_eviction`)
  uses that task's prompt and built-in evaluator.
- `evaluator_cmd` is run as `<evaluator_cmd> <path/to/candidate.py>` (no shell). On its last
  JSON line of stdout it must print `{"valid": bool, "score": number}`. It runs isolated, from
  a scratch working directory, so use absolute paths.
- `herd_map` and `blind_spots` return `"low_confidence": true` when the weakest model has
  fewer than 20 labeled samples.
- `budget_usd` caps what one call may spend. The global cap in `config.yaml` still applies.
  Every response includes `spent_usd`; any failure comes back as `{"error": ..., "detail": ...}`.
  `herd_check` tests one family at a time (yours, then blind spots by rank, then the herd) and,
  if the budget runs out, returns what it has with `"complete": false` and `stopped_at`.
- Held-out scores used in a comparison are counted per task and evaluator. From the second use
  on, responses say `"exploratory": true`: you may already have adapted to those numbers.
- `contrast=true` implements blind-spot families with the herd-contrast section (see
  `blind_spot_contrast`). It is off by default; see the measured results before relying on it.
- `herd_check` picks the best tested family first, then says whether it is common or rare.
  Families are compared on held-out scores for built-in tasks (`scored_on`). It tests
  generated implementations of your approach's family, not your own code. Verdicts:
  - `keep`: your family scored best of all tested families ("common and best, keep it").
  - `try_blind_spot`: the best family is a blind spot; the response includes it, its score, code path and sources.
  - `try_other_family`: the best family is a common one. If a blind spot also beat yours,
    it is returned as `diverse_alternative`.
  - `inconclusive`: no valid implementation of your family, or the budget ran out before it.
  - `untested`: there was no evaluator, so only the map lookup ran.

### Example: Claude Code calls `herd_check` mid-task

*The calls and response fields are the real interface. Values in `<angle brackets>` are
placeholders, not experimental results.*

To make Claude Code consult HERD by itself, add a line like this to your project's `CLAUDE.md`:

```md
When choosing an algorithm for a task that has an objective evaluator, call the HERD
`herd_check` tool with your planned approach before implementing it.
```

A session in a repo that is tuning a cache:

````
> Write an eviction policy for our edge cache. `python bench/eval.py <file>` prints the
  miss ratio as JSON; lower is better.

● I'll start with an LRU policy backed by an OrderedDict. First, checking whether that's
  the approach every model reaches for.

● herd - herd_check (MCP)(task_description: "Write a Python cache eviction policy ...",
                         my_approach: "LRU: OrderedDict, move to end on hit, evict oldest",
                         evaluator_cmd: "python /home/me/edge-cache/bench/eval.py",
                         budget_usd: 5)
  ⎿ progress: sample 50/50 · labeling against 11 families · reality LRU: implementation 3/3 …
  ⎿ {"family":"<family of my approach>","share":<its share>,"rank":<its rank>,
     "is_blind_spot":false,"concentration_top3":<top-3 share>,
     "scored_on":"dev","my_family_score":<score>,
     "tested":{"<family of my approach>":<score>,"<herd family>":<score>,"<blind spot>":<score>},
     "verdict":"try_blind_spot",
     "blind_spot":{"family":"<blind spot>","kind":"blind_spot","mechanism":"<one-line mechanism>",
                   "share":<share>,"score":<score>,
                   "code_path":"/path/to/herd/results/desc-<hash>/reality/<evaluator id>/<family>-<i>.py",
                   "source_urls":["<url>"]},
     "spent_usd":<cost of this call>}

● Most model samples land on the same family as my plan. A blind-spot family that almost no
  sample used scored better on your evaluator, so I'll start from HERD's best implementation
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
  **On Linux there is no OS sandbox yet**; run HERD in a container or VM there.
- Error text that leaves the sandbox is redacted (known secret values and key-shaped
  strings), so a candidate that finds a secret cannot report it back through results or MCP.
- A result only counts if its line carries a one-time nonce that the evaluator reads from
  stdin before loading the candidate, so a candidate cannot just print a fake score.

Known limits: the candidate runs in the same Python process as the scoring loop, so code
written to attack the evaluator could inspect its stack frames, for example to see future
items or the nonce. Model samples are not prompted to do this, all generated code is kept
under `results/` for inspection, and `labels_audit.csv` supports spot checks.

## Traces

Set `HERD_TRACE_DIR=<dir>` to append every model and search call to `<dir>/calls.jsonl`
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
