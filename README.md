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

Models, sample counts and the hard dollar cap (`budget_usd`, default $15) are in
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
spots (sourced families with < 2% share), then implement the top 3 herd families and every
blind spot K=3 times with one shared prompt template and score them. Outputs go to
`results/<task>/`. `labels_audit.csv` holds 20 random labels for a human to spot-check.

With fewer than 20 samples per model (`reality_check.min_samples_per_model`) the CLI skips
the reality check: at small N every literature family nobody happened to use looks like a
blind spot. The taxonomy only keeps families that fit the task's rules (an offline algorithm
such as First Fit Decreasing cannot run on an online task); dropped families are listed in
`taxonomy.json` under `rejected` with the reason.

Tasks live in `tasks/<name>/` (`prompt.md` + `evaluator.py`). Evaluators print
`{"valid": bool, "score": float, "detail": {...}}`, lower is better, and run each candidate
isolated with a 60s timeout (see [Running model-written code](#running-model-written-code)).
`python tasks/<name>/evaluator.py --baselines` prints the reference scores.

## Generation operators and the benchmark

```sh
uv run python -m herd bench online_bin_packing --budget 0.5   # USD per operator per repeat
```

Five operators (`herd/operators/`) each turn a task and a budget into candidate solutions.
All of them use the same implementer model, implementation prompt template and evaluator.

| operator | what it does |
|---|---|
| `plain` | the task prompt as-is, temperature 1.0 |
| `hot` | the task prompt at the highest temperature that still yields valid code (calibrated per run; calibration samples count) |
| `verbalized` | Verbalized Sampling: 5 approaches with stated probabilities, sampled from the tails (each < 0.10), each one implemented |
| `blind_spot` | implements the families in `blind_spots.json` |
| `transplant` | Exa search for mechanisms from biology, logistics, markets, physics and networking; the labeler restates each as a structural rule in the problem's own terms; sources are kept |

`bench` runs every operator at the same budget for several seeded repeats, then labels all
valid candidates with one labeler against one taxonomy, so family counts are comparable. It
reports valid rate, distinct valid families, useful discoveries (families beating the best
reference baseline), useful discoveries per dollar and best score, each as mean and range
across repeats, plus `bench.png` and a verdict that only claims a win when the ranges do not
overlap. Results go to `results/<task>/bench/`.

Fairness rules: each operator's budget is measured from zero, separately from the global
cap, and the benchmark refuses to start if the global cap cannot fund every operator in
full. Planning costs (searches, the verbalized-sampling call, hot's calibration) count
against the operator. The herd map behind `blind_spot` and the labeling are reported but
not billed to any operator. A failure while one operator plans ends only that operator's
run and is noted in the report. Cache hits are billed at their original cost inside a
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
| `blind_spots` | `task_description`, `budget_usd=5` | sourced families under 2% share, with mechanism and source URLs |
| `reality_check` | `task_description`, `family`, `evaluator_cmd`, `k=3`, `lower_is_better`, `baseline_files`, `budget_usd=3` | best code, best and median score, invalid count and errors, baseline comparison, best tested herd family |
| `herd_check` | `task_description`, `my_approach`, `evaluator_cmd`, `k=3`, `lower_is_better`, `budget_usd=5` | the family of `my_approach`, its share and rank, and a `verdict` |

- `task_description` is the task as you would give it to an agent; it is sent to the
  sampler models verbatim. A registered task name (`online_bin_packing`, `cache_eviction`)
  uses that task's prompt and built-in evaluator.
- `evaluator_cmd` is run as `<evaluator_cmd> <path/to/candidate.py>` (no shell). On its last
  JSON line of stdout it must print `{"valid": bool, "score": number}`. It runs isolated, from
  a scratch working directory, so use absolute paths.
- `herd_map` and `blind_spots` return `"low_confidence": true` when the map was built from
  fewer than 20 samples per model.
- `budget_usd` caps what one call may spend. The global cap in `config.yaml` still applies.
  Every response includes `spent_usd`. A call that would exceed the cap returns
  `{"error":"budget_exceeded",...}` instead of a partial result.
- `herd_check` verdicts:
  - `keep`: your family scored best of all tested families ("common and best, keep it").
  - `try_blind_spot`: a blind spot beat it; the response includes the family, its score, code path and sources.
  - `try_other_family`: a herd family beat it.
  - `inconclusive`: no valid implementation of your family.
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
     "my_family_best_score":<score>,
     "tested":{"<family of my approach>":<score>,"<herd family>":<score>,"<blind spot>":<score>},
     "verdict":"try_blind_spot",
     "blind_spot":{"family":"<blind spot>","mechanism":"<one-line mechanism>",
                   "share":<share>,"best_score":<score>,
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
  throwaway scratch directory. **On Linux there is no OS sandbox yet**; run HERD in a
  container or VM there.
- A result only counts if its line carries a one-time nonce that the evaluator reads from
  stdin before loading the candidate, so a candidate cannot just print a fake score.

Known limits: the candidate runs in the same Python process as the scoring loop, so code
written to attack the evaluator could inspect its stack frames, for example to see future
items or the nonce. Model samples are not prompted to do this, all generated code is kept
under `results/` for inspection, and `labels_audit.csv` supports spot checks.

## Tests

```sh
uv run pytest
```

The tests cover the evaluators and baselines, the sandbox (fake result lines, secrets,
network and file writes), caching and the budget ledger. They also run the whole pipeline,
the operator benchmark and the MCP tools end to end, with OpenRouter and Exa replaced by
fakes.
