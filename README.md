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

Tasks live in `tasks/<name>/` (`prompt.md` + `evaluator.py`). Evaluators run candidates in a
subprocess with a 60s timeout and print `{"valid": bool, "score": float, "detail": {...}}`.
Lower is better. `python tasks/<name>/evaluator.py --baselines` prints the reference scores.

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
  JSON line of stdout it must print `{"valid": bool, "score": number}`. Use absolute paths.
- `budget_usd` caps what one call may spend. The global cap in `config.yaml` still applies.
  Every response includes `spent_usd`. A call that would exceed the cap returns
  `{"error":"budget_exceeded",...}` instead of a partial result.
- `herd_check` verdicts:
  - `keep`: your family scored best of all tested families ("common and best, keep it").
  - `try_blind_spot`: a blind spot beat it; the response includes the family, its score, code path and sources.
  - `try_other_family`: a herd family beat it.
  - `inconclusive`: no valid implementation of your family.
  - `untested`: there was no evaluator, so only the map lookup ran.

Caution: `reality_check` and `herd_check` run LLM-written code through your evaluator.
The built-in evaluators isolate it in a subprocess with a timeout, but that is not a security
sandbox. Run HERD where executing generated code is acceptable.

### Example: Claude Code calls `herd_check` mid-task

*Illustrative transcript. The calls and response fields are the real interface; the
families and numbers below are made up, not experimental results.*

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
  ⎿ {"family":"Recency (LRU-family)","share":0.62,"rank":1,"is_blind_spot":false,
     "concentration_top3":0.91,"my_family_best_score":0.512,
     "tested":{"Recency (LRU-family)":0.512,"Frequency (LFU-family)":0.488,
               "Static/dynamic partitioning (SLRU, 2Q)":0.471},
     "verdict":"try_blind_spot",
     "blind_spot":{"family":"Static/dynamic partitioning (SLRU, 2Q)",
                   "mechanism":"new keys enter a probationary queue and only keys hit again are promoted",
                   "share":0.0,"best_score":0.471,
                   "code_path":"/path/to/herd/results/desc-3f2a9c1d0b7e/reality/9c1e4a7b2d/static_dynamic_partitioning_slru_2q-1.py",
                   "source_urls":["https://..."]},
     "spent_usd":3.41}

● 62% of model samples land on LRU, my plan included. A probationary-queue (2Q/SLRU) policy,
  which no sample used, scored 0.471 against 0.512 for the best LRU on your evaluator.
  I'll implement the 2Q-style policy instead, starting from HERD's best implementation, and
  check it with bench/eval.py.
````

## Tests

```sh
uv run pytest
```

The tests cover the evaluators and baselines, caching and the budget ledger. They also
run the whole pipeline and the MCP tools end to end, with OpenRouter and Exa replaced by
fakes.
