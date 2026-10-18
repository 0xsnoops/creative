# DIVERGE and the hackathon tracks

YC "Make Something Agents Want". DIVERGE is creative-search infrastructure for AI agents: when an
agent keeps landing on the same answer, DIVERGE shows it what it (and most models) converge on and
helps it find, test and pick genuinely different alternatives. It is built for agents first (MCP
tools, self-onboarding, a verdict and `next_action` an agent can act on without a human). This maps
each track's criteria to what is in the repo, and separates what has been demonstrated from what is
only supported by the code. All results are
preliminary; `escape` has not been shown to beat a fresh start (see `docs/RESULTS.md`).

## Armature: build something agents use

Criteria: coding agents choose and use it on their own, understand its value, onboard
themselves, with no human in the loop.

| criterion | what satisfies it in the repo |
|---|---|
| Agents choose it on their own | The MCP server instructions and tool descriptions (`diverge/server.py`, `INSTRUCTIONS`, `diverge_escape`, `diverge_status`) say when to call DIVERGE (stalled scored attempts, about to retry a variation, about to escalate) and when not to (first attempt, build errors, generic debugging). The skill `skills/diverge/SKILL.md` has the same trigger language in its description and says "Do not wait for the user to ask you to use DIVERGE". |
| Agents understand the value | `diverge_escape` returns one branchable word (`VERDICT: keep\|repair\|escape\|escalate\|inconclusive`) and a `next_action` sentence per verdict (`diverge/contract.py`, `NEXT_ACTION`), plus `next_experiment` with success and failure thresholds and evidence labeled `MEASURED` / `SOURCE_BACKED` / `HYPOTHESIZED`. The same object is returned as MCP structuredContent. |
| Agents onboard themselves | `python -m diverge onboard --client claude --target DIR [--json]` (`diverge/onboard.py`) installs the skill, the `/diverge` command and a `.mcp.json` entry into DIR, idempotently, with no secrets in the config, and returns `next_steps`. `diverge_status` (free, no network) and `python -m diverge preflight --json [--offline]` report readiness: key status (never values), remaining budget, cached tasks, what is missing. |
| No human in the loop | `can_continue_autonomously` is true for every verdict except `escalate`. Errors are structured (`error`, `detail`, `missing`, `next_action`, `offline_demo_available`), so an agent can recover or fall back to `python -m diverge demo strategy_loop` ($0, offline). The only point that needs a human is a missing API key: DIVERGE says so and never asks the agent to invent one. |

**Demonstrated** (run locally, $0): onboarding into a fresh directory (creates
`.claude/skills/diverge/SKILL.md`, `.claude/commands/diverge.md`, `.mcp.json`), offline preflight,
the offline demo, and the tests in `tests/test_onboard.py`, `tests/test_agent_ux.py`,
`tests/test_contract.py`.

**Autonomous selection:** `scripts/autonomous_demo.py` runs a headless `claude -p` agent in a
freshly onboarded project with a prompt that never mentions DIVERGE, and records whether and when
it called a DIVERGE tool. For what happened, see `docs/AUTONOMOUS_AGENT_DEMO.md`. One run is an
anecdote, not a rate.

**Only supported, not demonstrated:** other clients than Claude Code (`onboard` supports
`--client claude` only); that following a verdict improves the outcome (the preliminary rescue test,
n = 4 each. In round 1 no arm beat a best-fit plateau. In round 2, fresh-context escape pipelines did, but the
random-lever control rescued the most (3/4) and default escape v2 the fewest (1/4); see docs/RESULTS.md).

## Manufact: best MCP server for agents

| criterion | what satisfies it in the repo |
|---|---|
| A real MCP server | `python -m diverge serve` (stdio) and `diverge_mcp_http.py` (streamable HTTP at `/mcp`, an ASGI `app` for `uvicorn 'diverge_mcp_http:app'`), both serving the same `diverge.server.mcp` object, so the tool list is identical. |
| Designed for agents | Server `INSTRUCTIONS` that say when to use which tool; a free readiness tool (`diverge_status`); one primary tool (`diverge_escape`) with a verdict, `next_action` and structuredContent; structured errors; progress notifications on long calls; per-call `budget_usd` caps plus a global cap. |
| Safe to host | Bearer token (`DIVERGE_HTTP_TOKEN`, with `HERD_HTTP_TOKEN` as a fallback; constant-time compare): missing or wrong token gives 401, unset token gives 503 (fails closed). Candidate code runs in `tasks/_sandbox.py` (macOS sandbox, no secrets in the child environment, redacted errors). |

**Demonstrated:** local HTTP. `tests/test_manufact.py` starts the app on loopback and checks
that the HTTP tool list equals the stdio one and that 401/503 are returned; `docs/MANUFACT.md`
records a local `diverge_map` call over HTTP from cache at $0.

**Not done:** DIVERGE is not deployed to Manufact Cloud and we have no Manufact credentials. The
proxy behaviour, `uvicorn` on PATH after `uv sync --frozen`, gateway timeouts for long calls and
the cold cache on a fresh deploy are untested. Manual deployment steps are in `docs/MANUFACT.md`.
There is no OS sandbox on Linux yet, which matters for a Linux host.

## 20-second demo

```sh
mkdir -p /tmp/x && uv run python -m diverge onboard --client claude --target /tmp/x --json   # agent installs DIVERGE itself
uv run python -m diverge demo strategy_loop    # offline replay: 4 stalled attempts -> VERDICT + next_action + labeled evidence
```

Say: "Agents keep landing on the same answer. Here one onboards itself with one command; when its
scored attempts stall, DIVERGE shows it that the attempts are one idea in different code, points to
what most models overlook, and gives it one alternative to test with pass/fail thresholds, every
claim labeled measured, sourced or hypothesized. No human in the loop."

## 60-second demo

```sh
mkdir -p /tmp/x && uv run python -m diverge onboard --client claude --target /tmp/x --json
ls -a /tmp/x /tmp/x/.claude/skills/diverge                # skill, /diverge command, .mcp.json; no secrets
uv run python -m diverge preflight --json --offline       # same object diverge_status returns: ready, missing, cached tasks
uv run python -m diverge demo strategy_loop               # recorded run, $0: diagnosis, consensus evidence, the contract
uv run python scripts/autonomous_demo.py --run N       # PAID and slow: run it before the demo, not live
python3 -c "import json; print(json.dumps(json.load(open('docs/demo/autonomous-run-N.summary.json'))['moment'], indent=2))"
```

Framing (one line before starting): "This is infrastructure for agents that keep converging on
the same answer: it helps them find, test and pick an alternative, and an agent drives all of it."

1. Onboarding (10 s): show the three files written and the `next_steps` in the JSON.
2. Readiness (5 s): `ready`, `missing`, `recommended_primary_tool: diverge_escape`; key status
   only, never values.
3. Offline demo (20 s): the four stalled attempts, strategy vs code similarity, the overlooked
   input `n_open_bins` (5 of 95 consensus-map samples: the alternative most models miss), then section 6, the contract: `VERDICT: repair`
   and evidence by class. Point out that DIVERGE said repair, not escape, and that the lever it
   would have tried lost to fresh_start when implemented as an ablation (`README.md`, Measured
   so far).
4. Autonomous run (25 s): the prompt (no mention of DIVERGE), then the transcript moment from
   `docs/demo/autonomous-run-N.summary.json`: `moment.text_before` (the agent's reasoning just
   before), `moment.tool` and `moment.input`, and `tools_before`. If the agent did not select
   DIVERGE, say so; the outcome is written up in `docs/AUTONOMOUS_AGENT_DEMO.md`.

Close: "Preliminary: steering toward overlooked ideas gave more distinct working approaches per
dollar but not a better best program; it has not been shown to beat a fresh start, and in a 4-state
rescue test it tied simple baselines."

## Claims we do not make

- We do not claim that DIVERGE beats or outperforms fresh_start, continuing, another model, or any
  other agent or program-search system: none of these comparisons has been shown.
- We do not claim that agents pick DIVERGE reliably: the autonomous demo is a small number of runs.
- We do not claim that the Manufact deployment works: it is verified locally only.
