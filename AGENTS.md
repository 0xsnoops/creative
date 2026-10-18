# Agent notes for this repo

This repo is DIVERGE: it tells a coding agent when its scored attempts are one
strategy in different code, and names one evidence-backed alternative.

- The skill is `skills/diverge/SKILL.md` (also at `.claude/skills/diverge`). Read it when several
  scored attempts have stalled, before retrying another variation or escalating to a human.
- MCP server: `python -m diverge serve` (stdio). Readiness at $0: `python -m diverge preflight --json --offline`.
- Offline demo at $0: `python -m diverge demo strategy_loop`.
- Tests: `uv run pytest -q tests/`. Never print or commit keys; they live in the gitignored `.env`.
