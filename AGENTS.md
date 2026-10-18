# Agent notes for this repo

This repo is DIVERGE: creative-search infrastructure for AI agents. When an agent keeps landing
on the same answer, DIVERGE shows it what it (and most models) converge on and helps it find, test
and pick genuinely different alternatives, returning a verdict and a `next_action` it can act on
without a human. Results so far are preliminary (see "Measured so far" in `README.md`).

- The skill is `skills/diverge/SKILL.md` (also at `.claude/skills/diverge`). Read it when you are
  stuck (several scored attempts stalled), keep getting the same answer, need alternatives, are
  about to retry the same idea or escalate to a human, or are choosing a strategy before coding
  (`diverge_check`).
- MCP server: `python -m diverge serve` (stdio). Readiness at $0: `python -m diverge preflight --json --offline`.
- Offline demo at $0: `python -m diverge demo strategy_loop`.
- Tests: `uv run pytest -q tests/`. Never print or commit keys; they live in the gitignored `.env`.
