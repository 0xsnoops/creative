---
description: Find genuinely different alternatives when you keep landing on the same answer, and get one tested next step (DIVERGE)
argument-hint: "[task, evaluator command, current solution and earlier attempts, or planned approach]"
---

Follow the `diverge` skill (`.claude/skills/diverge/SKILL.md`, or `skills/diverge/SKILL.md` in the DIVERGE
repo; read it first) for the user's situation below.

1. Call `diverge_status` (zero cost). If DIVERGE is not ready, stop and say what is missing. If the
   `diverge_*` tools (e.g. `diverge_status`) are absent, run
   `uv --directory <diverge repo> run python -m diverge onboard --client claude --target <this project> --json`
   and follow its `next_steps`.
2. From the situation and the conversation, collect the task description, the evaluator
   (command or registered task name), the current solution and earlier attempts, or the planned
   approach. Ask the user for anything essential instead of guessing.
3. Scored attempts that stall: `diverge_escape`. A planned approach, or a request for a few different
   ways: `diverge_check` (without an evaluator it returns untested alternatives with strength, trap
   and how to test). Generic debugging or a bug with a known cause: say DIVERGE does not apply.
4. Act on the verdict as the skill says (keep, repair, escape, escalate, inconclusive; for
   `diverge_check`: keep, try_blind_spot, try_other_family, untested) and report evidence by class
   (MEASURED, SOURCE_BACKED, HYPOTHESIZED), the scope note and `spent_usd`.

User's situation: $ARGUMENTS
