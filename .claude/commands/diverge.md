---
description: Check whether you are stuck in a strategy loop and get one tested next step (DIVERGE)
argument-hint: "[task, evaluator command, current solution and earlier attempts, or planned approach]"
---

Follow the `herd` skill (`.claude/skills/diverge/SKILL.md`, or `skills/diverge/SKILL.md` in the DIVERGE
repo; read it first) for the user's situation below.

1. Call `diverge_status` (zero cost). If DIVERGE is not ready, stop and say what is missing. If the
   `herd_*` tools are absent, run `python -m diverge onboard --client claude --target <this project> --json`
   from the DIVERGE repo and follow its `next_steps`.
2. From the situation and the conversation, collect the task description, the evaluator
   (command or registered task name), the current solution and earlier attempts, or the planned
   approach. Ask the user for anything essential instead of guessing.
3. Scored attempts that stall: `diverge_escape`. Only a planned approach: `diverge_check`. Neither an
   evaluator nor scored attempts: say DIVERGE does not apply.
4. Act on the verdict as the skill says (keep, repair, escape, escalate, inconclusive) and report
   evidence by class (MEASURED, SOURCE_BACKED, HYPOTHESIZED), the scope note and `spent_usd`.

User's situation: $ARGUMENTS
