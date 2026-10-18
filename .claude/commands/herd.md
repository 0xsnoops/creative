---
description: Check whether you are stuck in a strategy loop and get one tested next step (HERD)
argument-hint: "[task, evaluator command, current solution and earlier attempts, or planned approach]"
---

Follow the `herd` skill (`.claude/skills/herd/SKILL.md`, or `skills/herd/SKILL.md` in the HERD
repo; read it first) for the user's situation below.

1. Call `herd_status` (zero cost). If HERD is not ready, stop and say what is missing. If the
   `herd_*` tools are absent, run `python -m herd onboard --client claude --target <this project> --json`
   from the HERD repo and follow its `next_steps`.
2. From the situation and the conversation, collect the task description, the evaluator
   (command or registered task name), the current solution and earlier attempts, or the planned
   approach. Ask the user for anything essential instead of guessing.
3. Scored attempts that stall: `herd_escape`. Only a planned approach: `herd_check`. Neither an
   evaluator nor scored attempts: say HERD does not apply.
4. Act on the verdict as the skill says (keep, repair, escape, escalate, inconclusive) and report
   evidence by class (MEASURED, SOURCE_BACKED, HYPOTHESIZED), the scope note and `spent_usd`.

User's situation: $ARGUMENTS
