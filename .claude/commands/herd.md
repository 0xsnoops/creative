---
description: Check whether you are stuck in a strategy loop and get one tested next step (HERD)
argument-hint: "[task, evaluator command, current solution and earlier attempts, or planned approach]"
---

Follow the HERD skill in `skills/herd/SKILL.md` (read it first) for the user's situation below.

Run `uv run python -m herd preflight` before anything else and stop if it fails. Then, from the
situation and the conversation so far, collect the task description, the evaluator (command or
registered task name), the current solution and 1-3 earlier attempts if they exist, or the
planned approach if nothing has been tried yet. Call `herd_escape` when there is a trajectory
and an evaluator, `herd_check` when there is only a planned approach, and say this does not
apply when there is neither an evaluator nor a trajectory. Ask the user for anything essential
that is missing instead of guessing it.

Act on the verdict as the skill says, and report evidence by class (MEASURED, SOURCE_BACKED,
HYPOTHESIZED), the scope note, and the amount spent.

User's situation: $ARGUMENTS
