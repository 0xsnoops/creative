"""A task is either a registered folder under tasks/ (prompt.md + evaluator.py) or a
free-text description with no built-in evaluator."""
import hashlib
import re
import importlib.util
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


class Task:
    def __init__(self, name, prompt, evaluator=None):
        self.name, self.prompt, self.evaluator = name, prompt, evaluator
        self.entrypoint = evaluator.ENTRYPOINT if evaluator else None  # what code must define

    @classmethod
    def registered(cls, name):
        folder = TASKS_DIR / name
        spec = importlib.util.spec_from_file_location(f"{name}_evaluator", folder / "evaluator.py")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        return cls(name, (folder / "prompt.md").read_text(), evaluator)

    @classmethod
    def resolve(cls, text):
        """A registered task name, or else a free-text task description used verbatim as the prompt."""
        if not text or not text.strip():
            raise ValueError("task_description is empty")
        name = text.strip()
        # Only a short identifier can name a built-in task; anything else is free text (checking a
        # long description as a path raised "File name too long").
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) and (TASKS_DIR / name / "evaluator.py").exists():
            return cls.registered(name)
        digest = hashlib.sha256(text.strip().encode()).hexdigest()[:12]
        return cls(f"desc-{digest}", text.strip())

    def evaluate(self, code_path, split="dev"):
        return self.evaluator.evaluate(code_path, split=split)

    def baselines(self, split="dev"):
        return self.evaluator.baselines(split) if self.evaluator else {}
