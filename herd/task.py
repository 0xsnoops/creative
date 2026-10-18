"""A task is a folder under tasks/ holding prompt.md and evaluator.py."""
import importlib.util
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


class Task:
    def __init__(self, name):
        self.name = name
        self.dir = TASKS_DIR / name
        if not (self.dir / "evaluator.py").exists():
            raise SystemExit(f"unknown task {name!r}; have {sorted(p.name for p in TASKS_DIR.iterdir())}")
        self.prompt = (self.dir / "prompt.md").read_text()
        spec = importlib.util.spec_from_file_location(f"{name}_evaluator", self.dir / "evaluator.py")
        self.evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.evaluator)
        self.entrypoint = self.evaluator.ENTRYPOINT

    def evaluate(self, code_path):
        return self.evaluator.evaluate(code_path)

    def baselines(self):
        return self.evaluator.baselines()
