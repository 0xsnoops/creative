"""python -m herd run <task> [--n N]   |   python -m herd spend"""
import argparse
import os
from pathlib import Path

import yaml

from .cache import JsonlCache
from .llm import LLM, Ledger
from .sample import run_sampling, summarize
from .task import Task

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """Read KEY=VALUE lines from .env without overriding the real environment."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def spent_so_far(cache_dir):
    return sum(JsonlCache(Path(cache_dir) / f).total_cost() for f in ("llm.jsonl", "exa.jsonl"))


def main():
    parser = argparse.ArgumentParser(prog="herd")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the blind-spot experiment on a task")
    run.add_argument("task")
    run.add_argument("--n", type=int, help="samples per model (overrides config)")
    sub.add_parser("spend", help="show estimated spend so far")
    args = parser.parse_args()

    load_env()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cache_dir = ROOT / cfg["paths"]["cache"]
    ledger = Ledger(cfg["budget_usd"], spent_so_far(cache_dir))

    if args.command == "spend":
        print(f"estimated spend: ${ledger.spent:.4f} of ${ledger.cap:.2f}")
        return

    task = Task(args.task)
    out_dir = ROOT / cfg["paths"]["results"] / task.name
    llm = LLM(cache_dir, ledger)
    n = args.n or cfg["sampling"]["n_per_model"]

    print(f"[sample] {task.name}: {n} per model x {len(cfg['models']['samplers'])} models")
    records = run_sampling(task, llm, cfg, out_dir, n)
    print(summarize(records))
    print(f"spend this run ${ledger.spent_this_run:.4f}; total ${ledger.spent:.4f} of ${ledger.cap:.2f}")


if __name__ == "__main__":
    main()
