"""Online bin packing evaluator. Lower score is better.

    python evaluator.py CANDIDATE.py [--split heldout]  -> prints {"valid", "score", "detail"} as JSON
    python evaluator.py --baselines [--split heldout]   -> scores the reference first_fit and best_fit

Score = mean over 5 fixed instances of the % excess bins over the lower bound
ceil(sum(items) / CAPACITY). The candidate runs isolated (tasks/_sandbox.py) with a
timeout.
"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _sandbox import emit, evaluate_isolated, read_nonce  # noqa: E402

CAPACITY = 100
N_ITEMS = 5000
# "dev" instances are used for all scoring and selection. "heldout" instances are only used
# to report final numbers for candidates already selected on dev. Both fixed before any results.
SPLITS = {"dev": [0, 1, 2, 3, 4], "heldout": [1000, 1001, 1002, 1003, 1004]}
TIMEOUT_S = 60
ENTRYPOINT = "priority"  # what candidate code must define
HERE = Path(__file__).resolve().parent


def make_items(seed):
    rng = np.random.default_rng(seed)
    sizes = np.round(rng.weibull(3.0, N_ITEMS) * 45)
    return np.clip(sizes, 1, CAPACITY).astype(int)


def pack(priority, items):
    """Place items one at a time; return the number of bins used."""
    remaining = np.zeros(len(items), dtype=int)  # remaining capacity of each opened bin
    n_bins = 0
    for item in items:
        item = int(item)
        fits = np.flatnonzero(remaining[:n_bins] >= item)
        capacities = remaining[fits].tolist() + [CAPACITY]  # last entry = a new empty bin
        scores = priority(item, list(capacities))
        if len(scores) != len(capacities):
            raise ValueError(f"priority returned {len(scores)} scores for {len(capacities)} bins")
        scores = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(scores)):
            raise ValueError("priority returned a non-finite score")
        choice = int(np.argmax(scores))
        if choice == len(fits):
            remaining[n_bins] = CAPACITY - item
            n_bins += 1
        else:
            remaining[fits[choice]] -= item
    return n_bins


def score_candidate(path, split="dev"):
    per_instance = []
    for seed in SPLITS[split]:
        items = make_items(seed)
        lower_bound = math.ceil(items.sum() / CAPACITY)
        bins = pack(load_function(path, ENTRYPOINT), items)  # fresh module state per instance
        per_instance.append({"seed": seed, "bins": bins, "lower_bound": lower_bound,
                             "excess_pct": 100.0 * (bins - lower_bound) / lower_bound})
    score = float(np.mean([r["excess_pct"] for r in per_instance]))
    return score, per_instance


def load_function(path, name):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def _inner(path, split):
    nonce = read_nonce()  # before any candidate code runs
    start = time.time()
    score, per_instance = score_candidate(path, split)
    detail = {"split": split, "instances": per_instance, "runtime_s": round(time.time() - start, 2)}
    emit(nonce, {"valid": True, "score": score, "detail": detail})


def evaluate(path, timeout=TIMEOUT_S, split="dev"):
    """Run the candidate in an isolated subprocess (see tasks/_sandbox.py). Crashes and
    timeouts are invalid."""
    return evaluate_isolated(Path(__file__).resolve(), path, timeout, ["--split", split])


def baselines(split="dev"):
    return {p.stem: evaluate(p, split=split) for p in sorted((HERE / "baselines").glob("*.py"))}


if __name__ == "__main__":
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "dev"
    if sys.argv[1] == "--inner":
        _inner(sys.argv[2], split)
    elif sys.argv[1] == "--baselines":
        for name, result in baselines(split).items():
            print(f"{name}: {result['score']:.3f}% excess bins")
    else:
        print(json.dumps(evaluate(sys.argv[1], split=split)))
