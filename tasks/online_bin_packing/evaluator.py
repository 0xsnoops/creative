"""Online bin packing evaluator. Lower score is better.

    python evaluator.py CANDIDATE.py   -> prints {"valid", "score", "detail"} as JSON
    python evaluator.py --baselines    -> scores the reference first_fit and best_fit

Score = mean over 5 fixed instances of the % excess bins over the lower bound
ceil(sum(items) / CAPACITY). The candidate runs in a subprocess with a timeout.
"""
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CAPACITY = 100
N_ITEMS = 5000
SEEDS = [0, 1, 2, 3, 4]
TIMEOUT_S = 60
ENTRYPOINT = "priority"  # what candidate code must define
HERE = Path(__file__).resolve().parent
RESULT_TAG = "__HERD_RESULT__"


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


def score_candidate(path):
    per_instance = []
    for seed in SEEDS:
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


def _inner(path):
    start = time.time()
    score, per_instance = score_candidate(path)
    detail = {"instances": per_instance, "runtime_s": round(time.time() - start, 2)}
    print(RESULT_TAG + json.dumps({"valid": True, "score": score, "detail": detail}))


def evaluate(path, timeout=TIMEOUT_S):
    """Run the candidate file in a subprocess. Crashes and timeouts are invalid."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--inner", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"valid": False, "score": None, "detail": {"error": f"timeout after {timeout}s"}}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_TAG):
            return json.loads(line[len(RESULT_TAG):])
    error = (proc.stderr.strip().splitlines() or ["no result"])[-1]
    return {"valid": False, "score": None,
            "detail": {"error": error, "stderr_tail": proc.stderr[-2000:]}}


def baselines():
    return {p.stem: evaluate(p) for p in sorted((HERE / "baselines").glob("*.py"))}


if __name__ == "__main__":
    if sys.argv[1] == "--inner":
        _inner(sys.argv[2])
    elif sys.argv[1] == "--baselines":
        for name, result in baselines().items():
            print(f"{name}: {result['score']:.3f}% excess bins")
    else:
        print(json.dumps(evaluate(sys.argv[1])))
