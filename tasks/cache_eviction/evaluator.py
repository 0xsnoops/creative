"""Cache eviction evaluator. Lower score is better.

    python evaluator.py CANDIDATE.py   -> prints {"valid", "score", "detail"} as JSON
    python evaluator.py --baselines    -> scores the reference LRU and FIFO

Score = mean miss ratio over 3 seeded synthetic traces x 2 cache sizes (1% and 10%
of the trace's unique keys). The policy only ever sees past requests. The candidate
runs in a subprocess with a timeout.
"""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

N_REQUESTS = 50_000
UNIVERSE = 5_000        # keys with Zipf popularity
ZIPF_ALPHA = 0.9
SCAN_LENGTH = 1_000     # keys per sequential scan
SCAN_EVERY = 10_000     # Zipf requests between scans
ONE_HIT_FRACTION = 0.5  # share of requests to keys that are never requested again
CACHE_FRACTIONS = [0.01, 0.10]
TIMEOUT_S = 60
HERE = Path(__file__).resolve().parent
RESULT_TAG = "__HERD_RESULT__"


def zipf_keys(rng, n):
    """n requests over UNIVERSE keys; key ids are shuffled so id order says nothing about popularity."""
    weights = 1.0 / np.arange(1, UNIVERSE + 1) ** ZIPF_ALPHA
    ranks = rng.choice(UNIVERSE, size=n, p=weights / weights.sum())
    return rng.permutation(UNIVERSE)[ranks]


def trace_zipf(seed):
    return zipf_keys(np.random.default_rng(seed), N_REQUESTS).tolist()


def trace_zipf_scans(seed):
    """Zipf traffic interrupted every SCAN_EVERY requests by a sequential scan of the same key range."""
    base = zipf_keys(np.random.default_rng(seed), N_REQUESTS).tolist()
    scan = list(range(UNIVERSE, UNIVERSE + SCAN_LENGTH))
    trace = []
    for start in range(0, N_REQUESTS, SCAN_EVERY):
        trace += base[start:start + SCAN_EVERY] + scan
    return trace


def trace_one_hit_wonders(seed):
    """Zipf traffic where ONE_HIT_FRACTION of requests go to fresh keys seen exactly once."""
    rng = np.random.default_rng(seed)
    one_hit = rng.random(N_REQUESTS) < ONE_HIT_FRACTION
    trace = zipf_keys(rng, N_REQUESTS)
    fresh_ids = UNIVERSE + rng.permutation(10 * N_REQUESTS)[:one_hit.sum()]
    trace[one_hit] = fresh_ids
    return trace.tolist()


TRACES = {"zipf": (trace_zipf, 1), "zipf_scans": (trace_zipf_scans, 2),
          "one_hit_wonders": (trace_one_hit_wonders, 3)}


def make_traces():
    return {name: make(seed) for name, (make, seed) in TRACES.items()}


def simulate(policy_cls, trace, size):
    """Return the miss ratio of policy_cls on trace with a cache holding `size` keys."""
    policy = policy_cls(size)
    cache, misses = set(), 0
    for key in trace:
        if key in cache:
            policy.on_hit(key)
            continue
        misses += 1
        if len(cache) >= size:
            victim = policy.evict()
            if victim not in cache:
                raise ValueError(f"evict() returned {victim!r}, which is not in the cache")
            cache.remove(victim)
        cache.add(key)
        policy.on_miss(key)
    return misses / len(trace)


def load_class(path, name):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def score_candidate(path):
    runs = []
    for name, trace in make_traces().items():
        unique = len(set(trace))
        for fraction in CACHE_FRACTIONS:
            size = max(1, round(fraction * unique))
            policy_cls = load_class(path, "Policy")  # fresh module state per run
            runs.append({"trace": name, "cache_size": size, "unique_keys": unique,
                         "miss_ratio": simulate(policy_cls, trace, size)})
    return float(np.mean([r["miss_ratio"] for r in runs])), runs


def _inner(path):
    start = time.time()
    score, runs = score_candidate(path)
    detail = {"runs": runs, "runtime_s": round(time.time() - start, 2)}
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
            runs = ", ".join(f"{r['trace']}@{r['cache_size']}={r['miss_ratio']:.3f}"
                             for r in result["detail"]["runs"])
            print(f"{name}: mean miss ratio {result['score']:.4f}  ({runs})")
    else:
        print(json.dumps(evaluate(sys.argv[1])))
