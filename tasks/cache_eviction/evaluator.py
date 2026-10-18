"""Cache eviction evaluator. Lower score is better.

    python evaluator.py CANDIDATE.py [--split heldout]  -> prints {"valid", "score", "detail"} as JSON
    python evaluator.py --baselines [--split heldout]   -> scores the reference LRU and FIFO
    probe(path, cases)                                   -> misses and evictions on tiny traces

Score = mean miss ratio over 3 seeded synthetic traces x 2 cache sizes (1% and 10%
of the trace's unique keys). The policy only ever sees past requests. The candidate
runs isolated (tasks/_sandbox.py) with a timeout.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _sandbox import emit, evaluate_isolated, read_nonce, read_nonce_and_payload  # noqa: E402

N_REQUESTS = 50_000
UNIVERSE = 5_000        # keys with Zipf popularity
ZIPF_ALPHA = 0.9
SCAN_LENGTH = 1_000     # keys per sequential scan
SCAN_EVERY = 10_000     # Zipf requests between scans
ONE_HIT_FRACTION = 0.5  # share of requests to keys that are never requested again
CACHE_FRACTIONS = [0.01, 0.10]
TIMEOUT_S = 60
ENTRYPOINT = "Policy"  # what candidate code must define
HERE = Path(__file__).resolve().parent


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


TRACES = {"zipf": trace_zipf, "zipf_scans": trace_zipf_scans, "one_hit_wonders": trace_one_hit_wonders}
# "dev" traces are used for all scoring and selection. "heldout" traces (same generators, new
# seeds) are only used to report final numbers for candidates already selected on dev.
SPLITS = {"dev": {"zipf": 1, "zipf_scans": 2, "one_hit_wonders": 3},
          "heldout": {"zipf": 11, "zipf_scans": 12, "one_hit_wonders": 13}}


def make_traces(split="dev"):
    return {name: TRACES[name](seed) for name, seed in SPLITS[split].items()}


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


class _Recorder:
    """Wraps a policy for simulate() and logs its misses and evictions."""

    def __init__(self, policy, log):
        self.policy, self.log = policy, log

    def on_hit(self, key):
        self.policy.on_hit(key)

    def on_miss(self, key):
        self.log["misses"] += 1
        self.policy.on_miss(key)

    def evict(self):
        victim = self.policy.evict()
        self.log["evictions"].append(victim)
        return victim


def probe_candidate(path, cases):
    """For each case {"trace": [keys], "capacity": n}: run simulate() itself with a fresh Policy
    and return {"misses", "evictions": victims in order}."""
    outputs = []
    for case in cases:
        trace, capacity = [int(k) for k in case["trace"]], int(case["capacity"])
        policy_cls, log = load_class(path, ENTRYPOINT), {"misses": 0, "evictions": []}
        simulate(lambda size, cls=policy_cls, log=log: _Recorder(cls(size), log), trace, capacity)
        outputs.append({"misses": log["misses"], "evictions": [int(v) for v in log["evictions"]]})
    return outputs


def score_candidate(path, split="dev"):
    runs = []
    for name, trace in make_traces(split).items():
        unique = len(set(trace))
        for fraction in CACHE_FRACTIONS:
            size = max(1, round(fraction * unique))
            policy_cls = load_class(path, ENTRYPOINT)  # fresh module state per run
            runs.append({"trace": name, "cache_size": size, "unique_keys": unique,
                         "miss_ratio": simulate(policy_cls, trace, size)})
    return float(np.mean([r["miss_ratio"] for r in runs])), runs


def _inner(path, split):
    nonce = read_nonce()  # before any candidate code runs
    start = time.time()
    score, runs = score_candidate(path, split)
    detail = {"split": split, "runs": runs, "runtime_s": round(time.time() - start, 2)}
    emit(nonce, {"valid": True, "score": score, "detail": detail})


def _inner_probe(path):
    nonce, cases = read_nonce_and_payload()  # before any candidate code runs
    emit(nonce, {"valid": True, "score": None, "detail": {"outputs": probe_candidate(path, cases)}})


def probe(path, cases, timeout=TIMEOUT_S):
    """Run the candidate on tiny traces (see probe_candidate) in the same sandbox as evaluation.
    The cases reach the child on stdin after the nonce."""
    return evaluate_isolated(Path(__file__).resolve(), path, timeout, ["--probe"], payload=cases)


def evaluate(path, timeout=TIMEOUT_S, split="dev"):
    """Run the candidate in an isolated subprocess (see tasks/_sandbox.py). Crashes and
    timeouts are invalid."""
    return evaluate_isolated(Path(__file__).resolve(), path, timeout, ["--split", split])


def baselines(split="dev"):
    return {p.stem: evaluate(p, split=split) for p in sorted((HERE / "baselines").glob("*.py"))}


if __name__ == "__main__":
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "dev"
    if sys.argv[1] == "--inner" and "--probe" in sys.argv:
        _inner_probe(sys.argv[2])
    elif sys.argv[1] == "--inner":
        _inner(sys.argv[2], split)
    elif sys.argv[1] == "--baselines":
        for name, result in baselines(split).items():
            runs = ", ".join(f"{r['trace']}@{r['cache_size']}={r['miss_ratio']:.3f}"
                             for r in result["detail"]["runs"])
            print(f"{name}: mean miss ratio {result['score']:.4f}  ({runs})")
    else:
        print(json.dumps(evaluate(sys.argv[1], split=split)))
