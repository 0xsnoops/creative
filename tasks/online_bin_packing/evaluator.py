"""Online bin packing evaluator. Lower score is better.

    python evaluator.py CANDIDATE.py [--split heldout]  -> prints {"valid", "score", "detail"} as JSON
    python evaluator.py --baselines [--split heldout]   -> scores the reference first_fit and best_fit
    probe(path, cases)                                   -> per-case choices on tiny item sequences
    fingerprint_perturbed(path, input_name)              -> probe choices before/after perturbing one input

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
from _sandbox import emit, evaluate_isolated, read_nonce, read_nonce_and_payload  # noqa: E402

CAPACITY = 100
N_ITEMS = 5000
# "dev" instances are used for all scoring and selection. "heldout" instances are only used
# to report final numbers for candidates already selected on dev. Both fixed before any results.
SPLITS = {"dev": [0, 1, 2, 3, 4], "heldout": [1000, 1001, 1002, 1003, 1004]}
# Behavioral fingerprints use their own seeds (never dev or held-out): probe states are taken
# from best-fit packing trajectories on these sequences.
PROBE_SEEDS = [5000, 5001, 5002, 5003]
PROBES_PER_SEED = 50
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


def make_probes():
    """~200 decision states (item, feasible capacities + new bin) from best-fit trajectories,
    sampled evenly so early (few bins) and late (many open, near-full bins) states both appear."""
    probes = []
    for seed in PROBE_SEEDS:
        items = make_items(seed)
        remaining, n_bins = np.zeros(len(items), dtype=int), 0
        every = len(items) // PROBES_PER_SEED
        for step, item in enumerate(items):
            item = int(item)
            fits = np.flatnonzero(remaining[:n_bins] >= item)
            if step % every == every // 2:
                probes.append((item, remaining[fits].tolist() + [CAPACITY]))
            if len(fits):
                best = fits[np.argmin(remaining[fits] - item)]
                remaining[best] -= item
            else:
                remaining[n_bins] = CAPACITY - item
                n_bins += 1
    return probes


def fingerprint_candidate(path):
    """The remaining capacity of the bin chosen on each probe (100 = a new bin). Using the
    capacity, not the bin's position, makes ties between equal bins invisible: those choices
    lead to the same future state."""
    priority = load_function(path, ENTRYPOINT)
    choices = []
    for item, capacities in make_probes():
        scores = np.asarray(priority(item, list(capacities)), dtype=float)
        if len(scores) != len(capacities) or not np.all(np.isfinite(scores)):
            raise ValueError("priority returned invalid scores on a probe")
        choices.append(capacities[int(np.argmax(scores))])
    return choices

# Admission perturbations (diverge/admission.py): each rewrites the probe states for one input of
# priority(), and the child runs the candidate on the original and the perturbed states, each
# with a fresh module. Chosen options are compared by the ORIGINAL capacity they stand for.
PERTURB_SEED = 7000
WARMUP_ITEMS = 50
WARMUP_SEED = 6000  # warm-up sequences for "history"; not a dev, held-out or probe seed
PERTURBABLE = ("item", "capacity_values", "bin_order", "n_open_bins", "history")


def perturb_probe(name, item, capacities, rng):
    """(item', capacities', back) for one probe, or None when the perturbation does not apply.
    back[i] is the original capacity offered at perturbed position i. The new bin stays last."""
    bins, new = list(capacities[:-1]), capacities[-1]
    if name == "item":  # +-1, only where every offered bin still fits it
        steps = [s for s in (1, -1) if 1 <= item + s <= min(capacities)]
        if not steps:
            return None
        return item + steps[int(rng.integers(len(steps)))], list(capacities), list(capacities)
    if name == "capacity_values":  # +-1 on each open bin, kept in item..CAPACITY-1
        noisy = []
        for c in bins:
            d = 1 if rng.random() < 0.5 else -1
            noisy.append(next((c + s for s in (d, -d) if item <= c + s <= CAPACITY - 1), c))
        if noisy == bins:
            return None
        return item, noisy + [new], bins + [new]
    if name == "bin_order":  # reverse the open bins
        if bins[::-1] == bins:  # includes fewer than two distinct capacities
            return None
        caps = bins[::-1] + [new]
        return item, caps, caps
    if name == "n_open_bins":  # one extra copy of an open bin: same options, longer list
        if not bins:
            return None
        caps = bins + [bins[int(rng.integers(len(bins)))], new]
        return item, caps, caps
    raise ValueError(f"no perturbation for {name!r}")


def warmup_items(block):
    rng = np.random.default_rng(WARMUP_SEED + block)
    return np.clip(np.round(rng.weibull(3.0, WARMUP_ITEMS) * 45), 1, CAPACITY).astype(int)


def _choose(priority, item, capacities):
    scores = np.asarray(priority(item, list(capacities)), dtype=float)
    if len(scores) != len(capacities) or not np.all(np.isfinite(scores)):
        raise ValueError("priority returned invalid scores on a probe")
    return int(np.argmax(scores))


def perturbed_fingerprint_candidate(path, name):
    """Fingerprints on the original probes ("base", and again as "repeat" to expose
    nondeterminism) and on probes with one input perturbed ("perturbed"). "applicable" marks
    the probes the perturbation actually changed. For "history" the states are unchanged and
    the candidate instead packs WARMUP_ITEMS items before each probe seed's block."""
    if name not in PERTURBABLE:
        raise ValueError(f"no perturbation for {name!r}")
    probes = make_probes()
    original = [(item, caps, caps) for item, caps in probes]
    if name == "history":
        perturbed, applicable = original, [True] * len(probes)
    else:
        rng, perturbed, applicable = np.random.default_rng(PERTURB_SEED), [], []
        for item, caps in probes:
            p = perturb_probe(name, item, caps, rng)
            applicable.append(p is not None)
            perturbed.append(p or (item, caps, caps))

    def run(states, warmup=False):
        priority, out = load_function(path, ENTRYPOINT), []  # fresh module state per run
        for k, (item, caps, back) in enumerate(states):
            if warmup and k % PROBES_PER_SEED == 0:
                pack(priority, warmup_items(k // PROBES_PER_SEED))
            out.append(int(back[_choose(priority, item, caps)]))
        return out
    return {"input": name, "base": run(original), "perturbed": run(perturbed, warmup=name == "history"),
            "repeat": run(original), "applicable": applicable}


def probe_candidate(path, cases):
    """For each case {"items": [sizes]}: run pack() itself with a fresh module and record each
    choice. "chosen" is the free space of the chosen bin before placing (CAPACITY = a new bin),
    so picks between equal bins do not count as different; "bins" is the matching bin index
    (lowest among equal bins) and "n_bins" what pack() returned."""
    outputs = []
    for case in cases:
        items = [int(x) for x in case["items"]]
        if not all(1 <= x <= CAPACITY for x in items):
            raise ValueError("probe item sizes must be in 1..CAPACITY")
        priority, chosen = load_function(path, ENTRYPOINT), []

        def recording(item, capacities, priority=priority, chosen=chosen):
            before = list(capacities)  # the candidate may mutate its argument
            scores = priority(item, capacities)
            chosen.append(int(before[int(np.argmax(np.asarray(scores, dtype=float)))]))
            return scores
        n_bins = pack(recording, np.asarray(items, dtype=int))
        remaining, bins = [], []
        for item, free in zip(items, chosen):
            i = remaining.index(free) if free < CAPACITY else len(remaining)
            if i == len(remaining):
                remaining.append(CAPACITY)
            remaining[i] -= item
            bins.append(i)
        outputs.append({"chosen": chosen, "bins": bins, "n_bins": int(n_bins)})
    return outputs


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


def _inner_fingerprint(path):
    nonce = read_nonce()  # before any candidate code runs
    emit(nonce, {"valid": True, "score": None, "detail": {"choices": fingerprint_candidate(path)}})


def _inner_perturbed(path):
    nonce, payload = read_nonce_and_payload()  # before any candidate code runs
    emit(nonce, {"valid": True, "score": None, "detail": perturbed_fingerprint_candidate(path, payload["input"])})


def _inner_probe(path):
    nonce, cases = read_nonce_and_payload()  # before any candidate code runs
    emit(nonce, {"valid": True, "score": None, "detail": {"outputs": probe_candidate(path, cases)}})


def probe(path, cases, timeout=TIMEOUT_S):
    """Run the candidate on tiny cases (see probe_candidate) in the same sandbox as evaluation.
    The cases reach the child on stdin after the nonce."""
    return evaluate_isolated(Path(__file__).resolve(), path, timeout, ["--probe"], payload=cases)


def fingerprint(path, timeout=TIMEOUT_S):
    """Behavioral fingerprint of a candidate, computed in the same sandbox as evaluation."""
    return evaluate_isolated(Path(__file__).resolve(), path, timeout, ["--fingerprint"])


def fingerprint_perturbed(path, input_name, timeout=TIMEOUT_S):
    """Fingerprints before and after perturbing one priority() input (see
    perturbed_fingerprint_candidate), in the same sandbox as evaluation. The input name reaches
    the child on stdin after the nonce, so it is not visible in the child's argv."""
    return evaluate_isolated(Path(__file__).resolve(), path, timeout, ["--fingerprint", "--perturb"],
                             payload={"input": input_name})


def evaluate(path, timeout=TIMEOUT_S, split="dev"):
    """Run the candidate in an isolated subprocess (see tasks/_sandbox.py). Crashes and
    timeouts are invalid."""
    return evaluate_isolated(Path(__file__).resolve(), path, timeout, ["--split", split])


def baselines(split="dev"):
    return {p.stem: evaluate(p, split=split) for p in sorted((HERE / "baselines").glob("*.py"))}


if __name__ == "__main__":
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "dev"
    if sys.argv[1] == "--inner" and "--perturb" in sys.argv:
        _inner_perturbed(sys.argv[2])
    elif sys.argv[1] == "--inner" and "--probe" in sys.argv:
        _inner_probe(sys.argv[2])
    elif sys.argv[1] == "--inner" and "--fingerprint" in sys.argv:
        _inner_fingerprint(sys.argv[2])
    elif sys.argv[1] == "--inner":
        _inner(sys.argv[2], split)
    elif sys.argv[1] == "--baselines":
        for name, result in baselines(split).items():
            print(f"{name}: {result['score']:.3f}% excess bins")
    else:
        print(json.dumps(evaluate(sys.argv[1], split=split)))
