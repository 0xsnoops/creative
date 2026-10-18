"""Behavioral fingerprints (model-review cluster C4: qwen, openai). Two candidates count as the
same behavior when they choose differently on at most THRESHOLD of the probe states. This
measures decisions, not labels; it does not prove global equivalence or novelty.

Only tasks whose evaluator defines fingerprint() support it (online_bin_packing). Probes are
single decisions on states taken from best-fit trajectories on separate probe seeds, so
long-horizon or stateful effects are not captured.

THRESHOLD was fixed from the calibration heuristics below, before any real candidate was
fingerprinted: rescaled and tie-change copies of a heuristic were at distance 0.000 from it,
and the five distinct heuristics were at least 0.070 apart."""
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

THRESHOLD = 0.03

# Calibration set: distinct mechanisms, plus copies that must land with their original.
CANONICAL = {
    "first_fit": "def priority(item, capacities):\n    return [-i for i in range(len(capacities))]\n",
    "best_fit": "def priority(item, capacities):\n    return [-(c - item) for c in capacities]\n",
    "worst_fit": ("def priority(item, capacities):\n    return [(c - item) if i < len(capacities) - 1 else -1e9"
                  " for i, c in enumerate(capacities)]\n"),
    "next_fit": ("def priority(item, capacities):\n    return [0.0] * (len(capacities) - 2) + [1.0, 0.5]"
                 " if len(capacities) >= 2 else [1.0]\n"),
    "funsearch_weibull": ("import numpy as np\ndef priority(item, capacities):\n    bins = np.asarray(capacities, dtype=float)\n"
                          "    m = max(bins)\n    s = (bins - m) ** 2 / item + bins ** 2 / item ** 2 + bins ** 2 / item ** 3\n"
                          "    s[bins > item] = -s[bins > item]\n    s[1:] -= s[:-1]\n    return s\n"),
}
COPIES = {  # copy -> the canonical heuristic it must match
    "best_fit_rescaled": ("best_fit", "def priority(item, capacities):\n    return [-1000.0 * (c - item) + 7 for c in capacities]\n"),
    "best_fit_ties_last": ("best_fit", "def priority(item, capacities):\n    return [-(c - item) + 1e-6 * i for i, c in enumerate(capacities)]\n"),
    "first_fit_rescaled": ("first_fit", "def priority(item, capacities):\n    return [-3.5 * i - 2 for i in range(len(capacities))]\n"),
}


def supported(task):
    return bool(task.evaluator) and hasattr(task.evaluator, "fingerprint")


def distance(a, b):
    return sum(x != y for x, y in zip(a, b)) / max(len(a), 1)


def fingerprints(task, paths):
    """{key: choices or None (invalid on probes)} for {key: path}, computed in the sandbox."""
    def one(item):
        key, path = item
        result = task.evaluator.fingerprint(path)
        return key, result["detail"]["choices"] if result["valid"] else None
    with ThreadPoolExecutor(8) as pool:
        return dict(pool.map(one, paths.items()))


def clusters(fps, threshold=THRESHOLD):
    """Single-linkage groups of keys whose fingerprints are within threshold."""
    keys = [k for k, v in fps.items() if v is not None]
    parent = {k: k for k in keys}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if distance(fps[a], fps[b]) <= threshold:
                parent[root(a)] = root(b)
    groups = {}
    for k in keys:
        groups.setdefault(root(k), []).append(k)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def validation_gate(task):
    """Copies must land with their originals; distinct heuristics must not merge."""
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for name, code in {**CANONICAL, **{k: v[1] for k, v in COPIES.items()}}.items():
            paths[name] = os.path.join(d, f"{name}.py")
            with open(paths[name], "w") as f:
                f.write(code)
        fps = fingerprints(task, paths)
    names = list(CANONICAL)
    separated = min(distance(fps[a], fps[b]) for i, a in enumerate(names) for b in names[i + 1:])
    copies = max(distance(fps[c], fps[orig]) for c, (orig, _) in COPIES.items())
    return {"passed": copies <= THRESHOLD < separated, "threshold": THRESHOLD,
            "max_copy_distance": copies, "min_distinct_distance": separated}


def label_agreement(groups, labels):
    """How behavior clusters line up with labeled families: clusters holding more than one
    label (the labels over-split behavior) and labels spread over several clusters (one label
    hides different behaviors)."""
    cluster_of = {k: i for i, g in enumerate(groups) for k in g}
    by_label = {}
    for k, label in labels.items():
        if k in cluster_of:
            by_label.setdefault(label, set()).add(cluster_of[k])
    return {"behavior_clusters": len(groups), "labeled_families": len(by_label),
            "clusters_with_several_labels": sum(len({labels[k] for k in g if k in labels}) > 1 for g in groups),
            "labels_split_across_clusters": sorted(l for l, cs in by_label.items() if len(cs) > 1)}
