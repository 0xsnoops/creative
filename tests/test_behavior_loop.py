"""Behavior-based loop detection (diverge.trajectory detector "behavior"), offline: real sandboxed
fingerprints and dev scoring for online_bin_packing, no API calls. The LLM extractor, where used,
is a fake that returns random strategy captions."""
import random

import pytest

from diverge import trajectory
from diverge.diagnose import diagnose
from diverge.task import Task

BIN, CACHE = Task.resolve("online_bin_packing"), Task.resolve("cache_eviction")

BEST_FITS = [
    "def priority(item, capacities):\n    return [-(c - item) for c in capacities]\n",
    "import numpy as np\ndef priority(item, capacities):\n    caps = np.asarray(capacities, dtype=float)\n"
    "    return -(caps - item)\n",
    "def priority(item, capacities):\n    out = []\n    for c in capacities:\n        out.append(item - c)\n    return out\n",
]
# Nearly the same text, different decisions: best fit, worst fit, first fit.
LOOKALIKES = [
    "def priority(item, capacities):\n    return [-(c - item) for i, c in enumerate(capacities)]\n",
    "def priority(item, capacities):\n    return [(c - item) for i, c in enumerate(capacities)]\n",
    "def priority(item, capacities):\n    return [-(i) for i, c in enumerate(capacities)]\n",
]


def attempts(codes, scores, details=None):
    details = details or [None] * len(codes)
    return [{"id": f"a{i}", "code": c, "score": s, "dev_detail": d}
            for i, (c, s, d) in enumerate(zip(codes, scores, details))]


def detail(worst):
    """A bin-packing dev detail whose two worst instances are `worst`."""
    return {"instances": [{"seed": s, "excess_pct": 5.0 if s in worst else 1.0} for s in range(5)]}


def random_extract(seed):
    rng = random.Random(seed)
    families = ["best fit", "harmonic packing", "Weibull scoring", "reinforcement learning", "first fit",
                "simulated annealing", "genetic search", "lookahead"]

    def extract(code):
        return {"primary_signal": rng.choice(["item", "capacity_values", "n_open_bins", "history"]),
                "secondary_signal": rng.choice([None, "item", "history"]),
                "assumptions": [rng.choice(["bins are scarce", "items arrive sorted", "the future is uniform"])],
                "mechanism_family": rng.choice(families)}
    return extract


def test_identical_behavior_and_failures_with_flat_scores_is_a_behavior_loop():
    report = trajectory.analyze(BIN, attempts(BEST_FITS, [4.15, 4.14, 4.15]))
    b = report["behavior"]
    assert report["detector"] == "behavior" and report["loop_detected"]
    assert b["conditions"] == {"a": True, "b": True, "c": True}
    assert len(b["clusters"]) == 1 and len(set(b["cluster_of_recent"].values())) == 1
    assert b["failure_jaccard"]["min"] == 1.0 and b["max_probe_disagreement"] == 0.0
    assert b["improvement_last3"] == pytest.approx(0.01 / 4.15)
    assert report["strategy_similarity"] is not None and report["code_similarity"] is not None  # display only
    d = diagnose(BIN, report)
    assert d["action"] == "switch_strategy" and d["evidence"]["detector"] == "behavior"
    assert any(r.startswith("3/3 recent attempts in one behavior cluster (max probe disagreement 0.000")
               for r in d["reasons"])
    worst = b["failure_sets"]["a0"]
    assert any(r == f"failure sets identical (Jaccard 1.00): {', '.join(worst)}" for r in d["reasons"])
    assert len(worst) == 2 and all(w.startswith("seed:") for w in worst)
    assert any(r.startswith("no dev improvement over the last 3 attempts") for r in d["reasons"])


def test_lookalike_code_with_different_behavior_is_not_a_loop():
    report = trajectory.analyze(BIN, attempts(LOOKALIKES, [4.1, 4.1, 4.1]))
    b = report["behavior"]
    assert report["code_similarity"] >= 0.5  # only the return line differs
    assert not report["loop_detected"] and b["conditions"]["a"] is False and b["conditions"]["c"] is True
    assert len(b["clusters"]) == 3
    d = diagnose(BIN, report)
    assert d["action"] == "repair"
    assert any("do not share one behavior cluster" in r for r in d["reasons"])
    assert any(r.startswith("only 1/3 recent attempts share a behavior cluster") for r in d["reasons"])


def test_same_behavior_but_different_failure_sets_is_repair():
    details = [detail({0, 1}), detail({0, 1}), detail({3, 4})]
    report = trajectory.analyze(BIN, attempts(BEST_FITS, [4.1, 4.1, 4.1], details))
    b = report["behavior"]
    assert b["conditions"] == {"a": True, "b": False, "c": True} and not report["loop_detected"]
    assert b["failure_jaccard"]["min"] == 0.0
    d = diagnose(BIN, report)
    assert d["action"] == "repair" and any("their failure sets differ" in r for r in d["reasons"])
    assert any(r.startswith("failure sets differ (min Jaccard 0.00 < 0.8)") for r in d["reasons"])


def test_improving_scores_continue():
    report = trajectory.analyze(BIN, attempts(BEST_FITS, [5.0, 4.0, 3.0]))
    b = report["behavior"]
    assert b["conditions"] == {"a": True, "b": True, "c": False} and not report["loop_detected"]
    assert b["improvement_last3"] == pytest.approx(0.4)
    d = diagnose(BIN, report)
    assert d["action"] == "continue" and any("+40.0%" in r for r in d["reasons"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_llm_captions_never_change_the_behavior_verdict(seed):
    plain = trajectory.analyze(BIN, attempts(BEST_FITS, [4.15, 4.14, 4.15]))
    captioned = trajectory.analyze(BIN, attempts(BEST_FITS, [4.15, 4.14, 4.15]), extract=random_extract(seed))
    assert captioned["loop_detected"] == plain["loop_detected"] is True
    assert captioned["behavior"]["conditions"] == plain["behavior"]["conditions"]
    assert diagnose(BIN, captioned)["action"] == diagnose(BIN, plain)["action"] == "switch_strategy"
    # and random captions cannot conjure a loop out of behaviorally different attempts
    different = trajectory.analyze(BIN, attempts(LOOKALIKES, [4.1, 4.1, 4.1]),
                                   extract=lambda code: {"primary_signal": "item", "mechanism_family": "best fit",
                                                         "assumptions": ["bins are scarce"]})
    assert different["strategy_similarity"] >= trajectory.SAME_STRATEGY and not different["loop_detected"]


def test_without_probes_shared_failures_decide_and_captions_never_do():
    # behavior off: the failure detector, on measured per-case results (scored with the task's evaluator)
    off = trajectory.analyze(BIN, attempts(BEST_FITS, [4.15, 4.14, 4.15]), extract=random_extract(7),
                             behavior_loop=False)
    assert off["detector"] == "failures" and off["behavior"] is None and off["loop_detected"] is True
    assert off["failures"]["conditions"] == {"b": True, "c": True}
    assert diagnose(BIN, off)["action"] == "switch_strategy"
    # identical captions (strategy similarity 1.0) cannot make a loop out of different failures
    same_caption = lambda code: {"primary_signal": "item", "mechanism_family": "best fit", "assumptions": ["x"]}
    differ = trajectory.analyze(BIN, attempts(BEST_FITS, [4.1, 4.1, 4.1], [detail({0, 1}), detail({2, 3}),
                                                                            detail({3, 4})]),
                                extract=same_caption, behavior_loop=False)
    assert differ["strategy_similarity"] >= trajectory.SAME_STRATEGY
    assert differ["detector"] == "failures" and differ["loop_detected"] is False
    d = diagnose(BIN, differ)
    assert d["action"] == "repair" and any("different dev cases" in r for r in d["reasons"])


def test_no_per_case_results_means_no_detector_and_an_undetermined_diagnosis():
    free = Task.resolve("Rank search results for a query.")
    flat = [{"id": f"a{i}", "code": f"def rank(x):\n    return x  # {i}\n", "score": s, "dev_detail": {"rows": 3}}
            for i, s in enumerate([0.41, 0.41, 0.41])]
    report = trajectory.analyze(free, flat, extract=random_extract(1))
    assert report["detector"] == "none" and report["loop_detected"] is None
    d = diagnose(free, report)
    assert d["action"] == "undetermined" and any("cannot tell a strategy loop" in r for r in d["reasons"])
    rising = [{**a, "score": s} for a, s in zip(flat, [0.30, 0.35, 0.45])]
    assert diagnose(free, trajectory.analyze(free, rising, extract=random_extract(1),
                                         lower_is_better=False))["action"] == "continue"
    cases = lambda worst: {"cases": [{"id": f"q{j}", "loss": 1.0 if j in worst else 0.1} for j in range(6)]}
    looped = [{**a, "dev_detail": cases({1, 4})} for a in flat]
    report = trajectory.analyze(free, looped, extract=random_extract(1), lower_is_better=False)
    assert report["detector"] == "failures" and report["loop_detected"] is True
    assert report["failures"]["failure_sets"]["a2"] == ["q1", "q4"]


def test_two_attempts_can_be_a_behavior_loop():
    report = trajectory.analyze(BIN, attempts(BEST_FITS[:2], [4.15, 4.15]))
    assert report["behavior"]["conditions"] == {"a": True, "b": True, "c": True} and report["loop_detected"]
    assert diagnose(BIN, report)["action"] == "switch_strategy"


def test_epsilon_and_window():
    imp = trajectory.improvement_last3
    assert imp([4.0], True) is None
    assert imp([10.0, 9.95], True) == pytest.approx(0.005)            # 0.5% <= 1%: stalled
    assert imp([10.0, 12.0, 9.8, 9.95], True) == pytest.approx(0.02)  # best earlier 10.0 vs best recent 9.8
    assert imp([0.5, 0.6, 0.6], False) == pytest.approx(0.2)
    assert imp([1.0, 2.0, 3.0], True) < 0                             # getting worse counts as stalled
