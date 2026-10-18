"""Strategy fingerprints and trajectory loop detection, offline: static analysis on hand-written
code, a fake extractor, synthetic dev details, and a faked OpenRouter for llm_extractor."""
import json

import pytest

import herd.pipeline as pipeline
from herd import strategy, trajectory
from herd.llm import LLM
from herd.task import Task
from tests.test_pipeline_offline import MODELS, fake_world  # noqa: F401  (fixture)

BIN, CACHE = Task.resolve("online_bin_packing"), Task.resolve("cache_eviction")

BEST_FIT_LOOP = """def priority(item, capacities):
    scores = []
    for cap in capacities:
        leftover = cap - item
        scores.append(-leftover)
    return scores
"""
BEST_FIT_NUMPY = '''import numpy as np


def priority(item, capacities):
    """Tightest fit wins."""
    slack = np.asarray(capacities, dtype=float) - float(item)
    return -slack
'''
BEST_FIT_MAP = """def priority(item, capacities):
    fit = lambda remaining: item - remaining
    return list(map(fit, capacities))
"""
FIRST_FIT = """def priority(item, capacities):
    return [-i for i in range(len(capacities))]
"""
WITH_COUNTER = """calls = 0
sizes = []


def priority(item, capacities):
    global calls
    calls += 1
    sizes.append(item)
    mean = sum(sizes) / len(sizes)
    return [-(c - item) - abs(c - item - mean) * 0.1 for c in capacities]
"""
LOCAL_ONLY = """scores = None


def priority(item, capacities):
    scores = [-(c - item) for c in capacities]
    return scores
"""
LRU = """from collections import OrderedDict


class Policy:
    def __init__(self, capacity):
        self.capacity = capacity
        self.order = OrderedDict()

    def on_hit(self, key):
        self.order.move_to_end(key)

    def on_miss(self, key):
        self.order[key] = None

    def evict(self):
        key, _ = self.order.popitem(last=False)
        return key
"""
LFU = """class Policy:
    def __init__(self, capacity):
        self.capacity = capacity
        self.counts = {}

    def on_hit(self, key):
        self.counts[key] += 1

    def on_miss(self, key):
        self.counts[key] = 1

    def evict(self):
        victim = min(self.counts, key=self.counts.get)
        del self.counts[victim]
        return victim
"""
ARC = """from collections import OrderedDict, deque


class Policy:
    def __init__(self, capacity):
        self.capacity = capacity
        self.p = 0
        self.t1, self.t2 = OrderedDict(), OrderedDict()
        self.b1, self.b2 = deque(), deque()

    def on_hit(self, key):
        if key in self.t1:
            del self.t1[key]
        else:
            self.t2.move_to_end(key)
        self.t2[key] = None

    def on_miss(self, key):
        if key in self.b1:
            self.p = min(self.capacity, self.p + 1)
            self.b1.remove(key)
            self.t2[key] = None
        elif key in self.b2:
            self.p = max(0, self.p - 1)
            self.b2.remove(key)
            self.t2[key] = None
        else:
            self.t1[key] = None

    def evict(self):
        if self.t1 and (len(self.t1) > self.p or not self.t2):
            key, _ = self.t1.popitem(last=False)
            self.b1.append(key)
            if len(self.b1) > self.capacity:
                self.b1.popleft()
        else:
            key, _ = self.t2.popitem(last=False)
            self.b2.append(key)
            if len(self.b2) > self.capacity:
                self.b2.popleft()
        return key
"""

# What an extractor would say about each strategy, keyed on a marker in the code.
FAKE = {
    "best_fit": {"primary_signal": "capacity_values", "secondary_signal": None, "mechanism_family": "Best Fit",
                 "assumptions": ["the tightest remaining capacity is always best",
                                 "the new bin is scored like any other bin"]},
    "first_fit": {"primary_signal": "bin_order", "secondary_signal": None, "mechanism_family": "First Fit",
                  "assumptions": ["older bins should be filled first"]},
    "adaptive": {"primary_signal": "history", "secondary_signal": "capacity_values",
                 "mechanism_family": "distribution-aware fit", "assumptions": ["past item sizes predict future ones"]},
}


def fake_extract(code):
    if "sizes" in code:
        return FAKE["adaptive"]
    if "range(len(capacities))]" in code:
        return FAKE["first_fit"]
    return FAKE["best_fit"]


# ---------------------------------------------------------------- fingerprint
def test_bin_packing_fingerprints():
    bf = strategy.fingerprint(BIN, BEST_FIT_LOOP)
    assert bf["inputs_used"] == ["item", "capacity_values"]
    assert "bin_order" in bf["inputs_unused"] and bf["inputs_ambiguous"] == []
    assert bf["state"] == {"persistent": False, "features": []}
    assert bf["decision_rule"] == {"primary_signal": None, "secondary_signal": None}
    assert bf["assumptions"] == [] and bf["mechanism_family"] is None and bf["behavior_signature"] is None
    assert bf["failure_signature"] == {"weak_regimes": []}

    ff = strategy.fingerprint(BIN, FIRST_FIT)
    assert ff["inputs_used"] == ["bin_order"] and ff["state"]["persistent"] is False

    counter = strategy.fingerprint(BIN, WITH_COUNTER, extract=fake_extract)
    assert counter["state"] == {"persistent": True, "features": ["calls", "sizes"]}
    assert "history" in counter["inputs_used"]
    assert counter["decision_rule"] == {"primary_signal": "history", "secondary_signal": "capacity_values"}
    assert counter["mechanism_family"] == "distribution-aware fit"

    assert strategy.fingerprint(BIN, LOCAL_ONLY)["state"]["persistent"] is False  # shadowing local, not state
    broken = strategy.fingerprint(BIN, "def priority(item, capacities):\n    return [\n")
    assert broken["inputs_ambiguous"] == list(strategy.INPUTS["online_bin_packing"]) and broken["inputs_used"] == []


def test_cache_fingerprints():
    lru, lfu, arc = (strategy.fingerprint(CACHE, c) for c in (LRU, LFU, ARC))
    assert lru["state"] == {"persistent": True, "features": ["self.order"]}
    assert "frequency" not in lru["inputs_used"]
    assert "frequency" in lfu["inputs_used"] and lfu["state"]["features"] == ["self.counts"]
    assert {"self.b1", "self.b2", "self.p", "self.t1", "self.t2"} == set(arc["state"]["features"])
    assert {"eviction_history", "capacity", "occupancy"} <= set(arc["inputs_used"])
    assert "eviction_history" not in lru["inputs_used"] + lfu["inputs_used"]
    assert "workload_mix" in lru["inputs_ambiguous"]  # left to the extractor, as in overlook


def test_free_text_task_takes_inputs_from_the_extractor():
    task = Task.resolve("Write a function that sorts numbers.")
    fp = strategy.fingerprint(task, "def f(xs):\n    return sorted(xs)\n",
                              extract=lambda code: {"inputs_used": ["xs"], "primary_signal": "value",
                                                    "assumptions": "inputs fit in memory", "mechanism_family": "timsort"})
    assert fp["inputs_used"] == ["xs"] and fp["inputs_unused"] == [] and fp["assumptions"] == ["inputs fit in memory"]
    assert fp["behavior_signature"] is None and strategy.fingerprint(task, "x = 1", behavior=True)["behavior_signature"] is None


def test_failure_signature_bin_packing_instances():
    base = {"instances": [{"seed": s, "excess_pct": 4.0} for s in (1, 2, 3)]}
    dev = {"instances": [{"seed": 1, "excess_pct": 4.5}, {"seed": 2, "excess_pct": 3.0},
                         {"seed": 3, "excess_pct": 6.0}]}
    fp = strategy.fingerprint(BIN, BEST_FIT_LOOP, dev_detail=dev, baseline_detail=base)
    assert fp["failure_signature"]["weak_regimes"] == ["instance seed 3", "instance seed 1"]  # worst first
    assert strategy.fingerprint(BIN, BEST_FIT_LOOP, dev_detail=dev)["failure_signature"]["weak_regimes"] == []


def test_failure_signature_cache_runs():
    run = lambda trace, size, ratio: {"trace": trace, "cache_size": size, "miss_ratio": ratio}
    base = {"runs": [run("zipf", 10, 0.30), run("scan", 10, 0.50), run("scan", 100, 0.20)]}
    dev = {"runs": [run("zipf", 10, 0.29), run("scan", 10, 0.70), run("scan", 100, 0.25)]}
    assert strategy.weak_regimes(dev, base) == ["trace scan at cache size 10", "trace scan at cache size 100"]


# ---------------------------------------------------------------- similarity
def test_different_code_same_strategy():
    assert strategy.code_similarity(BEST_FIT_LOOP, BEST_FIT_NUMPY) < 0.3
    assert strategy.code_similarity(BEST_FIT_LOOP, BEST_FIT_LOOP + "\n\n") == 1.0
    a, b, ff = (strategy.fingerprint(BIN, c, extract=fake_extract) for c in (BEST_FIT_LOOP, BEST_FIT_NUMPY, FIRST_FIT))
    same = strategy.similarity(a, b)
    assert same["overall"] > 0.8 and same["parts"]["inputs"] == 1.0 and same["parts"]["behavior"] is None
    other = strategy.similarity(a, ff)
    assert other["overall"] < 0.5 < same["overall"]
    assert other["parts"]["mechanism_family"] == 0.0 and other["parts"]["inputs"] == 0.0
    # Static parts alone already separate them; parts neither side has are left out.
    static = strategy.similarity(strategy.fingerprint(BIN, BEST_FIT_LOOP), strategy.fingerprint(BIN, BEST_FIT_NUMPY))
    assert static["overall"] == 1.0 and static["parts"]["mechanism_family"] is None


def test_similarity_family_is_case_insensitive_and_assumptions_use_tokens():
    a = {"mechanism_family": "best fit", "assumptions": ["The tightest bin is best"]}
    b = {"mechanism_family": "Best Fit ", "assumptions": ["tightest bin best"]}
    parts = strategy.similarity(a, b)["parts"]
    assert parts["mechanism_family"] == 1.0 and parts["assumptions"] == 1.0
    assert strategy.similarity({}, {})["overall"] == 1.0  # only the (equal) persistent flags to compare


def test_behavior_signatures_go_through_the_sandbox():
    a, b, ff = (strategy.fingerprint(BIN, c, behavior=True) for c in (BEST_FIT_LOOP, BEST_FIT_NUMPY, FIRST_FIT))
    assert a["behavior_signature"] and a["behavior_signature"] == b["behavior_signature"]
    assert strategy.similarity(a, b)["parts"]["behavior"] == 1.0
    assert strategy.similarity(a, ff)["parts"]["behavior"] < 0.95


# ---------------------------------------------------------------- trajectory
def test_trajectory_detects_a_loop():
    attempts = [{"id": f"a{i}", "code": c, "score": s, "dev_detail": None}
                for i, (c, s) in enumerate([(BEST_FIT_LOOP, 4.15), (BEST_FIT_NUMPY, 4.14), (BEST_FIT_MAP, 4.15)])]
    report = trajectory.analyze(BIN, attempts, extract=fake_extract)
    assert report["n"] == 3 and set(report["fingerprints"]) == {"a0", "a1", "a2"}
    assert report["loop_detected"] and report["strategy_similarity"] > 0.9
    assert report["code_similarity"] < report["strategy_similarity"]
    shared = report["shared"]
    assert shared["inputs_used"] == ["item", "capacity_values"] and "history" in shared["inputs_unused"]
    assert shared["state_persistent"] is False and shared["primary_signal"] == "capacity_values"
    assert shared["mechanism_family"] == "Best Fit" and len(shared["assumptions"]) == 2
    assert report["progress"]["first"] == 4.15 and report["progress"]["best"] == 4.14
    assert report["progress"]["relative_improvement"] == pytest.approx(0.01 / 4.15)
    assert report["loop_score"] > 0.8
    assert "3/3 attempts use only inputs item, capacity_values" in report["reasons"]
    assert any("stalled" in r for r in report["reasons"])


def test_trajectory_with_different_strategies_and_progress_is_not_a_loop():
    attempts = [{"id": "ff", "code": FIRST_FIT, "score": 6.0, "dev_detail": None},
                {"id": "bf", "code": BEST_FIT_LOOP, "score": 4.2, "dev_detail": None},
                {"id": "adaptive", "code": WITH_COUNTER, "score": 3.5, "dev_detail": None}]
    report = trajectory.analyze(BIN, attempts, extract=fake_extract)
    assert not report["loop_detected"] and report["strategy_similarity"] < 0.7
    assert report["shared"]["primary_signal"] is None and report["shared"]["state_persistent"] is None
    assert report["progress"]["relative_improvement"] == pytest.approx(2.5 / 6.0)
    assert report["loop_score"] < 0.4
    assert any("improving" in r for r in report["reasons"])


def test_trajectory_same_strategy_but_improving_is_not_a_loop_and_higher_is_better():
    attempts = [{"id": "a", "code": BEST_FIT_LOOP, "score": 0.50, "dev_detail": None},
                {"id": "b", "code": BEST_FIT_NUMPY, "score": 0.60, "dev_detail": None}]
    report = trajectory.analyze(BIN, attempts, lower_is_better=False)
    assert report["progress"]["relative_improvement"] == pytest.approx(0.2) and not report["loop_detected"]


def test_trajectory_shared_weak_regimes_and_unknown_progress():
    base = {"runs": [{"trace": t, "cache_size": 10, "miss_ratio": 0.3} for t in ("zipf", "scan")]}
    detail = lambda scan, zipf: {"runs": [{"trace": "scan", "cache_size": 10, "miss_ratio": scan},
                                          {"trace": "zipf", "cache_size": 10, "miss_ratio": zipf}]}
    attempts = [{"id": "lru", "code": LRU, "score": None, "dev_detail": detail(0.6, 0.29)},
                {"id": "lru2", "code": LRU.replace("self.order", "self.od"), "score": None,
                 "dev_detail": detail(0.5, 0.31)},
                {"id": "undetailed", "code": LRU, "score": None, "dev_detail": None}]
    lru = {"primary_signal": "time", "mechanism_family": "LRU", "assumptions": ["recency predicts reuse"]}
    report = trajectory.analyze(CACHE, attempts, extract=lambda code: lru, baseline_detail=base)
    # Renaming the state attribute halves the state part; the extracted rule keeps them together.
    assert report["fingerprints"]["lru2"]["state"]["features"] == ["self.od"]
    assert report["strategy_similarity"] > 0.8
    assert report["shared"]["weak_regimes"] == ["trace scan at cache size 10"]
    assert report["progress"]["relative_improvement"] is None and report["loop_detected"]  # unknown counts as stalled
    assert any("2/2 attempts with dev results are worse than the best baseline on trace scan" in r
               for r in report["reasons"])
    assert any("progress unknown" in r for r in report["reasons"])


def test_trajectory_edge_cases():
    one = trajectory.analyze(BIN, [{"id": "x", "code": BEST_FIT_LOOP, "score": 4.0, "dev_detail": None}])
    assert one["loop_score"] == 0.0 and not one["loop_detected"] and one["strategy_similarity"] is None
    empty = trajectory.analyze(BIN, [])
    assert empty["n"] == 0 and empty["shared"]["inputs_used"] == [] and empty["reasons"] == []


# ---------------------------------------------------------------- llm_extractor
def test_llm_extractor_uses_the_extractor_model_and_caches(fake_world, monkeypatch):
    seen = []

    def fake_post(self, body, attempts=4):
        seen.append(body)
        reply = {"primary_signal": "Capacity_Values", "secondary_signal": "none",
                 "assumptions": ["ties do not matter"], "mechanism_family": "best fit"}
        return {"choices": [{"message": {"content": f"```json\n{json.dumps(reply)}\n```"}, "finish_reason": "stop"}],
                "usage": {"cost": 0.001, "completion_tokens": 10}}
    monkeypatch.setattr(LLM, "_post", fake_post)
    herd = pipeline.Herd(progress=lambda m: None)
    herd.cfg["models"]["extractor"] = MODELS[1]
    extract = strategy.llm_extractor(herd, BIN)
    first = extract(BEST_FIT_LOOP)
    assert first == {"primary_signal": "capacity_values", "secondary_signal": None,
                     "assumptions": ["ties do not matter"], "mechanism_family": "best fit"}
    assert len(seen) == 1 and seen[0]["model"] == MODELS[1]
    prompt = seen[0]["messages"][0]["content"]
    assert BEST_FIT_LOOP in prompt and "capacity_values" in prompt and "Read the CODE" in prompt
    assert strategy.llm_extractor(herd, BIN)(BEST_FIT_LOOP) == first and len(seen) == 1  # cached
    extract(BEST_FIT_NUMPY)
    assert len(seen) == 2  # different code, different key

    del herd.cfg["models"]["extractor"]
    strategy.llm_extractor(herd, BIN)(FIRST_FIT)
    assert seen[-1]["model"] == herd.cfg["models"]["labeler"]
