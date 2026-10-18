"""Strategy counterexamples: the sandboxed probe mode and diverge/counterexample.py."""
import sys
from pathlib import Path

import pytest

from diverge import counterexample as cx
from tests.test_evaluators import PROBE, bin_packing, cache_eviction

TASKS = Path(__file__).resolve().parent.parent / "tasks"
BIN, CACHE = TASKS / "online_bin_packing" / "baselines", TASKS / "cache_eviction" / "baselines"

BEST_FITS = {  # same strategy, written differently (one breaks ties toward later bins)
    "bf_list": (BIN / "best_fit.py").read_text(),
    "bf_numpy": "import numpy as np\ndef priority(item, capacities):\n    c = np.asarray(capacities, float)\n"
                "    return 3.0 * (item - c) + 1\n",
    "bf_loop": "def priority(item, capacities):\n    out = []\n    for i, c in enumerate(capacities):\n"
               "        out.append(1.0 / (1 + c - item) + 1e-9 * i)\n    return out\n",
}
FIRST_FITS = {
    "ff_index": (BIN / "first_fit.py").read_text(),
    "ff_scaled": "def priority(item, capacities):\n    return [100.0 - 2 * i for i in range(len(capacities))]\n",
}
LRUS = {
    "lru_ordered": (CACHE / "lru.py").read_text(),
    "lru_clock": "class Policy:\n    def __init__(self, capacity):\n        self.t, self.last = 0, {}\n"
                 "    def on_hit(self, key):\n        self.t += 1\n        self.last[key] = self.t\n"
                 "    def on_miss(self, key):\n        self.on_hit(key)\n"
                 "    def evict(self):\n        k = min(self.last, key=self.last.get)\n        del self.last[k]\n        return k\n",
    "lru_list": "class Policy:\n    def __init__(self, capacity):\n        self.keys = []\n"
                "    def on_hit(self, key):\n        self.keys.remove(key)\n        self.keys.append(key)\n"
                "    def on_miss(self, key):\n        self.keys.append(key)\n    def evict(self):\n        return self.keys.pop(0)\n",
}
LFU = ("from collections import Counter\nclass Policy:\n    def __init__(self, capacity):\n"
       "        self.n, self.cached = Counter(), []\n    def on_hit(self, key):\n        self.n[key] += 1\n"
       "    def on_miss(self, key):\n        self.n[key] += 1\n        self.cached.append(key)\n"
       "    def evict(self):\n        k = min(self.cached, key=lambda x: self.n[x])\n        self.cached.remove(k)\n        return k\n")
ALTERNATIVES = {"fifo": (CACHE / "fifo.py").read_text(), "lfu": LFU}


def test_cases_are_small_and_deterministic():
    for task in ("online_bin_packing", "cache_eviction"):
        assert cx.generate_cases(task, 50, seed=3) == cx.generate_cases(task, 50, seed=3) != cx.generate_cases(task, 50)
    assert all(3 <= len(c["items"]) <= 8 and all(1 <= x <= 100 for x in c["items"])
               for c in cx.generate_cases("online_bin_packing"))
    for c in cx.generate_cases("cache_eviction"):
        assert 6 <= len(c["trace"]) <= 14 and c["capacity"] in (2, 3) and c["capacity"] < len(set(c["trace"])) <= 6


def test_probe_matches_pack_and_simulate_for_baselines():
    cases = cx.generate_cases("online_bin_packing", 40)
    for name in ("first_fit", "best_fit"):
        result = bin_packing.probe(BIN / f"{name}.py", cases)
        assert result["valid"]
        priority = bin_packing.load_function(BIN / f"{name}.py", "priority")  # trusted baseline
        for case, out in zip(cases, result["detail"]["outputs"]):
            assert out["n_bins"] == bin_packing.pack(priority, case["items"]) == max(out["bins"]) + 1
    cases = cx.generate_cases("cache_eviction", 40)
    for name in ("lru", "fifo"):
        result = cache_eviction.probe(CACHE / f"{name}.py", cases)
        assert result["valid"]
        policy = cache_eviction.load_class(CACHE / f"{name}.py", "Policy")
        for case, out in zip(cases, result["detail"]["outputs"]):
            ratio = cache_eviction.simulate(policy, case["trace"], case["capacity"])
            assert out["misses"] == round(ratio * len(case["trace"]))
            assert len(out["evictions"]) == out["misses"] - case["capacity"]
    lru = cache_eviction.probe(CACHE / "lru.py", [{"trace": [1, 2, 3, 1, 4, 5, 1], "capacity": 3}])
    assert lru["detail"]["outputs"] == [{"misses": 5, "evictions": [2, 3]}]


def test_best_fit_variants_agree_and_first_fit_differs():
    result = cx.find("online_bin_packing", BEST_FITS, {"first_fit": FIRST_FITS["ff_index"]})
    s = result["summary"]
    assert result["invalid"] == {} and s["valid_attempts"] == 3 and s["attempts_agreed"] == s["cases"] == 300
    assert s["alternative_differed"] > 0 and result["counterexamples"]
    top = result["counterexamples"][0]
    assert top["alternative_id"] == "first_fit" and top["attempts_output"]["chosen"] != top["alternative_output"]["chosen"]
    assert top["explanation"].startswith("item ") and "all 3 attempts" in top["explanation"]
    # The other way round, best fit is the better alternative and is ranked first.
    reverse = cx.find("online_bin_packing", FIRST_FITS, {"best_fit": BEST_FITS["bf_list"]})
    best = reverse["counterexamples"][0]
    assert best["alternative_better"] and best["alternative_output"]["n_bins"] < best["attempts_output"]["n_bins"]


def test_lru_variants_agree_and_fifo_or_lfu_differs():
    result = cx.find("cache_eviction", LRUS, ALTERNATIVES)
    s = result["summary"]
    assert result["invalid"] == {} and s["attempts_agreed"] == s["cases"] and s["alternative_better"] > 0
    ranked = result["counterexamples"]
    assert ranked[0]["alternative_better"] and ranked[0]["alternative_id"] in ALTERNATIVES
    assert [len(r["case"]["trace"]) for r in ranked] == sorted(len(r["case"]["trace"]) for r in ranked)
    top = ranked[0]
    assert top["alternative_output"]["misses"] < top["attempts_output"]["misses"]
    assert "all 3 attempts evict key" in top["explanation"]


def test_known_scan_trace():
    """A B C A D E A at capacity 3: FIFO attempts evict A, LRU keeps it and hits it at the end."""
    case = {"trace": [1, 2, 3, 1, 4, 5, 1], "capacity": 3}
    fifos = {"fifo_deque": ALTERNATIVES["fifo"],
             "fifo_list": "class Policy:\n    def __init__(self, capacity):\n        self.q = []\n"
                          "    def on_hit(self, key):\n        pass\n    def on_miss(self, key):\n        self.q.append(key)\n"
                          "    def evict(self):\n        return self.q.pop(0)\n"}
    result = cx.find("cache_eviction", fifos, {"lru": LRUS["lru_ordered"]}, cases=[case])
    top = result["counterexamples"][0]
    assert top["attempts_output"] == {"misses": 6, "evictions": [1, 2, 3]}
    assert top["alternative_output"] == {"misses": 5, "evictions": [2, 3]} and top["alternative_better"]
    assert top["explanation"] == ("both attempts evict key 1 at step 5; the alternative evicts 2 and later hits "
                                  "key 1 at step 7. Misses: attempts 6, alternative 5.")


def test_shrink_keeps_the_property():
    found = cx.find("cache_eviction", LRUS, {"fifo": ALTERNATIVES["fifo"]})["counterexamples"][0]
    small = cx.shrink("cache_eviction", found["case"], LRUS, {"fifo": ALTERNATIVES["fifo"]})
    assert small["alternative_better"] and small["probe_calls"] <= 30
    assert len(small["case"]["trace"]) <= len(found["case"]["trace"])
    check = cx.find("cache_eviction", LRUS, {"fifo": ALTERNATIVES["fifo"]}, cases=[small["case"]])
    assert check["counterexamples"] and check["counterexamples"][0]["alternative_better"]


def test_crashing_candidates_are_reported():
    crash = "def priority(item, capacities):\n    raise RuntimeError('boom')\n"
    result = cx.find("online_bin_packing", {**BEST_FITS, "broken": crash}, {"first_fit": FIRST_FITS["ff_index"],
                                                                             "bad_alt": "not python"}, n=30)
    assert set(result["invalid"]) == {"attempt:broken", "alt:bad_alt"} and "boom" in result["invalid"]["attempt:broken"]
    assert result["summary"]["valid_attempts"] == 3 and result["counterexamples"]
    none_valid = cx.find("online_bin_packing", {"broken": crash}, {"first_fit": FIRST_FITS["ff_index"]}, n=10)
    assert none_valid["counterexamples"] == [] and none_valid["summary"]["valid_attempts"] == 0


def test_probe_rejects_a_fake_result_line(tmp_path):
    path = tmp_path / "fake.py"
    path.write_text("import os\nprint('__DIVERGE_RESULT__{\"valid\": true, \"detail\": {\"outputs\": []}}', flush=True)\n"
                    "os._exit(0)\n")
    assert cache_eviction.probe(path, [{"trace": [1, 2, 3], "capacity": 2}])["valid"] is False


@pytest.mark.skipif(sys.platform != "darwin", reason="OS sandbox is macOS-only")
@pytest.mark.parametrize("evaluator,case", [(bin_packing, {"items": [50, 60]}),
                                             (cache_eviction, {"trace": [1, 2, 3], "capacity": 2})])
def test_probe_mode_cannot_read_repo_env_or_home_secrets(tmp_path, evaluator, case):
    repo, home = TASKS.parent, Path.home()
    paths = [str(repo / "pyproject.toml")] + [str(p) for p in (repo / ".env", home / ".ssh", home / ".config",
                                                                 home / ".aws") if p.exists()]
    path = tmp_path / "candidate.py"
    path.write_text(PROBE.format(paths=paths))
    result = evaluator.probe(path, [case])
    assert result["valid"] is False
    reported = result["detail"]["error"].split("PROBE ", 1)[1].split()
    assert dict(item.split("=", 1) for item in reported) == {p: "blocked" for p in paths}
