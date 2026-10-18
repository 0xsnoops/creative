import importlib.util
from pathlib import Path

import pytest

TASKS = Path(__file__).resolve().parent.parent / "tasks"


def load(task):
    spec = importlib.util.spec_from_file_location(f"{task}_evaluator", TASKS / task / "evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bin_packing, cache_eviction = load("online_bin_packing"), load("cache_eviction")


def write(tmp_path, code):
    path = tmp_path / "candidate.py"
    path.write_text(code)
    return path


def test_best_fit_beats_first_fit():
    scores = {name: r["score"] for name, r in bin_packing.baselines().items()}
    assert scores["best_fit"] < scores["first_fit"]


def test_bin_packing_lower_bound_holds():
    result = bin_packing.evaluate(TASKS / "online_bin_packing" / "baselines" / "best_fit.py")
    assert result["valid"]
    assert all(r["bins"] >= r["lower_bound"] for r in result["detail"]["instances"])


def test_lru_beats_fifo_on_zipf():
    trace = cache_eviction.trace_zipf(1)
    lru = cache_eviction.load_class(TASKS / "cache_eviction" / "baselines" / "lru.py", "Policy")
    fifo = cache_eviction.load_class(TASKS / "cache_eviction" / "baselines" / "fifo.py", "Policy")
    for fraction in cache_eviction.CACHE_FRACTIONS:
        size = round(fraction * len(set(trace)))
        assert cache_eviction.simulate(lru, trace, size) < cache_eviction.simulate(fifo, trace, size)


def test_traces_are_deterministic():
    assert cache_eviction.make_traces() == cache_eviction.make_traces()


@pytest.mark.parametrize("code", [
    "def priority(item, capacities):\n    raise RuntimeError('boom')\n",
    "def priority(item, capacities):\n    return [0.0]\n",                      # wrong length
    "def priority(item, capacities):\n    return [float('nan')] * len(capacities)\n",
    "this is not python",
])
def test_bin_packing_bad_candidates_are_invalid(tmp_path, code):
    result = bin_packing.evaluate(write(tmp_path, code))
    assert result["valid"] is False and result["score"] is None and result["detail"]["error"]


@pytest.mark.parametrize("code", [
    "class Policy:\n    def __init__(self, c): pass\n    def on_hit(self, k): pass\n"
    "    def on_miss(self, k): pass\n    def evict(self): return -12345\n",      # key not in cache
    "class Policy:\n    pass\n",
])
def test_cache_bad_candidates_are_invalid(tmp_path, code):
    result = cache_eviction.evaluate(write(tmp_path, code))
    assert result["valid"] is False and result["detail"]["error"]


def test_timeout_is_invalid(tmp_path):
    code = "import time\ndef priority(item, capacities):\n    time.sleep(10)\n    return capacities\n"
    result = bin_packing.evaluate(write(tmp_path, code), timeout=2)
    assert result == {"valid": False, "score": None, "detail": {"error": "timeout after 2s"}}


def test_candidate_stdout_does_not_break_parsing(tmp_path):
    code = "def priority(item, capacities):\n    print('debug', item)\n    return [-(c - item) for c in capacities]\n"
    result = bin_packing.evaluate(write(tmp_path, code))
    best_fit = bin_packing.evaluate(TASKS / "online_bin_packing" / "baselines" / "best_fit.py")
    assert result["valid"] and result["score"] == best_fit["score"]
