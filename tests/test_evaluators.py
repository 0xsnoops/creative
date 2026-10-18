import importlib.util
import json
import sys
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


BEST_FIT_BODY = "    return [-(c - item) for c in capacities]\n"


def test_candidate_cannot_fake_a_result_line(tmp_path):
    code = ("import os\nprint('__HERD_RESULT__' + '{\"valid\": true, \"score\": 0.0, \"detail\": {}}', flush=True)\n"
            "os._exit(0)\ndef priority(item, capacities):\n" + BEST_FIT_BODY)
    result = bin_packing.evaluate(write(tmp_path, code))
    assert result["valid"] is False


def test_candidate_cannot_see_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    code = ("import os\nassert 'OPENROUTER_API_KEY' not in os.environ, 'key leaked'\n"
            "def priority(item, capacities):\n" + BEST_FIT_BODY)
    assert bin_packing.evaluate(write(tmp_path, code))["valid"]


@pytest.mark.skipif(sys.platform != "darwin", reason="OS sandbox is macOS-only")
def test_candidate_has_no_network(tmp_path):
    code = ("import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)\n"
            "def priority(item, capacities):\n" + BEST_FIT_BODY)
    result = bin_packing.evaluate(write(tmp_path, code))
    assert result["valid"] is False and "not permitted" in result["detail"]["error"].lower()


@pytest.mark.skipif(sys.platform != "darwin", reason="OS sandbox is macOS-only")
def test_candidate_cannot_write_outside_scratch(tmp_path):
    target = tmp_path / "pwned.txt"
    code = (f"open({str(target)!r}, 'w').write('x')\n"
            "def priority(item, capacities):\n" + BEST_FIT_BODY)
    result = bin_packing.evaluate(write(tmp_path, code))
    assert result["valid"] is False and not target.exists()


def test_heldout_split_is_separate_and_keeps_baseline_order():
    dev = {n: r["score"] for n, r in bin_packing.baselines("dev").items()}
    heldout = {n: r["score"] for n, r in bin_packing.baselines("heldout").items()}
    assert heldout["best_fit"] < heldout["first_fit"] and heldout != dev
    assert set(bin_packing.SPLITS["dev"]).isdisjoint(bin_packing.SPLITS["heldout"])
    assert cache_eviction.make_traces("heldout")["zipf"] != cache_eviction.make_traces("dev")["zipf"]


# --- Sandbox hardening (review I1): candidates must not read secrets or leak them via errors.

REPO = TASKS.parent
PROBE = '''import os
def probe(path):
    try:
        if os.path.isdir(path):
            os.listdir(path)
        else:
            open(path).read(1)
        return "readable"
    except Exception:
        return "blocked"
paths = {paths!r}
raise RuntimeError("PROBE " + " ".join(p + "=" + probe(p) for p in paths))
def priority(item, capacities):
    return [0] * len(capacities)
'''


def probe_results(tmp_path, paths):
    result = bin_packing.evaluate(write(tmp_path, PROBE.format(paths=paths)))
    assert result["valid"] is False
    line = result["detail"]["error"]
    return dict(item.split("=", 1) for item in line.split("PROBE ", 1)[1].split())


@pytest.mark.skipif(sys.platform != "darwin", reason="OS sandbox is macOS-only")
def test_candidate_cannot_read_repo_files_or_home_secrets(tmp_path):
    home = Path.home()
    paths = [str(REPO / "pyproject.toml")] + [str(p) for p in (REPO / ".env", home / ".ssh", home / ".config",
                                                                 home / ".aws") if p.exists()]
    results = probe_results(tmp_path, paths)
    assert results == {p: "blocked" for p in paths}


def test_secret_values_are_redacted_from_evaluator_errors(tmp_path, monkeypatch):
    fake = "sk-or-v1-" + "0123456789abcdef" * 4
    monkeypatch.setenv("OPENROUTER_API_KEY", fake)
    code = f"raise RuntimeError('leaking {fake} and token=abc123secretvalue456')\n"
    result = bin_packing.evaluate(write(tmp_path, code))
    text = json.dumps(result)
    assert fake not in text and "[REDACTED]" in result["detail"]["error"]
