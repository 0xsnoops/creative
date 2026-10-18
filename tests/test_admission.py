"""Admission gate (diverge/admission.py) and the evaluator's sandboxed perturbation mode."""
import sys
from pathlib import Path

import numpy as np
import pytest

from diverge import admission
from diverge.fingerprint import CANONICAL
from diverge.overlook import INPUTS
from diverge.task import Task
from tests.test_evaluators import PROBE, bin_packing, write

TASK = Task.registered("online_bin_packing")
REPO = Path(__file__).resolve().parent.parent
BF, FF = CANONICAL["best_fit"], CANONICAL["first_fit"]
# Opens a new bin whenever an even number of options is offered: depends on len(capacities).
PARITY = ("def priority(item, capacities):\n    if len(capacities) % 2 == 0:\n"
          "        return [0.0] * (len(capacities) - 1) + [1.0]\n    return [-(c - item) for c in capacities]\n")
# Module-level history: the first 100 decisions always open a new bin, then best fit.
HISTORY = ("seen = []\ndef priority(item, capacities):\n    seen.append(item)\n    if len(seen) <= 100:\n"
           "        return [0.0] * (len(capacities) - 1) + [1.0]\n    return [-(c - item) for c in capacities]\n")
CRASH = "def priority(item, capacities):\n    raise RuntimeError('boom')\n"


def test_best_fit_vs_itself_is_not_admitted():
    r = admission.admit(TASK, BF, [BF], "n_open_bins")
    assert r["valid"] and not r["admitted"]
    assert r["disagreement"] == r["min_disagreement_vs_stuck"] == 0.0
    assert any("too close" in x for x in r["reasons"])


def test_first_fit_vs_best_fit_disagreement_is_measured():
    # Measured: first fit and best fit differ on 7.0% of the 200 probe states (most probes offer
    # 0-1 open bins), so at the default 10% first fit does NOT clear the gate.
    r = admission.admit(TASK, FF, [BF], "bin_order")
    assert r["valid"] and r["min_disagreement_vs_stuck"] == pytest.approx(0.07)
    assert not r["admitted"] and r["sensitivity"] > 0.9  # first fit is all about bin order
    assert admission.admit(TASK, FF, [BF], "bin_order", min_disagreement=0.05)["admitted"]


def test_disagreement_uses_the_minimum_over_stuck_programs():
    r = admission.admit(TASK, PARITY, {"bf": BF, "ff": FF, "parity": PARITY}, "n_open_bins")
    assert r["disagreement_vs_stuck"]["parity"] == 0.0 and r["min_disagreement_vs_stuck"] == 0.0
    assert r["disagreement"] > 0.2 and not r["admitted"]


def test_policy_using_n_open_bins_is_sensitive_and_admitted():
    r = admission.admit(TASK, PARITY, [BF, FF], "n_open_bins")
    assert r["min_disagreement_vs_stuck"] >= 0.10  # measured 0.375
    assert r["sensitivity"] >= 0.05  # measured 1.0 on the 103 probes with an open bin
    assert r["admitted"] and r["probes_perturbed"] > 0


def test_best_fit_does_not_depend_on_n_open_bins():
    r = admission.admit(TASK, BF, [FF], "n_open_bins", min_disagreement=0.0)
    assert r["sensitivity"] == 0.0 and r["sensitivity_noise"] == 0.0
    assert not r["admitted"] and any("does not depend on n_open_bins" in x for x in r["reasons"])


def test_history_policy_is_sensitive_to_history():
    r = admission.admit(TASK, HISTORY, [BF], "history")
    assert r["sensitivity"] > 0.05 and r["admitted"]  # measured 0.125
    assert admission.admit(TASK, BF, [FF], "history", min_disagreement=0.0)["sensitivity"] == 0.0


@pytest.mark.parametrize("lever", ["new_bin_slot", "item_distribution", "bin_capacity", "not_an_input"])
def test_no_sound_perturbation_decides_on_disagreement_only(lever):
    r = admission.admit(TASK, PARITY, [BF], {"type": "unused_input", "name": lever})
    assert r["sensitivity"] is None and r["admitted"]
    assert f"no sound perturbation for {lever}; admission decided on disagreement only" in r["reasons"]


def test_crashing_candidate_is_invalid():
    r = admission.admit(TASK, CRASH, [BF], "item")
    assert r["valid"] is False and r["admitted"] is False and "boom" in r["reasons"][0]


def test_candidate_crashing_only_after_perturbation_is_invalid():
    # The plain fingerprint makes exactly 200 calls; only the history warm-up goes past that.
    code = ("calls = [0]\ndef priority(item, capacities):\n    calls[0] += 1\n    if calls[0] > 200:\n"
            "        raise RuntimeError('late')\n    return [-(c - item) for c in capacities]\n")
    r = admission.admit(TASK, code, [FF], "history", min_disagreement=0.0)
    assert r["valid"] is False and not r["admitted"] and "perturbed probes" in r["reasons"][0]


def test_unsupported_task_raises():
    with pytest.raises(ValueError):
        admission.admit(Task.registered("cache_eviction"), "", [], "time")


def test_perturbation_base_matches_the_fingerprint():
    path = REPO / "tasks" / "online_bin_packing" / "baselines" / "best_fit.py"
    plain = bin_packing.fingerprint(path)["detail"]["choices"]
    d = bin_packing.fingerprint_perturbed(path, "capacity_values")["detail"]
    assert d["base"] == d["repeat"] == plain and len(d["applicable"]) == len(plain)


@pytest.mark.parametrize("name", ["item", "capacity_values", "bin_order", "n_open_bins"])
def test_perturbed_states_are_feasible_and_keep_the_new_bin_last(name):
    rng, applied = np.random.default_rng(0), 0
    for item, caps in bin_packing.make_probes():
        p = bin_packing.perturb_probe(name, item, caps, rng)
        if p is None:
            continue
        applied += 1
        item2, caps2, back = p
        assert caps2[-1] == bin_packing.CAPACITY and back[-1] == bin_packing.CAPACITY and len(back) == len(caps2)
        assert all(item2 <= c for c in caps2) and 1 <= item2 and (item2, caps2) != (item, caps)
        if name in ("bin_order", "n_open_bins"):
            assert set(caps2) == set(caps) and back == caps2  # same options
        if name == "capacity_values":
            assert back == caps and all(abs(a - b) <= 1 and a <= 99 for a, b in zip(caps2[:-1], caps[:-1]))
        if name == "item":
            assert abs(item2 - item) == 1 and caps2 == caps
    assert applied > 0
    assert set(admission.PERTURBATIONS["online_bin_packing"]) == set(INPUTS["online_bin_packing"])


def test_perturbations_table_matches_the_evaluator():
    sound = {k for k, v in admission.PERTURBATIONS["online_bin_packing"].items() if v}
    assert sound == set(bin_packing.PERTURBABLE)


@pytest.mark.skipif(sys.platform != "darwin", reason="OS sandbox is macOS-only")
def test_perturbation_mode_cannot_read_repo_env(tmp_path):
    home = Path.home()
    paths = [str(REPO / "pyproject.toml")] + [str(p) for p in (REPO / ".env", home / ".ssh") if p.exists()]
    result = bin_packing.fingerprint_perturbed(write(tmp_path, PROBE.format(paths=paths)), "n_open_bins")
    assert result["valid"] is False
    line = result["detail"]["error"]
    results = dict(item.split("=", 1) for item in line.split("PROBE ", 1)[1].split())
    assert results == {p: "blocked" for p in paths}
