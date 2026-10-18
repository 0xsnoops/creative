"""Offline tests for diverge/overlook.py: static input audit on hand-written code, assumption
classification, grid gaps and minority detection."""
from diverge.overlook import (audit_bin_packing, audit_cache, classify_assumption, grid_gaps, minority)

BP = "online_bin_packing"


def used(result):
    return {k for k, (status, _) in result.items() if status == "used"}


def test_bin_packing_audit_best_fit_vs_first_fit():
    best = "def priority(item, capacities):\n    return [-(c - item) for c in capacities]\n"
    first = "def priority(item, capacities):\n    return [-i for i in range(len(capacities))]\n"
    assert used(audit_bin_packing(best)) == {"item", "capacity_values"}
    assert used(audit_bin_packing(first)) == {"bin_order"}


def test_bin_packing_audit_state_new_bin_distribution_and_counts():
    code = '''import numpy as np
SEEN = []
def priority(item, capacities):
    SEEN.append(item)
    mean = sum(SEEN) / len(SEEN)
    caps = np.asarray(capacities, dtype=float)
    scores = -(caps - item) + 0.01 * mean - 0.001 * len(capacities)
    scores[-1] -= 45
    return scores
'''
    u = used(audit_bin_packing(code))
    assert {"item", "capacity_values", "history", "new_bin_slot", "n_open_bins", "item_distribution"} <= u
    assert "bin_order" not in u


def test_bin_packing_audit_index_used_only_for_reading_is_not_order():
    code = ("def priority(item, capacities):\n    out = []\n    for i in range(len(capacities)):\n"
            "        out.append(-(capacities[i] - item))\n    return out\n")
    u = used(audit_bin_packing(code))
    assert "bin_order" not in u and "capacity_values" in u


def test_bin_packing_audit_new_bin_by_position():
    code = ("def priority(item, capacities):\n    n = len(capacities)\n"
            "    return [(-1000 if i == n - 1 else 0) + c for i, c in enumerate(capacities)]\n")
    assert "new_bin_slot" in used(audit_bin_packing(code))


LRU = '''from collections import OrderedDict
class Policy:
    def __init__(self, capacity):
        self.order = OrderedDict()
    def on_hit(self, key):
        self.order.move_to_end(key)
    def on_miss(self, key):
        self.order[key] = None
    def evict(self):
        return next(iter(self.order))
'''
LFU = '''class Policy:
    def __init__(self, capacity):
        self.count = {}
    def on_hit(self, key):
        self.count[key] += 1
    def on_miss(self, key):
        self.count[key] = 1
    def evict(self):
        victim = min(self.count, key=self.count.get)
        del self.count[victim]
        return victim
'''
GHOST = '''from collections import OrderedDict
class Policy:
    def __init__(self, capacity):
        self.cap = capacity
        self.order = OrderedDict()
        self.ghost = set()
        self.tick = 0
        self.last = None
    def on_hit(self, key):
        self.tick += 1
        self.order.move_to_end(key)
    def on_miss(self, key):
        self.tick += 1
        if key in self.ghost:
            self.order[key] = self.tick
        else:
            self.order[key] = 0
        if self.last is not None and key == self.last + 1:
            self.order[key] = -1
        self.last = key
    def evict(self):
        if len(self.order) >= self.cap:
            victim = min(self.order, key=self.order.get)
        else:
            victim = next(iter(self.order))
        self.order.pop(victim)
        self.ghost.add(victim)
        return victim
'''


def test_cache_audit():
    lru, lfu, ghost = audit_cache(LRU), audit_cache(LFU), audit_cache(GHOST)
    assert lru["capacity"][0] == "unused" and lru["key_value"][0] == "unused"
    assert lfu["frequency"][0] == "used"
    assert {"capacity", "key_value", "time", "eviction_history", "occupancy"} <= used(ghost)
    assert lru["workload_mix"][0] == "ambiguous"  # left to the extractor by design


PROMPT = ("The function is called once per item. Module-level state (e.g. statistics of past items) is allowed; "
          "your module is freshly loaded for each instance.")


def test_assumption_classification_needs_a_verbatim_quote():
    hard = {"assumption": "module state is reset per instance",
            "prompt_quote": "your module is freshly loaded for each instance"}
    assert classify_assumption(hard, PROMPT)[0] == "hard"
    paraphrased = {"assumption": "state resets", "prompt_quote": "the module is reloaded for every instance"}
    assert classify_assumption(paraphrased, PROMPT)[0] == "incidental"
    short = {"assumption": "called per item", "prompt_quote": "once per item"}  # under 6 words
    assert classify_assumption(short, PROMPT)[0] == "incidental"
    assert classify_assumption({"assumption": "no history used", "prompt_quote": ""}, PROMPT) == ("incidental", None)


def test_grid_gaps():
    session = {"s1": {"score": "tight fit", "state": "none"}, "s2": {"score": "tight fit", "state": "none"},
            "s3": {"score": "gap balance", "state": "none"}, "s4": {"score": "gap balance", "state": "item histogram"},
            "s5": {"score": "tight fit", "state": "none"}, "s6": {"score": "gap balance", "state": "item histogram"}}
    lit = {"Harmonic": {"score": "size classes", "state": "class counts"},
           "Best Fit": {"score": "tight fit", "state": "none"}}
    lit_gaps, pairs = grid_gaps(session, lit)
    assert [g["family"] for g in lit_gaps] == ["Harmonic"]  # Best Fit's cell is occupied by the consensus
    assert {"pair": [["score", "tight fit"], ["state", "item histogram"]], "support": [3, 2]} in pairs
    assert all(p["pair"] != [["score", "gap balance"], ["state", "item histogram"]] for p in pairs)  # seen together


def test_minority_detection():
    models = ["a", "b", "c", "d", "e"]
    items = {"common": {m: ["x"] for m in models},
             "two": {"a": ["s1", "s2"], "c": ["s3"]},
             "one": {"e": ["s4"]},
             "three": {"a": ["s5"], "b": ["s6"], "c": ["s7"]}}
    out = minority(items, models)
    assert [r["item"] for r in out] == ["one", "two"]
    assert out[1]["absent_from"] == ["b", "d", "e"] and out[1]["count"] == 3


def test_build_end_to_end_with_fake_models(monkeypatch):
    import json
    import re
    import pytest
    from diverge import overlook, pipeline
    from diverge.exa import Exa
    from diverge.llm import LLM
    from diverge.task import Task
    from tests.test_pipeline_offline import reply_to
    task = Task.resolve(BP)

    def fake(prompt):
        if prompt.startswith("These solutions were written by AI models"):
            return json.dumps({"assumptions": [
                {"assumption": "decides from the current item and capacities only", "prompt_quote": ""},
                {"assumption": "module may keep state", "prompt_quote": "Module-level state (e.g. statistics of past items) is allowed"},
                {"assumption": "quotes something not in the prompt", "prompt_quote": "the bins are sorted by size before every call"}]})
        if prompt.startswith("For each numbered assumption"):
            ids = re.findall(r'<solution id="([^"]+)"', prompt)
            return json.dumps({"checks": [{"assumption": a, "sample_id": i, "holds": a != 3 or i != ids[0], "evidence": "l1"}
                                          for a in (1, 2, 3) for i in ids]})
        if prompt.startswith("Decompose each solution"):
            ids = re.findall(r'<(?:solution|family) id="([^"]+)"', prompt)
            return json.dumps({"items": [{"id": i, "scoring_term": "harmonic classes" if "Harmonic" in i else "bin order",
                                          "tie_break": "earliest", "state_kept": "class counts" if "Harmonic" in i else "none",
                                          "new_bin_rule": "only if nothing fits"} for i in ids]})
        if prompt.startswith("Merge these values"):
            values = re.findall(r"^- (.*)$", prompt, re.M)
            return json.dumps({"mapping": {v: v for v in values}})
        if prompt.startswith("For the task below, a solution would combine"):
            n = len(re.findall(r"^\d+\. ", prompt, re.M))
            return json.dumps({"pairs": [{"n": i + 1, "coherent": True, "reason": "ok"} for i in range(n)]})
        if "Do NOT repeat these approaches" in prompt:
            return json.dumps({"approaches": [{"name": "Harmonic", "mechanism": "size classes; one open bin per class",
                                               "reference": ""}, {"name": "Mystery fit", "mechanism": "vibes"}]})
        if "Which approaches do you know" in prompt:
            return json.dumps({"approaches": [{"name": "First Fit", "mechanism": "first bin that fits"}]})
        if prompt.startswith("Match each proposed approach"):
            names = re.findall(r"^\d+\. ([^:]+):", prompt, re.M)
            fams = {"Harmonic": "Harmonic", "First Fit": "First Fit"}
            return json.dumps({"matches": [{"n": i + 1, "family": fams.get(n, "none"), "precise": n in fams}
                                           for i, n in enumerate(names)]})
        return reply_to(prompt)

    import tests.test_pipeline_offline as offline
    import yaml, tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    cfg = yaml.safe_load((pipeline.ROOT / "config.yaml").read_text())
    cfg["models"] = {"samplers": offline.MODELS, "labeler": offline.MODELS[0], "implementer": offline.MODELS[0],
                     "extractor": offline.MODELS[1]}
    cfg["sampling"]["n_per_model"] = 3
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg))
    (tmp / "cache").mkdir()
    (tmp / "cache" / "openrouter_models.json").write_text(json.dumps({"data": [
        {"id": m, "pricing": {"prompt": "0.000001", "completion": "0.000002"}, "top_provider": {"max_completion_tokens": 64000}}
        for m in offline.MODELS]}))
    monkeypatch.setattr(pipeline, "ROOT", tmp)
    monkeypatch.setattr(pipeline, "_caches", {})
    seen = {"models": []}

    def post(self, body, attempts=4):
        seen["models"].append(body["model"])
        return {"choices": [{"message": {"content": fake(body["messages"][-1]["content"]
                                                           if body["messages"][0]["role"] == "system"
                                                           else body["messages"][0]["content"])},
                             "finish_reason": "stop"}], "usage": {"cost": 0.001}}
    monkeypatch.setattr(LLM, "_post", post)
    monkeypatch.setattr(Exa, "_post", lambda self, request, attempts=3: {
        "costDollars": {"total": 0.005}, "results": [{"url": f"https://{x}.example", "title": x, "highlights": ["..."]} for x in "abc"]})
    h = pipeline.Session(progress=lambda m: None)
    pipeline.ensure_map(h, task)
    result = overlook.build(h, task)
    inputs = {r["input"]: r for r in result["inputs"]}
    assert inputs["bin_order"]["share"] == 1.0 and inputs["history"]["overlooked"]  # FIRST_FIT uses only order
    a = {x["assumption"]: x for x in result["assumptions"]["assumptions"]}
    assert a["module may keep state"]["kind"] == "hard"           # verbatim prompt quote
    assert a["quotes something not in the prompt"]["kind"] == "incidental"
    assert a["decides from the current item and capacities only"]["shared"] is True
    assert a["quotes something not in the prompt"]["shared"] is False  # one checked sample does not hold it
    assert "Harmonic" in [g["family"] for g in result["grid"]["literature_only_cells"]]
    lt = result["latent_tail"]
    # with the consensus shown: Harmonic (taxonomy) counts; "Mystery fit" (vague, no source mentions it) does not.
    # without: only First Fit, which is a consensus family here, so nothing counts.
    assert lt["verified_non_herd_per_model"][offline.MODELS[0]] == {"with_herd": 1, "without_herd": 0}
    assert offline.MODELS[1] in seen["models"]  # the extractor model did the reading
    assert (tmp / "results" / BP / "overlook.json").exists()
