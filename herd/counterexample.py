"""Strategy counterexamples: tiny cases on which an agent's recent attempts all behave the same
while an alternative behaves differently (ideally better). They show concretely that code that
looks different is the same strategy, e.g. for caching, trace 1 2 3 1 4 5 1 at capacity 3: every
attempt evicts key 1, the alternative keeps it.

Candidates run only in the evaluator's sandbox (evaluator.probe, one call per candidate with all
cases batched); this module only compares the outputs. Supported tasks: online_bin_packing and
cache_eviction. Agreement is on these random tiny cases only, not a proof of equivalence.

Bin packing outputs are compared on the free space of each chosen bin, so picks between equal
bins (ties) do not count as different behavior."""
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor

from herd.task import Task

BIN, CACHE = "online_bin_packing", "cache_eviction"
CAPACITY = 100


def _name(task):
    return getattr(task, "name", task)


def _evaluator(task):
    task = Task.registered(task) if isinstance(task, str) else task
    if not (task.evaluator and hasattr(task.evaluator, "probe")):
        raise ValueError(f"task {task.name!r} has no probe mode")
    return task.evaluator


def generate_cases(task_name, n=300, seed=0):
    """n small random cases, deterministic given seed."""
    rng, name = random.Random(f"counterexample:{_name(task_name)}:{seed}"), _name(task_name)
    if name == BIN:
        def size():  # mostly the task's Weibull(3, 45) item sizes, some uniform
            x = rng.weibullvariate(45, 3) if rng.random() < 0.8 else rng.uniform(1, CAPACITY)
            return min(CAPACITY, max(1, round(x)))
        return [{"items": [size() for _ in range(rng.randint(3, 8))]} for _ in range(n)]
    if name != CACHE:
        raise ValueError(f"no counterexample cases for task {name!r}")
    cases = []
    while len(cases) < n:
        length, k, capacity = rng.randint(6, 14), rng.randint(3, 6), rng.randint(2, 3)
        pattern = rng.choice(["uniform", "skewed", "scan", "loop"])
        if pattern == "uniform":
            trace = [rng.randint(1, k) for _ in range(length)]
        elif pattern == "skewed":  # key 1 hottest
            trace = [min(k, 1 + int(rng.expovariate(0.8))) for _ in range(length)]
        elif pattern == "scan":  # a hot key interleaved with a run of new keys (A B C A D E A)
            trace, fresh = [], 2
            while len(trace) < length:
                if rng.random() < 0.35:
                    trace.append(1)
                else:
                    trace.append(fresh)
                    fresh = fresh + 1 if fresh < k else 2
        else:  # cyclic loop over k keys, the classic LRU worst case
            trace = [1 + i % k for i in range(length)]
        if len(set(trace)) > capacity:  # otherwise nothing is ever evicted
            cases.append({"trace": trace, "capacity": capacity})
    return cases


def _probe_all(task, codes, cases):
    """{key: outputs list} and {key: error} for {key: code}, one sandboxed probe per candidate."""
    evaluator = _evaluator(task)
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for i, (key, code) in enumerate(codes.items()):
            paths[key] = os.path.join(d, f"candidate_{i}.py")
            with open(paths[key], "w") as f:
                f.write(code)

        def one(key):
            return key, evaluator.probe(paths[key], cases)
        with ThreadPoolExecutor(8) as pool:
            results = dict(pool.map(one, paths))
    outputs = {k: r["detail"]["outputs"] for k, r in results.items() if r["valid"]}
    errors = {k: r["detail"].get("error", "invalid") for k, r in results.items() if not r["valid"]}
    return outputs, errors


def _behavior(task, output):
    return output["chosen"] if _name(task) == BIN else output["evictions"]


def _cost(task, output):
    return output["n_bins"] if _name(task) == BIN else output["misses"]


def _length(case):
    return len(case.get("items") or case.get("trace"))


def _cache_steps(case, evictions):
    """Replay a trace with a given eviction order: (1-based step of each eviction, hit steps)."""
    cache, queue, steps, hits = set(), list(evictions), [], set()
    for step, key in enumerate(case["trace"], 1):
        if key in cache:
            hits.add(step)
            continue
        if len(cache) >= case["capacity"]:
            cache.discard(queue.pop(0))
            steps.append(step)
        cache.add(key)
    return steps, hits


def explain(task, case, attempts_output, alternative_output, n_attempts):
    """One plain-text line on where the alternative first departs from the attempts."""
    who = {1: "the attempt", 2: "both attempts"}.get(n_attempts, f"all {n_attempts} attempts")
    a, b = attempts_output, alternative_output
    if _name(task) == BIN:
        i = next(j for j, (x, y) in enumerate(zip(a["chosen"], b["chosen"])) if x != y)

        def where(free):
            return "a new bin" if free == CAPACITY else f"a bin with {free} free"
        return (f"item {i + 1} (size {case['items'][i]}): {who} put it in {where(a['chosen'][i])}; the alternative "
                f"puts it in {where(b['chosen'][i])}. Bins used: attempts {a['n_bins']}, alternative {b['n_bins']}.")
    a_steps, a_hits = _cache_steps(case, a["evictions"])
    b_steps, b_hits = _cache_steps(case, b["evictions"])
    i = next(j for j, (x, y) in enumerate(zip(a["evictions"], b["evictions"])) if x != y)
    victim, step = a["evictions"][i], a_steps[i]
    text = f"{who} evict key {victim} at step {step}; the alternative evicts {b['evictions'][i]}"
    later = sorted(s for s in b_hits - a_hits if s > step)
    if later:
        text += f" and later hits key {case['trace'][later[0] - 1]} at step {later[0]}"
    return text + f". Misses: attempts {a['misses']}, alternative {b['misses']}."


def _counterexample(task, case_outputs, attempt_keys, alt_keys):
    """(attempts_output, alt_key, alt_output, better) for one case, or None."""
    first = case_outputs[attempt_keys[0]]
    if any(_behavior(task, case_outputs[k]) != _behavior(task, first) for k in attempt_keys[1:]):
        return None
    differing = [(k, case_outputs[k]) for k in alt_keys if _behavior(task, case_outputs[k]) != _behavior(task, first)]
    if not differing:
        return None
    key, out = min(differing, key=lambda ko: _cost(task, ko[1]))
    return first, key, out, _cost(task, out) < _cost(task, first)


def find(task, attempts, alternatives, n=300, seed=0, max_results=5, cases=None):
    """Counterexamples where all attempts agree and an alternative differs, alternative-better
    first, then shortest. attempts / alternatives: {id: code}."""
    cases = cases if cases is not None else generate_cases(_name(task), n, seed)
    codes = {**{("attempt", k): v for k, v in attempts.items()}, **{("alt", k): v for k, v in alternatives.items()}}
    outputs, errors = _probe_all(task, codes, cases)
    attempt_keys = [k for k in codes if k[0] == "attempt" and k in outputs]
    alt_keys = [k for k in codes if k[0] == "alt" and k in outputs]
    found, agreed = [], 0
    for i, case in enumerate(cases):
        if not attempt_keys:
            break
        per = {k: outputs[k][i] for k in attempt_keys + alt_keys}
        agreed += len({repr(_behavior(task, per[k])) for k in attempt_keys}) == 1
        hit = _counterexample(task, per, attempt_keys, alt_keys) if alt_keys else None
        if hit:
            first, key, out, better = hit
            found.append({"case": case, "attempts_output": first, "alternative_id": key[1],
                          "alternative_output": out, "alternative_better": better,
                          "explanation": explain(task, case, first, out, len(attempt_keys))})
    ranked = sorted(found, key=lambda r: (not r["alternative_better"], _length(r["case"])))
    return {"counterexamples": ranked[:max_results],
            "summary": {"cases": len(cases), "valid_attempts": len(attempt_keys),
                        "valid_alternatives": len(alt_keys), "attempts_agreed": agreed,
                        "alternative_differed": len(found),
                        "alternative_better": sum(r["alternative_better"] for r in found)},
            "invalid": {f"{kind}:{k}": err for (kind, k), err in errors.items()}}


def _smaller(task, case):
    """Cases with one element, or a run of 2 or 3, removed."""
    field = "items" if _name(task) == BIN else "trace"
    seq, out = case[field], []
    for width in (3, 2, 1):
        for i in range(len(seq) - width + 1):
            rest = seq[:i] + seq[i + width:]
            if rest and (field == "items" or len(set(rest)) > case["capacity"]):
                out.append({**case, field: rest})
    return out


def shrink(task, case, attempts, alternative, max_calls=30):
    """Greedily drop items / requests while all attempts still agree and the alternative still
    differs (and stays better, if it was). alternative: {id: code} with one entry. Each round
    probes every candidate once on all smaller cases; stops after about max_calls probes."""
    alt_id = next(iter(alternative))
    calls = calls_used = len(attempts) + 1
    start = find(task, attempts, alternative, cases=[case])
    if not start["counterexamples"]:
        return None
    best = start["counterexamples"][0]
    while calls_used + calls <= max_calls:
        options = _smaller(task, best["case"])
        if not options:
            break
        result = find(task, attempts, alternative, cases=options, max_results=len(options))
        calls_used += calls
        keep = [r for r in result["counterexamples"] if r["alternative_id"] == alt_id
                and (r["alternative_better"] or not best["alternative_better"])]
        if not keep:
            break
        best = min(keep, key=lambda r: _length(r["case"]))
    return {**best, "probe_calls": calls_used}
