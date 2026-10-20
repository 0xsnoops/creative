"""Planted-answer positive control (docs/PREREGISTRATION-planted.md): can DIVERGE's escape surface a
mechanism that is known, by construction, to be the right one?

    DIVERGE_TRACE_DIR=results/traces/planted uv run python scripts/planted_control.py [--dry-run]

Task: predict a relay's delivery delay (ms) from size_kb, queue_len, hour, link_id, retries. Ground
truth: 18 + 1.5*queue_len + 0.1*size_kb + 25*retries, plus a fixed extra delay on three slow links
(link 7: +160, link 19: +130, link 33: +90), plus small noise; hour has no effect (a decoy). The task
prompt shows 40 example records, two from each slow link. The planted mechanism M: a per-link offset.

Stuck states: 5, each three human-written attempts (different code, one idea) that fit the numeric
features and ignore link_id, so every attempt is worst on links 7 and 19 and the scores are flat.
Arms per state, same implementer and the same per-arm cap:
  - diverge: diverge_escape's path (escape_for_agent, lever_policy "chosen": no consensus map exists,
    so the lever comes from inputs the attempts never use, else one model-proposed alternative);
  - random: the same with lever_policy "random";
  - fresh_start: the task prompt plus a failure summary (scores, worst links), no code, K candidates;
  - patch: escape_patch on the best attempt, K candidates.
Selection on dev; held-out scored once per arm. "Found M": held-out slow-link MAE at most 20% of the
stuck best's, and held-out MAE below the stuck best's. Models: no Anthropic model (MODELS below).
Writes results/planted/{data,arms.json,summary.json}."""
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from diverge.llm import BudgetExceeded, message_text  # noqa: E402
from diverge.pipeline import Session  # noqa: E402
from diverge.reality import Evaluator, dev_detail  # noqa: E402
from diverge.sample import extract_code  # noqa: E402
from diverge.task import Task  # noqa: E402

ROUND = int(sys.argv[sys.argv.index("--round") + 1]) if "--round" in sys.argv else 1
OUT = ROOT / "results" / ("planted" if ROUND == 1 else f"planted-r{ROUND}")  # data is identical across rounds
EVAL = ROOT / "scripts" / "planted" / "eval_link.py"
SLOW = {7: 160.0, 19: 130.0, 33: 90.0}
N_LINKS, K, ARM_BUDGET = 40, 3, 0.25  # 5 states x 4 arms x $0.25: a hard $5 bound
MODELS = {"implementer": "deepseek/deepseek-v4-pro-0813", "extractor": "meta-llama/llama-4-maverick",
          "labeler": "mistralai/mistral-large-4-0"}
FOUND_SLOW_RATIO = 0.20

TASK = """A message relay forwards messages over {n} network links. Write a Python function
`predict(record)` that returns the expected delivery delay in milliseconds for one message.
`record` is a dict with these fields:
- size_kb: message size in KB (1-64)
- queue_len: messages waiting in the relay's queue when it arrived (0-30)
- hour: hour of day, 0-23
- link_id: which link carried it, an integer 1-{n}
- retries: how many times it was retransmitted (0-2)

Measured examples (record -> delay in ms):
{examples}

`predict` is scored by mean absolute error on many more recorded messages; lower is better.
Use only the Python standard library.""".replace("{n}", str(N_LINKS))


def delay(r, rng):
    base = 18 + 1.5 * r["queue_len"] + 0.1 * r["size_kb"] + 25 * r["retries"] + SLOW.get(r["link_id"], 0.0)
    return round(base + rng.gauss(0, 2), 1)


def record(rng, link=None):
    return {"size_kb": rng.randint(1, 64), "queue_len": rng.randint(0, 30), "hour": rng.randint(0, 23),
            "link_id": link or rng.randint(1, N_LINKS), "retries": rng.choice([0, 0, 0, 1, 1, 2])}


def make_data():
    """(prompt, dev path, heldout path, train examples). Fixed seeds; written under OUT/data."""
    d = OUT / "data"
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261020)
    train = [record(rng, link) for link in sorted(SLOW) for _ in range(2)]
    train += [record(rng, rng.choice([k for k in range(1, N_LINKS + 1) if k not in SLOW])) for _ in range(34)]
    rng.shuffle(train)
    train = [{"input": r, "expected": delay(r, rng)} for r in train]
    for name, seed, n in (("dev", 1, 600), ("heldout", 2, 600)):
        rng2 = random.Random(seed)
        rows = [{"id": f"{name}-{i}", "input": (r := record(rng2)), "expected": delay(r, rng2)} for i in range(n)]
        (d / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    examples = "\n".join(f"{json.dumps(t['input'])} -> {t['expected']}" for t in train)
    return TASK.replace("{examples}", examples), d / "dev.jsonl", d / "heldout.jsonl", train


def fit(train, features):
    """Least squares of delay on the given numeric features (with intercept), standard library only."""
    xs = [[1.0] + [float(t["input"][f]) for f in features] for t in train]
    ys = [t["expected"] for t in train]
    n = len(xs[0])
    a = [[sum(x[i] * x[j] for x in xs) for j in range(n)] + [sum(x[i] * y for x, y in zip(xs, ys))] for i in range(n)]
    for c in range(n):  # Gauss-Jordan
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        a[c], a[p] = a[p], a[c]
        for r in range(n):
            if r != c:
                f = a[r][c] / a[c][c]
                a[r] = [u - f * v for u, v in zip(a[r], a[c])]
    return [a[i][n] / a[i][i] for i in range(n)]


def linear_variants(coef, features):
    """Three codings of one linear idea: plain, rounded constants, a loop over a weight table."""
    terms = " + ".join(f"{c:.4f} * record['{f}']" for c, f in zip(coef[1:], features))
    plain = f"def predict(record):\n    return {coef[0]:.4f} + {terms}\n"
    rounded = ("def predict(record):\n    return " + f"{round(coef[0], 1)} + " +
               " + ".join(f"{round(c, 2)} * record['{f}']" for c, f in zip(coef[1:], features)) + "\n")
    table = "{" + ", ".join(f"'{f}': {c:.4f}" for c, f in zip(coef[1:], features)) + "}"
    loop = (f"WEIGHTS = {table}\n\n\ndef predict(record):\n    total = {coef[0]:.4f}\n"
            "    for name, weight in WEIGHTS.items():\n        total += weight * record[name]\n    return total\n")
    return [plain, rounded, loop]


def stuck_states(train):
    out = []
    for i, feats in enumerate([["queue_len", "size_kb", "retries"], ["queue_len", "size_kb", "retries", "hour"],
                               ["queue_len", "retries"], ["queue_len", "size_kb", "retries"],
                               ["queue_len", "retries", "hour"]]):
        sub = train if i != 3 else [t for t in train if t["input"]["link_id"] not in SLOW] + train[:4]
        out.append({"state": i, "features": feats, "attempts": linear_variants(fit(sub, feats), feats)})
    return out


def heldout(code, path):
    if not code:
        return None, None
    f = OUT / "code" / f"final_{abs(hash(code)) % 10**12}.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(code)
    r = json.loads(subprocess.run([sys.executable, str(EVAL), "--data", str(path), str(f)], capture_output=True,
                                  text=True, timeout=300).stdout.strip().splitlines()[-1])
    if not r["valid"]:
        return None, None
    rows = [json.loads(line) for line in open(path)]
    by = {c["id"]: c["loss"] for c in r["detail"]["cases"]}
    counts = {}
    for row in rows:
        counts[row["input"]["link_id"]] = counts.get(row["input"]["link_id"], 0) + 1
    slow = sum(by[f"link-{k}"] * counts[k] for k in SLOW) / sum(counts[k] for k in SLOW)
    return r["score"], slow


def session_with_models(budget):
    s = Session(budget_usd=budget, progress=lambda m: None)
    s.cfg["models"].update(MODELS)
    return s


def summary_text(attempts, scores, worst):
    return ("Earlier attempts scored (mean absolute error in ms on the development records, lower is better): "
            + ", ".join(f"{x:.3f}" for x in scores) + f".\nEvery attempt did worst on the same links: {', '.join(worst)}."
            "\nThese attempts have stalled.")


def fresh_arm(task, state, evaluator, summary):
    session = session_with_models(ARM_BUDGET)
    prompt = f"{task.prompt}\n\n---\n{summary}\n\nWrite a solution. Reply with your complete solution in a single Python code block."
    cands = []
    for i in range(K):
        try:
            entry = session.llm.chat(MODELS["implementer"], [{"role": "user", "content": prompt}], temperature=1.0,
                                     max_tokens=8000, sample_idx=f"planted:fresh:{state}:{i}")
        except BudgetExceeded:
            break
        code = extract_code(message_text(entry), "predict")
        if code is None:
            cands.append({"code": None, "dev_score": None, "valid": False})
            continue
        path = OUT / "code" / f"fresh-{state}-{i}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        r = evaluator(path)
        cands.append({"code": code, "dev_score": r["score"], "valid": r["valid"]})
    valid = [c for c in cands if c["valid"]]
    best = min(valid, key=lambda c: c["dev_score"]) if valid else None
    return {"arm": "fresh_start", "code": best and best["code"], "dev_score": best and best["dev_score"],
            "cost_usd": round(session.ledger.new_spend, 4), "calls": len(cands)}


def escape_arm(name, task, state, evaluator, policy=None):
    from diverge.escape import escape_patch
    from diverge.server import escape_for_agent
    session = session_with_models(ARM_BUDGET)
    codes = state["attempts"]
    try:
        if name == "patch":
            r = escape_patch(session, task, codes[-1], codes[:-1], evaluator, k=K, seed=state["state"])
            lever = None
        else:
            r = escape_for_agent(session, task, codes[-1], codes[:-1], evaluator, k=K, counterexamples=False,
                                 lever_policy=policy)
            lever = r.get("overlooked_levers", [{}])[0] if r.get("overlooked_levers") else None
    except BudgetExceeded as e:
        return {"arm": name, "code": None, "dev_score": None, "error": f"budget: {e}",
                "cost_usd": round(session.ledger.new_spend, 4)}
    d = r.get("diagnosis") or {}
    return {"arm": name, "code": r.get("candidate"), "dev_score": (r.get("evaluation") or {}).get("after"),
            "action": r.get("action"), "detector": (d.get("evidence") or {}).get("detector"),
            "lever": lever and {k: lever.get(k) for k in ("type", "name", "mechanism", "why")},
            "escape": r.get("escape"), "parent": r.get("parent"), "cost_usd": round(session.ledger.new_spend, 4)}


def main():
    prompt, dev, held, train = make_data()
    task = Task("planted_relay_delay", prompt)
    task.entrypoint = "predict"
    evaluator = Evaluator(task, f"{sys.executable} {EVAL} --data {dev}")
    states = stuck_states(train)
    if "--dry-run" in sys.argv:
        for s in states:
            scores = [evaluator(_write(s, i, c))["score"] for i, c in enumerate(s["attempts"])]
            print(s["state"], s["features"], [round(x, 3) for x in scores], "heldout", heldout(s["attempts"][0], held))
        return
    from concurrent.futures import ThreadPoolExecutor
    arms_path, arms = OUT / "arms.json", {}

    def run_state(s):  # states run in parallel; arms within a state run one after another
        results = [evaluator(_write(s, i, c)) for i, c in enumerate(s["attempts"])]
        scores = [r["score"] for r in results]
        worst = sorted(results[0]["detail"]["cases"], key=lambda c: -c["loss"])[:2]
        summary = summary_text(s["attempts"], scores, [c["id"] for c in worst])
        best_code = s["attempts"][min(range(3), key=lambda i: scores[i])]
        stuck_held, stuck_slow = heldout(best_code, held)
        out = []
        for name, policy in (("diverge", "chosen"), ("random", "random"), ("fresh_start", None), ("patch", None)):
            a = fresh_arm(task, s["state"], evaluator, summary) if name == "fresh_start" else \
                escape_arm(name, task, s, evaluator, policy)
            a["heldout_score"], a["heldout_slow_mae"] = heldout(a.get("code"), held)
            a["found_m"] = (a["heldout_slow_mae"] is not None and a["heldout_score"] is not None
                            and a["heldout_slow_mae"] <= FOUND_SLOW_RATIO * stuck_slow
                            and a["heldout_score"] < stuck_held)
            out.append(a)
            print(f"state {s['state']} {name}: action={a.get('action')} lever={(a.get('lever') or {}).get('name')} "
                  f"held-out {a['heldout_score']} slow {a['heldout_slow_mae']} found={a['found_m']} "
                  f"${a.get('cost_usd')}", flush=True)
        return s, {"features": s["features"], "dev_scores": scores, "worst": [c["id"] for c in worst],
                   "stuck_best": {"heldout": stuck_held, "heldout_slow_mae": stuck_slow}, "arms": out}

    with ThreadPoolExecutor(len(states)) as pool:
        for s, rec in pool.map(run_state, states):
            arms[str(s["state"])] = rec
            arms_path.write_text(json.dumps(arms, indent=2))
    spent = sum(a.get("cost_usd") or 0.0 for r in arms.values() for a in r["arms"])
    names = ["diverge", "random", "fresh_start", "patch"]
    summary = {"preliminary": True, "models": MODELS, "k": K, "arm_budget_usd": ARM_BUDGET,
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_states": len(arms),
               "per_arm": {n: {"found_m": sum(bool(a.get("found_m")) for s in arms.values() for a in s["arms"]
                                              if a["arm"] == n),
                               "cost_usd": round(sum(a.get("cost_usd") or 0 for s in arms.values() for a in s["arms"]
                                                     if a["arm"] == n), 4)} for n in names},
               "total_cost_usd": round(spent, 4)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _write(state, i, code):
    path = OUT / "code" / f"stuck-{state['state']}-{i}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return path


if __name__ == "__main__":
    main()
