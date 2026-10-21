"""Round 3 of the rescue test: does DIVERGE's chosen lever beat a random lever? (docs/PREREGISTRATION-lever.md)

    DIVERGE_TRACE_DIR=results/traces/lever uv run python scripts/lever_test.py --pilot   # map, overlook, 2 trajectories
    DIVERGE_TRACE_DIR=results/traces/lever uv run python scripts/lever_test.py

1. Consensus map and overlook study for online_bin_packing, built with MAP_MODELS (no Anthropic model):
   N_PER_MODEL samples from each of 5 labs, a non-Anthropic labeler. Written to results/online_bin_packing/.
2. Trajectories as in scripts/rescue_test.py (attempt 1 a fresh sample, attempts 2-6 revisions with dev-score
   feedback), implementer AGENT_MODEL, in batches of BATCH until ELIGIBLE_TARGET eligible stuck states exist or
   MAX_TRAJ trajectories were run. Eligible: stuck (rescue_test.stuck) and the stuck best's held-out score is
   no worse than the first-fit baseline's held-out score (states far above a trivial baseline, like round 2's
   148.12, are rescued by every arm and tell nothing).
3. Arms per eligible state, same implementer (AGENT_MODEL), K candidates and ARM_BUDGET each:
   chosen (escape v2, lever chosen by DIVERGE), random_lever (escape v2, a random input that at least half the
   consensus uses: round 2's control), fresh_start (rescue_test's prompt), patch (escape_patch on the best
   attempt). Escape arms run from the stuck state regardless of the diagnosis, as in round 2.
4. Held-out once per arm (dev-best candidate) and for the stuck best. Rescue = held-out better than the stuck
   best's. Primary: paired held-out scores, chosen vs random_lever (two-sided sign test).
Writes results/lever/{trajectories,arms,summary}.json."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import rescue_test as rt  # noqa: E402
from diverge.llm import BudgetExceeded  # noqa: E402
from diverge.pipeline import Session, ensure_map, read_json  # noqa: E402
from diverge.reality import Evaluator  # noqa: E402
from diverge.task import Task  # noqa: E402

TASK = "online_bin_packing"
MAP_MODELS = {"samplers": ["qwen/qwen3.8-max-0902", "openai/gpt-6.1-sol", "google/gemini-4-argon",
                           "meta-llama/llama-4-maverick", "deepseek/deepseek-v4-pro-0813"],
              "labeler": "openai/gpt-6.1-sol", "extractor": "meta-llama/llama-4-maverick",
              "implementer": "google/gemini-4-argon"}
N_PER_MODEL, MAP_BUDGET, OVERLOOK_BUDGET = 20, 6.0, 1.5
AGENT_MODEL = "deepseek/deepseek-v4.1-flash"
BATCH, ELIGIBLE_TARGET, MAX_TRAJ = 20, 24, 80
K, ARM_BUDGET = 3, 0.40
WORKERS = 8  # trajectories of a batch, and eligible states, run in parallel (each one stays sequential)
OUT = ROOT / "results" / "lever"
ARMS = ("chosen", "random_lever", "fresh_start", "patch")


def models(session, implementer=AGENT_MODEL):
    session.cfg["models"].update({**MAP_MODELS, "implementer": implementer})
    session.cfg["sampling"]["n_per_model"] = N_PER_MODEL
    return session


def build_map_and_overlook(task):
    from diverge import overlook
    out = ROOT / "results" / TASK
    if not (out / "herd_map.json").exists() or not (out / "blind_spots.json").exists():
        session = models(Session(budget_usd=MAP_BUDGET), MAP_MODELS["implementer"])
        ensure_map(session, task)
        print(f"map built: ${session.ledger.new_spend:.4f}", flush=True)
    if not (out / "overlook.json").exists():
        session = models(Session(budget_usd=OVERLOOK_BUDGET), MAP_MODELS["implementer"])
        overlook.build(session, task)
        print(f"overlook built: ${session.ledger.new_spend:.4f}", flush=True)
    return read_json(out / "overlook.json"), read_json(out / "blind_spots.json")


def escape_arm(name, state, task, evaluator, overlook, families, lever=None):
    from diverge.escape import escape_patch, escape_v2
    session = models(Session(budget_usd=ARM_BUDGET, progress=lambda m: None))
    attempts = [{"id": a["id"], "code": a["code"], "score": a["dev_score"], "valid": a["valid"],
                 "dev_detail": a["dev_detail"]} for a in state["attempts"]]
    seed = 3000 + state["run"]
    try:
        if name == "patch":
            res = escape_patch(session, task, attempts[-1], attempts[:-1], evaluator, overlook=overlook, k=K, seed=seed)
        else:
            res = escape_v2(session, task, attempts[-1], attempts[:-1], evaluator, overlook=overlook, families=families,
                            lever=lever, k_sketches=K, max_resample=2, seed=seed)
    except BudgetExceeded as e:
        return {"arm": name, "code": None, "dev_score": None, "error": f"budget: {e}",
                "cost_usd": round(session.ledger.new_spend, 4)}
    lv = res.get("lever") or {}
    return {"arm": name, "code": res.get("candidate"), "dev_score": (res.get("evaluation") or {}).get("after"),
            "lever": {k: lv.get(k) for k in ("type", "name")} if lv else None, "action": res.get("action"),
            "n_admitted": len(res.get("admitted") or []), "n_rejected": len(res.get("rejections") or []),
            "parent": res.get("parent"), "cost_usd": round(session.ledger.new_spend, 4)}


def sign_test(wins, losses):
    """Two-sided exact binomial p-value for wins vs losses (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    pilot = "--pilot" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    rt.OUT, rt.MODEL, rt.SUFFIX = OUT, AGENT_MODEL, ":r3"
    rt.with_model = lambda session: models(session)
    task = Task.resolve(TASK)
    evaluator = Evaluator(task)
    overlook, families = build_map_and_overlook(task)
    used = sorted(r["input"] for r in overlook["inputs"] if (r.get("share") or 0) >= 0.5)
    base_dev = {k: r["score"] for k, r in task.baselines("dev").items()}
    first_fit_held = task.baselines("heldout")["first_fit"]["score"]
    traj_path = OUT / "trajectories.json"
    trajs = read_json(traj_path, {}) or {}

    def eligible(t):
        if not t.get("stuck"):
            return False
        if "stuck_held" not in t:
            best = min((a for a in t["attempts"] if a["valid"]), key=lambda a: a["dev_score"])
            t["stuck_best"], t["stuck_held"] = best["id"], rt.heldout(evaluator, best["code"])
        return t["stuck_held"] is not None and t["stuck_held"] <= first_fit_held

    limit = 2 if pilot else MAX_TRAJ
    while len(trajs) < limit and sum(eligible(t) for t in trajs.values()) < (1 if pilot else ELIGIBLE_TARGET):
        batch = range(len(trajs), min(len(trajs) + (2 if pilot else BATCH), limit))
        with ThreadPoolExecutor(WORKERS) as pool:
            for r, t in zip(batch, pool.map(lambda r: rt.trajectory(r, task, evaluator, base_dev["best_fit"]), batch)):
                t["stuck"] = rt.stuck(t["attempts"])
                trajs[str(r)] = t
        traj_path.write_text(json.dumps(trajs, indent=2))
        n_el = sum(eligible(t) for t in trajs.values())
        traj_path.write_text(json.dumps(trajs, indent=2))
        print(f"{len(trajs)} trajectories, {sum(t['stuck'] for t in trajs.values())} stuck, {n_el} eligible", flush=True)
        if pilot:
            break
    if pilot:
        for t in trajs.values():
            print(t["run"], [a["dev_score"] and round(a["dev_score"], 3) for a in t["attempts"]], "stuck", t["stuck"],
                  "held", t.get("stuck_held"), "cost", t["cost_usd"])
        print("used inputs (random pool):", used, "| chosen lever would come from overlook + blind spots")
        return

    arms_path = OUT / "arms.json"
    arms = read_json(arms_path, {}) or {}
    todo = [(key, t) for key, t in sorted(trajs.items(), key=lambda kv: int(kv[0])) if eligible(t) and key not in arms]

    def run_state(item):
        key, t = item
        rng = random.Random(5000 + t["run"])
        results = [escape_arm("chosen", t, task, evaluator, overlook, families),
                   escape_arm("random_lever", t, task, evaluator, overlook, families, lever=rng.choice(used)),
                   rt.plain_arm("fresh_start", rt.FRESH.format(task=task.prompt, summary=rt.failure_summary(t["attempts"])),
                                t, task, evaluator),
                   escape_arm("patch", t, task, evaluator, overlook, families)]
        for a in results:
            a["heldout_score"] = rt.heldout(evaluator, a.get("code"))
            a["rescued"] = a["heldout_score"] is not None and a["heldout_score"] < t["stuck_held"]
            a.pop("candidates", None)
        print(f"state {key}: stuck held-out {t['stuck_held']:.3f}; " + "; ".join(
            f"{a['arm']} {a['heldout_score']} {'R' if a['rescued'] else '-'} ${a['cost_usd']}" for a in results), flush=True)
        return key, {"run": t["run"], "stuck_best": {"id": t["stuck_best"], "heldout_score": t["stuck_held"]},
                     "arms": results}

    with ThreadPoolExecutor(WORKERS) as pool:
        for key, rec in pool.map(run_state, todo):
            arms[key] = rec
            arms_path.write_text(json.dumps(arms, indent=2))

    def held(state, arm):
        return next(a["heldout_score"] for a in state["arms"] if a["arm"] == arm)
    INF = float("inf")
    wins = sum((held(s, "chosen") or INF) < (held(s, "random_lever") or INF) for s in arms.values())
    losses = sum((held(s, "random_lever") or INF) < (held(s, "chosen") or INF) for s in arms.values())
    summary = {"preliminary": len(arms) < ELIGIBLE_TARGET, "agent_model": AGENT_MODEL, "map_models": MAP_MODELS,
               "n_per_model": N_PER_MODEL, "n_trajectories": len(trajs),
               "n_stuck": sum(t["stuck"] for t in trajs.values()), "n_eligible": len(arms), "k_per_arm": K,
               "arm_budget_usd": ARM_BUDGET, "first_fit_heldout": first_fit_held,
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "per_arm": {n: {"rescued": sum(a["rescued"] for s in arms.values() for a in s["arms"] if a["arm"] == n),
                               "no_candidate": sum(a["heldout_score"] is None for s in arms.values()
                                                   for a in s["arms"] if a["arm"] == n),
                               "cost_usd": round(sum(a["cost_usd"] for s in arms.values() for a in s["arms"]
                                                     if a["arm"] == n), 4)} for n in ARMS},
               "chosen_vs_random": {"chosen_better": wins, "random_better": losses,
                                    "ties": len(arms) - wins - losses, "sign_test_p": round(sign_test(wins, losses), 4)},
               "trajectory_cost_usd": round(sum(t["cost_usd"] for t in trajs.values()), 4)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
