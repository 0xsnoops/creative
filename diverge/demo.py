"""Deterministic demo and preflight check.

`python -m diverge demo <name>` replays a recorded run: fixtures/trajectories/<name>.json (the agent's
attempts) and fixtures/runs/<name>.json (what escape and a fresh-start baseline produced on them,
recorded once with real API calls). The demo itself makes no API or network calls, never builds a
pipeline or a client, adds no timestamps and uses no randomness, so the same files print the same text.

`python -m diverge preflight` checks budget, API keys and model slugs before a live run (`--json`
prints readiness(), the same dict the diverge_status MCP tool returns)."""
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIN_REMAINING_USD = 1.0


def _load(path):
    import json
    return json.loads(Path(path).read_text())


def _fmt(v, digits=4):
    return "-" if v is None else f"{v:.{digits}f}"


def _pct(v):
    return "-" if v is None else f"{v:.0%}"


def code_summary(code, width=72):
    """One line: size and the first return expression (joined across lines, whitespace collapsed),
    else the first def line; just the size when neither says anything."""
    raw = [l.strip() for l in (code or "").splitlines()]
    lines = [l for l in raw if l and not l.startswith("#")]
    if not lines:
        return "(no code)"
    key = ""
    start = next((i for i, l in enumerate(lines) if l.startswith("return ")), None)
    if start is not None:
        parts, depth = [], 0
        for l in lines[start:start + 12]:
            parts.append(l)
            depth += sum(l.count(c) for c in "([{") - sum(l.count(c) for c in ")]}")
            if depth <= 0:
                break
        key = re.sub(r"\s+", " ", " ".join(parts))
        key = re.sub(r"([(\[{]) ", r"\1", re.sub(r" ([)\]}])", r"\1", key))
    if not key or key in ("return", "return None"):
        key = next((l for l in lines if l.startswith("def ")), "")
    if not key:
        return f"{len(lines)} lines"
    key = key if len(key) <= width else key[:width - 3].rstrip() + "..."
    return f"{len(lines)} lines; {key}"


def _first_line(text, empty="(no feedback)"):
    text = (text or "").strip()
    return text.splitlines()[0] if text else empty


def comparison(escape_score, fresh_score):
    """Honest one-line comparison on held-out (lower is better)."""
    if escape_score is None or fresh_score is None:
        missing = [n for n, s in (("escape", escape_score), ("fresh_start", fresh_score)) if s is None]
        return f"cannot compare on held-out: no held-out score for {' and '.join(missing)}"
    diff = round(fresh_score - escape_score, 4)
    tail = f"({_fmt(escape_score)} vs {_fmt(fresh_score)} % excess bins; lower is better)"
    if diff == 0:
        return f"escape tied fresh_start on held-out {tail}"
    word = "beat" if diff > 0 else "lost to"
    return f"escape {word} fresh_start on held-out by {abs(diff):.4f} points {tail}"


def _missing_run_help(name, root):
    scripts = sorted(p.name for p in (Path(root) / "scripts").glob("*.py")
                     if any(w in p.name for w in ("fixture", "demo", "record")))
    how = (f"run scripts/{scripts[0]} for {name}" if scripts else
           "run the escape tool and the fresh-start baseline on the fixture once (this is the only "
           "step that calls the API) and save the result there")
    return (f"no recorded run for {name!r}: fixtures/runs/{name}.json is missing.\n"
            f"To create it, {how}. Needs OPENROUTER_API_KEY and budget; check with `python -m diverge preflight`.")


_OLD_NAMES = ((re.compile(r"\bherd_(escape|check|status|map)\b(?!\.json)"), r"diverge_\1"),
              (re.compile(r"\bHERD\b"), "DIVERGE"), (re.compile(r"\bHerd\b"), "Consensus"),
              (re.compile(r"\bherd(?=[ _]|\b)(?!_map\.json)"), "consensus"))


# Lever mechanisms wrap an input description that may itself end in a parenthesis:
# "n_open_bins (how many ... offered (len(capacities)))" reads as "(how many ... offered: len(capacities))".
_NESTED_PAREN = re.compile(r"\(([^()]*?) \(([^()]*\([^()]*\))\)\)")


def current_wording(text):
    """Recorded fixtures predate the product rename to DIVERGE: show their text in current wording
    (product DIVERGE, the concept "consensus"). Display only; fixture files are left as recorded."""
    for pattern, new in _OLD_NAMES:
        text = pattern.sub(new, text)
    return _NESTED_PAREN.sub(r"(\1: \2)", text)


def run(name, root=REPO, out=print):
    """Print the recorded demo for fixture `name`. Returns 0, or 1 when a file is missing."""
    raw_out = out
    out = lambda text="": raw_out(current_wording(str(text)))  # noqa: E731
    root = Path(root)
    traj_path, run_path = root / "fixtures" / "trajectories" / f"{name}.json", root / "fixtures" / "runs" / f"{name}.json"
    if not traj_path.exists():
        out(f"no fixture {name!r}: {traj_path.relative_to(root)} is missing")
        return 1
    if not run_path.exists():
        out(_missing_run_help(name, root))
        return 1
    traj, rec = _load(traj_path), _load(run_path)
    esc, fresh = rec.get("escape") or {}, rec.get("fresh_start") or {}
    result = esc.get("result") or {}

    out(f"DIVERGE demo: {name}  (task: {rec.get('task') or traj.get('task')})")
    out(f"recorded run: commit {rec.get('commit')}, {rec.get('n_runs')} run(s), "
        f"budget ${rec.get('budget_usd_each', 0):.2f} each; replayed offline, no API calls")
    if traj.get("how_generated"):
        out(f"attempts: {traj['how_generated']}")

    out("\n== 1. The agent's attempts (dev score, lower is better) ==")
    attempts = traj.get("attempts", [])
    for i, a in enumerate(attempts):
        out(f"  {a.get('id')}  dev {_fmt(a.get('dev_score'))}  [{a.get('source')}]")
        last = i == len(attempts) - 1
        out(f"      feedback: {_first_line(a.get('feedback'), '(current attempt)' if last else '(no feedback)')}")
        out(f"      code: {code_summary(a.get('code'))}")

    out("\n== 2. Loop diagnosis ==")
    out(f"  strategy similarity {_fmt2(result.get('strategy_similarity'))} vs code similarity "
        f"{_fmt2(result.get('code_similarity'))}")
    out(f"  loop detected: {result.get('loop_detected')}   action: {result.get('action')}")
    if result.get("shared_strategy"):
        out(f"  shared strategy: {result['shared_strategy']}")
    for reason in result.get("reasons") or []:
        out(f"  - {reason}")

    out("\n== 3. Consensus evidence ==")
    excerpt = rec.get("overlook_excerpt") or {}
    for r in excerpt.get("inputs") or []:
        flag = "  [overlooked]" if r.get("overlooked") else ""
        out(f"  {r.get('input')}: used by {r.get('used_by')}/{r.get('decided')} consensus samples "
            f"({_pct(r.get('share'))}){flag}")
    assumptions = excerpt.get("assumptions") or []
    if isinstance(assumptions, dict):
        assumptions = assumptions.get("assumptions") or []
    for a in assumptions:
        if isinstance(a, dict):
            held = f", held by {a['held_by']}/{a['checked']}" if a.get("held_by") is not None else ""
            out(f"  assumption: {a.get('assumption')} ({a.get('kind')}{held})")
        else:
            out(f"  assumption: {a}")
    out(f"  consensus prevalence of the attempts' strategy profile: {result.get('herd_prevalence')}")
    for lever in result.get("overlooked_levers") or []:
        out(f"  lever [{lever.get('type')}] {lever.get('name')}: {lever.get('why')}")

    out("\n== 4. Scope of the consensus claims ==")
    para = rec.get("paraphrase")
    if para:
        out(f"  paraphrase test: {para.get('verdict')} (mean paraphrase TVD {para.get('mean_P')} vs "
            f"max resampling TVD {para.get('max_R')}); read consensus shares as 'what models converge on "
            f"for this wording'")
    else:
        out("  no paraphrase test recorded; consensus shares hold for this exact prompt wording only")

    text = esc.get("contract")
    if not text:
        try:
            from . import contract
            text = contract.render(result)
        except ImportError:
            text = "(diverge.contract is not available and the run file has no recorded contract)"
    verdict = text.splitlines()[0].replace("VERDICT:", "").strip() if text.startswith("VERDICT:") else None
    switch = result.get("recommended_switch") or {}
    if verdict == "escape":
        out("\n== 5. Escape lever ==")
        out(f"  switch to: {switch.get('mechanism')}")
    else:
        out(f"\n== 5. Lever (not applied: verdict {verdict}) ==")
        out(f"  option if progress stalls: {switch.get('mechanism')}")
    out(f"  why: {switch.get('why')}")

    out("\n== 6. Contract (what the agent receives) ==")
    out(text)

    out("\n== 7. Results (% excess bins, lower is better) ==")
    base = rec.get("baselines") or {}
    best = rec.get("best_attempt") or {}
    no_candidate = esc.get("dev_score") is None and esc.get("heldout_score") is None and verdict != "escape"
    rows = [(f"best attempt ({best.get('id')})", best.get("dev_score"), best.get("heldout_score"), ""),
            ("escape candidate", None, None, f"not generated (verdict {verdict})") if no_candidate else
            ("escape candidate", esc.get("dev_score"), esc.get("heldout_score"),
             f"${esc.get('cost_usd', 0):.4f} / {esc.get('calls') or 0} calls"),
            ("fresh_start candidate", fresh.get("dev_score"), fresh.get("heldout_score"),
             f"${fresh.get('cost_usd', 0):.4f} / {fresh.get('calls') or 0} calls")]
    forced = rec.get("forced_escape")
    if forced:
        rows.append((f"forced escape ({forced.get('lever')})*", forced.get("dev_score"), forced.get("heldout_score"),
                     f"${forced.get('cost_usd', 0):.4f} / {forced.get('calls') or 0} calls"))
    for b in ("best_fit", "first_fit"):
        rows.append((b, (base.get("dev") or {}).get(b), (base.get("heldout") or {}).get(b), ""))
    width = max(24, *(len(r[0]) for r in rows))
    out(f"  {'':{width}} {'dev':>9} {'held-out':>9}  cost")
    for label, dev, held, cost in rows:
        if cost.startswith("not generated"):
            out(f"  {label:{width}} {cost}")
        else:
            out(f"  {label:{width}} {_fmt(dev):>9} {_fmt(held):>9}  {cost}".rstrip())

    out(f"\nPRELIMINARY: {rec.get('n_runs') or 1} run" + ("s" if (rec.get('n_runs') or 1) != 1 else ""))
    if verdict is not None and verdict != "escape":
        out(f"no escape candidate: verdict {verdict}, so DIVERGE did not switch strategy and there is nothing to "
            f"compare with fresh_start")
    else:
        out(comparison(esc.get("heldout_score"), fresh.get("heldout_score")))
    if forced:
        out(f"* {forced.get('note')}. Forced escape vs fresh_start: "
            + comparison(forced.get("heldout_score"), fresh.get("heldout_score")).replace("escape", "forced escape", 1))
    return 0


def _fmt2(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)


def _configured_models(cfg):
    m = cfg["models"]
    return list(dict.fromkeys([*m["samplers"], *(m[k] for k in ("labeler", "implementer", "extractor") if m.get(k))]))


def _live_model_ids(timeout=10):
    """Live OpenRouter /models ids, or None when the network or the endpoint fails."""
    try:
        import httpx
        r = httpx.get("https://openrouter.ai/api/v1/models", timeout=timeout)
        r.raise_for_status()
        return {m["id"] for m in r.json()["data"]}
    except Exception:
        return None


OFFLINE_DEMO = "strategy_loop"
OFFLINE_DEMO_COMMAND = f"python -m diverge demo {OFFLINE_DEMO}"
PRIMARY_TOOL = "diverge_escape"
WHEN_TO_CALL = ("diverge_escape when you keep getting the same answer: several scored attempts have stalled, the same "
                "failure keeps coming back, code changes a lot but the score barely moves, or you are about to retry "
                "another variation or escalate; diverge_check before committing to a strategy, to see whether it is "
                "the default answer most models give and get alternatives (free-text mode needs no evaluator). Not for "
                "generic debugging.")
KEYS = (("openrouter", "OPENROUTER_API_KEY", True, "every model call (fingerprinting, implementing, map builds)"),
        ("exa", "EXA_API_KEY", False, "literature search, only when a consensus map is built for a new task"))


def _status(session, env):
    """(cfg, spent, cap) without a network call: a given Session, or config plus cache totals."""
    from . import pipeline
    if session is not None:
        return session.cfg, session.ledger.total_spent, session.ledger.global_cap
    if env is None:
        pipeline.load_env()  # same .env the tools read; values are only checked for presence
    import yaml
    cfg = yaml.safe_load((pipeline.ROOT / "config.yaml").read_text())
    cache = pipeline.ROOT / cfg["paths"]["cache"]
    spent = pipeline.shared_cache(cache / "llm.jsonl").total_cost() + pipeline.shared_cache(cache / "exa.jsonl").total_cost()
    return cfg, spent, cfg["budget_usd"]


def _cached_tasks(cfg):
    from . import pipeline
    from .task import TASKS_DIR
    results = pipeline.ROOT / cfg.get("paths", {}).get("results", "results")
    return sorted(d.name for d in TASKS_DIR.iterdir()
                  if (d / "evaluator.py").exists() and (results / d.name / "herd_map.json").exists())


def _catalog_check(models, ids):
    missing = [m for m in models if m not in ids]
    return {"present": len(models) - len(missing), "configured": len(models), "missing": missing}


def readiness(session=None, env=None, live=False, root=REPO):
    """Can DIVERGE's live tools run here? A dict an agent can act on. Zero cost: no model or search call,
    and no network call unless live=True (then one unauthenticated GET of OpenRouter's public model
    list, reported under "model_catalog" and optional). Reports whether each key is configured, never
    its value."""
    from . import pipeline
    cfg, spent, cap = _status(session, env)
    env = os.environ if env is None else env
    remaining = cap - spent
    requirements, missing = {}, []
    keys = KEYS
    if str(env.get("DIVERGE_PROVIDER", "")).lower() == "anthropic":  # OpenRouter disabled
        keys = (("anthropic", "ANTHROPIC_API_KEY", True, "every model call, sent to the Anthropic API"),) + KEYS[1:]
    for name, key, required, used_for in keys:
        present = bool(env.get(key))
        requirements[name] = {"env": key, "status": "configured" if present else
                              ("missing" if required else "missing (optional)"),
                              "required": required, "used_for": used_for}
        if required and not present:
            missing.append(key)
    if remaining < MIN_REMAINING_USD:
        missing.append(f"budget: remaining ${remaining:.2f} is under ${MIN_REMAINING_USD:.2f}")
    live_ok = not missing
    demo_ok = (Path(root) / "fixtures" / "runs" / f"{OFFLINE_DEMO}.json").exists() and \
        (Path(root) / "fixtures" / "trajectories" / f"{OFFLINE_DEMO}.json").exists()
    models = _configured_models(cfg)
    catalog_path = pipeline.ROOT / cfg["paths"]["cache"] / "openrouter_models.json"
    cached_catalog = (_catalog_check(models, {m["id"] for m in _load(catalog_path)["data"]})
                      if catalog_path.exists() else {"error": f"{cfg['paths']['cache']}/openrouter_models.json not found"})
    if live_ok:
        action = (f"DIVERGE is configured. Call {PRIMARY_TOOL} yourself when attempts stall (see when_to_call); "
                  f"diverge_check before committing to a strategy, to get alternatives to the default answer.")
    elif any(k in missing for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")):
        needed = "ANTHROPIC_API_KEY" if "ANTHROPIC_API_KEY" in missing else "OPENROUTER_API_KEY"
        action = (f"Live DIVERGE calls need {needed}, which only the user can provide (in .env); ask them, and "
                  "never invent or guess a credential. Until then, " +
                  (f"run the offline demo ({OFFLINE_DEMO_COMMAND}, no API calls)." if demo_ok else
                   "continue without DIVERGE."))
    else:
        action = ("The spend cap is nearly reached; ask the user whether to raise budget_usd in config.yaml before "
                  "any live DIVERGE call." + (f" The offline demo still works: {OFFLINE_DEMO_COMMAND}." if demo_ok else ""))
    out = {"ready": live_ok, "live_mode_available": live_ok, "remaining_budget_usd": round(remaining, 4),
           "budget": {"spent_usd": round(spent, 4), "cap_usd": cap}, "requirements": requirements,
           "missing": missing, "offline_demo_available": demo_ok,
           "offline_demo_command": OFFLINE_DEMO_COMMAND if demo_ok else None,
           "cached_tasks": _cached_tasks(cfg), "cached_catalog": cached_catalog,
           "recommended_primary_tool": PRIMARY_TOOL, "when_to_call": WHEN_TO_CALL, "next_action": action}
    if live:
        ids = _live_model_ids()
        out["model_catalog"] = {"optional": True, "status": "unavailable (network or endpoint failed)"}
        if ids is not None:
            out["model_catalog"] = {"optional": True, "status": "ok", **_catalog_check(models, ids)}
    return out


def preflight(session=None, out=print, live=True, env=None):
    """Budget, keys and model slugs before a live run, printed from readiness(). Returns 1 when
    remaining budget is under $1 or OPENROUTER_API_KEY is missing, else 0. Never prints key values."""
    r = readiness(session, env, live)
    b = r["budget"]
    out(f"spend ${b['spent_usd']:.4f} of cap ${b['cap_usd']:.2f}; remaining ${r['remaining_budget_usd']:.4f}")
    if r["remaining_budget_usd"] < MIN_REMAINING_USD:
        out(f"FAIL remaining budget is under ${MIN_REMAINING_USD:.2f}")
    for req in r["requirements"].values():
        configured = req["status"] == "configured"
        out(f"{req['env']}: {'set' if configured else 'NOT set'}")
        if req["required"] and not configured:
            out(f"FAIL {req['env']} is required")
    for label, c in (("cached catalog", r["cached_catalog"]), ("live catalog", r.get("model_catalog"))):
        if c is None:
            continue
        if "present" not in c:
            out(f"{label}: " + (f"WARN {c['error']}" if "error" in c else f"{c['status']}; skipped"))
        else:
            out(f"{label}: {c['present']}/{c['configured']} configured models present"
                + (f"; WARN missing {c['missing']}" if c["missing"] else ""))
    if not r["ready"]:
        out(f"next: {r['next_action']}")
    out("preflight OK" if r["ready"] else "preflight FAILED")
    return 0 if r["ready"] else 1
