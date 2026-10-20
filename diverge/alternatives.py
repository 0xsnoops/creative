"""What the agent tells DIVERGE about its own situation, and what DIVERGE does with it.

- already_tried [{approach, outcome, score}]: never offered again, and their outcomes inform the
  alternatives (each says which failure it builds_on).
- constraints [str]: hard rules every alternative must satisfy.
- context (str): the agent's real figures (log tail, results table); alternatives cite them, and a
  deterministic check flags cited numbers that do not appear in it.
- the outcome log (diverge_report_outcome): reported outcomes join already_tried on later calls.

One screening call (a model that did not write the alternatives) drops duplicates of already-tried
approaches and constraint violators; both are returned in filtered_out, never dropped silently.
Alternatives are ranked by coarse test cost (decisive and cheap first); nothing here is MEASURED."""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .llm import BudgetExceeded, ask_json
from .reality import sandbox

CONTEXT_LIMIT = 8000
TEST_COSTS = ("minutes", "hours", "days")
OUTCOMES = ("LEGIT", "WEAK", "WRONG")
DEPTHS = ("quick", "deep")
DEEP_SAMPLES = 3
DEEP_TEMPERATURES = (0.7, 1.0, 1.2)
CRITIC_AXES = ("novelty", "constraint_fit", "testability")
ALT_FIELDS = ("name", "mechanism", "strength", "trap", "how_to_test")
EXTENDED_FIELDS = ("builds_on", "grounding", "test_cost", "kill_result")

EXTENDED_PROMPT = """You are proposing alternative strategies for a programming task, before any code is written.

Task:
{task}

The agent's planned approach:
{approach}
{herd}{tried}{constraints}{context}
Propose {n} genuinely different strategies (different mechanisms, not tuned variants of the planned approach \
or of anything already tried). For each give: name (a short family name), mechanism (one sentence: how it \
decides), strength (when it should beat the planned approach), trap (the most likely way it fails or misleads), \
how_to_test (one concrete measurement that would confirm or refute it, against the planned approach), builds_on \
(the already-tried approach whose outcome it responds to, or ""), grounding (the specific figures from the \
context it relies on, quoted, or ""), test_cost ("minutes", "hours" or "days": the effort of one decisive test), \
decisive (true if one test would clearly confirm or kill it), kill_result (the observation that would prove it \
wrong). Also give relax_hint: one sentence naming the single hard rule whose relaxation would open the most \
different strategy, or "" if none would. Never propose a strategy that breaks a hard rule.
Return JSON only: {{"alternatives": [{{"name": "...", "mechanism": "...", "strength": "...", "trap": "...", \
"how_to_test": "...", "builds_on": "", "grounding": "", "test_cost": "hours", "decisive": true, \
"kill_result": "..."}}], "relax_hint": ""}}"""

SCREEN_MARKER = "Screen proposed strategies"
SCREEN_PROMPT = SCREEN_MARKER + """ for the task below against what the agent already tried and its hard rules. \
You review them; you did not write them.

Task:
{task}

Already tried:
{tried}

Hard rules:
{constraints}

Proposed strategies:
{proposals}

For each proposal give "duplicate_of": the index of the already-tried approach it is the same mechanism as (same \
decision signal and rule, possibly re-weighted, re-tuned or renamed), or null; "violates": the index of a hard \
rule it breaks, or null; "reason": one sentence. Reply with only a JSON object:
{{"screen": [{{"index": 0, "duplicate_of": null, "violates": null, "reason": "..."}}]}}"""

CRITIC_MARKER = "Critique proposed strategies"
CRITIC_PROMPT = CRITIC_MARKER + """ for the task below. You are a reviewer, not their author.

Task:
{task}

The agent's planned approach: {approach}
{herd}{tried}{constraints}
Proposed strategies (from separate authors):
{proposals}

For each proposal give integer scores 0-10: "novelty" (how different its mechanism is from the planned approach, \
the consensus and everything already tried), "constraint_fit" (how well it respects the hard rules), \
"testability" (how cheaply one test would confirm or kill it). Set "same_as" to the index of an EARLIER proposal \
with the same mechanism (same decision signal and rule, possibly renamed or re-tuned), else null. Reply with only \
a JSON object:
{{"scores": [{{"index": 0, "novelty": 0, "constraint_fit": 0, "testability": 0, "same_as": null}}]}}"""


# ------------------------------------------------------------------ agent inputs
def redact(text):
    return sandbox.redact(str(text or ""))


def clean_context(context):
    """(redacted context capped at CONTEXT_LIMIT chars, truncated?)."""
    text = redact(context).strip()
    if len(text) <= CONTEXT_LIMIT:
        return text, False
    return text[:CONTEXT_LIMIT] + f"\n[... truncated by DIVERGE to {CONTEXT_LIMIT} characters]", True


def _score(x):
    try:
        return None if x is None or isinstance(x, bool) else float(x)
    except (TypeError, ValueError):
        return None


def clean_tried(already_tried):
    """[{approach, outcome, score, source}] from strings or dicts; redacted, empty approaches dropped."""
    out = []
    for t in already_tried or []:
        t = {"approach": t} if isinstance(t, str) else t if isinstance(t, dict) else {}
        approach = " ".join(redact(t.get("approach")).split())
        if approach:
            out.append({"approach": approach, "outcome": " ".join(redact(t.get("outcome")).split()),
                        "score": _score(t.get("score")), "source": t.get("source") or "agent"})
    return out


def clean_constraints(constraints):
    return [c for c in (" ".join(redact(c).split()) for c in constraints or []) if c]


def tried_line(t):
    score = f" (score {t['score']:.6g})" if t["score"] is not None else ""
    return f"{t['approach']}: {t['outcome'] or 'no outcome given'}{score}"


def prompt_blocks(tried, constraints, context):
    """The three optional prompt blocks; each is empty when its input is."""
    return {
        "tried": ("\nAlready tried, with outcomes (do not propose these mechanisms again; use what each outcome "
                  "shows):\n" + "\n".join(f"- {tried_line(t)}" for t in tried) + "\n") if tried else "",
        "constraints": ("\nHard rules (every strategy must satisfy all of them):\n"
                        + "\n".join(f"- {c}" for c in constraints) + "\n") if constraints else "",
        "context": ("\nContext from the agent's own work (real figures; cite the ones you rely on):\n<<<\n"
                    + context + "\n>>>\n") if context else ""}


def escape_notes(tried, constraints, context):
    """The agent's inputs as one block for diverge_escape's prompts (empty when there are none)."""
    blocks = prompt_blocks(tried, constraints, context)
    text = "".join(blocks[k] for k in ("tried", "constraints", "context")).strip()
    return "Notes from the agent:\n" + text if text else ""


# ------------------------------------------------------------------ outcome log
def outcomes_path(task):
    """results/<task>/outcomes.jsonl, found without a Session (which may fetch the model catalog)."""
    import yaml

    from . import pipeline
    cfg = yaml.safe_load((pipeline.ROOT / "config.yaml").read_text())
    return pipeline.ROOT / cfg["paths"]["results"] / task.name / "outcomes.jsonl"


def record_outcome(task, alternative, outcome, note="", score=None):
    outcome = str(outcome or "").strip().upper()
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {list(OUTCOMES)}, got {outcome!r}")
    alternative = " ".join(redact(alternative).split())
    if not alternative:
        raise ValueError("alternative is empty")
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "alternative": alternative, "outcome": outcome,
           "note": " ".join(redact(note).split()), "score": _score(score)}
    path = outcomes_path(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def logged_outcomes(task):
    path = outcomes_path(task)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def logged_as_tried(rows):
    return [{"approach": r["alternative"], "outcome": r["outcome"] + (f": {r['note']}" if r.get("note") else ""),
             "score": r.get("score"), "source": "logged"} for r in rows if r.get("alternative")]


def merge_tried(tried, logged):
    """The agent's already_tried plus logged outcomes it did not repeat (same approach text)."""
    seen = {t["approach"].lower() for t in tried}
    return tried + [t for t in logged if t["approach"].lower() not in seen]


# ------------------------------------------------------------------ alternatives
def normalize(a, extended):
    out = {k: " ".join(str(a.get(k) or "").split()) for k in ALT_FIELDS}
    if extended:
        out |= {k: " ".join(str(a.get(k) or "").split()) for k in EXTENDED_FIELDS}
        out["test_cost"] = out["test_cost"].lower() if out["test_cost"].lower() in TEST_COSTS else "unknown"
        d = a.get("decisive")
        out["decisive"] = d if isinstance(d, bool) else str(d).strip().lower() == "true"
    return out | {"evidence_class": "HYPOTHESIZED"}


NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")  # unsigned: "22-61 ms" is 22 and 61


def numbers(text):
    out = set()
    for m in NUMBER.findall(str(text or "")):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def ungrounded_numbers(alt, context):
    """Numbers an alternative states (in its grounding, mechanism, strength or kill_result) that do not
    appear in the context. Single-digit integers are ignored: they are mostly counts ("top 2"), not
    claims about the agent's data. Deterministic, no model call; a flag, never a reason to drop."""
    have = numbers(context)
    said = set().union(*(numbers(alt.get(k)) for k in ("grounding", "mechanism", "strength", "kill_result")))
    return sorted(x for x in said if x not in have and not (x.is_integer() and 0 <= x <= 9))


def rank(alts):
    """Decisive before non-decisive, then cheaper test first; the model's order otherwise (stable)."""
    cost = {c: i for i, c in enumerate(TEST_COSTS)}
    return sorted(alts, key=lambda a: (not a.get("decisive"), cost.get(a.get("test_cost"), len(TEST_COSTS))))


def _index(x, n):
    return x if isinstance(x, int) and not isinstance(x, bool) and 0 <= x < n else None


def screen(llm, model, task, alts, tried, constraints):
    """(kept, filtered_out) from one batched call by a model that did not write the alternatives."""
    proposals = "\n".join(f"[{i}] {a['name']}: {a['mechanism']}" for i, a in enumerate(alts))
    reply = ask_json(llm, model, SCREEN_PROMPT.format(
        task=task.prompt, tried="\n".join(f"[{i}] {tried_line(t)}" for i, t in enumerate(tried)) or "(none)",
        constraints="\n".join(f"[{i}] {c}" for i, c in enumerate(constraints)) or "(none)", proposals=proposals),
        max_tokens=8000, required=("screen",))  # room for screen models that reason first
    verdicts = {r.get("index"): r for r in reply.get("screen") or [] if isinstance(r, dict)}
    kept, dropped = [], []
    for i, a in enumerate(alts):
        r = verdicts.get(i) or {}
        dup, bad = _index(r.get("duplicate_of"), len(tried)), _index(r.get("violates"), len(constraints))
        if dup is not None:
            dropped.append({"name": a["name"], "reason": "duplicate_of", "of": tried[dup]["approach"],
                            "why": str(r.get("reason") or "")})
        elif bad is not None:
            dropped.append({"name": a["name"], "reason": "violates", "of": constraints[bad],
                            "why": str(r.get("reason") or "")})
        else:
            kept.append(a)
    return kept, dropped


def company(model):
    return model.split("/", 1)[0]


def deep_models(cfg):
    """Up to DEEP_SAMPLES configured sampler models, one per company; in anthropic-only mode, the Claude
    models among the samplers, the implementer and the extractor."""
    from .llm import provider
    models = cfg["models"]
    pool = list(models["samplers"])
    if provider() == "anthropic":
        pool = [m for m in [*pool, models.get("implementer"), models.get("extractor")]
                if m and m.startswith("anthropic/")]
    chosen, seen = [], set()
    for m in pool:
        key = company(m) if provider() != "anthropic" else m
        if key not in seen:
            seen.add(key)
            chosen.append(m)
    return chosen[:DEEP_SAMPLES]


def critic_model(cfg, authors):
    models = cfg["models"]
    for m in (models.get("labeler"), models.get("extractor"), models.get("implementer"), *models["samplers"]):
        if m and m not in authors:
            return m
    return models.get("labeler") or models.get("extractor")


def _critic_total(s):
    def v(k):
        try:
            return max(0.0, min(10.0, float(s.get(k))))
        except (TypeError, ValueError):
            return 0.0
    return {k: v(k) for k in CRITIC_AXES}


def deep_alternatives(session, task, prompt, critic_inputs, n):
    """(alternatives with critic scores, relax_hints, meta). DEEP_SAMPLES isolated samples from models of
    different companies at spread temperatures, then one critic that wrote none of them scores novelty,
    constraint fit and testability and marks same-mechanism repeats, which are dropped."""
    authors = deep_models(session.cfg)
    errors, replies = [], []

    def one(i, model):
        return ask_json(session.llm, model, prompt, max_tokens=8000, key=f"deep:{i}",
                        temperature=DEEP_TEMPERATURES[i % len(DEEP_TEMPERATURES)], required=("alternatives",))
    with ThreadPoolExecutor(len(authors) or 1) as pool:
        futures = [(m, pool.submit(one, i, m)) for i, m in enumerate(authors)]
        for m, f in futures:
            try:
                replies.append((m, f.result()))
            except BudgetExceeded:
                raise
            except Exception as e:
                errors.append({"model": m, "error": redact(str(e))[:300]})
    if not replies:
        raise ValueError("every deep-mode sample failed: " + "; ".join(e["error"] for e in errors))
    alts, hints = [], []
    for m, reply in replies:
        hints.append(str(reply.get("relax_hint") or "").strip())
        alts += [normalize(a, True) | {"author": m} for a in reply.get("alternatives") or []
                 if isinstance(a, dict) and a.get("name")]
    critic = critic_model(session.cfg, authors)
    meta = {"models_used": authors, "critic_model": critic, "sample_errors": errors, "n_proposed": len(alts)}
    if not alts:
        return [], hints, meta
    proposals = "\n".join(f"[{i}] {a['name']}: {a['mechanism']}" for i, a in enumerate(alts))
    try:
        reply = ask_json(session.llm, critic, CRITIC_PROMPT.format(proposals=proposals, **critic_inputs),
                         max_tokens=6000, required=("scores",))
        scores = {s.get("index"): s for s in reply.get("scores") or [] if isinstance(s, dict)}
    except BudgetExceeded:
        raise
    except Exception as e:
        meta["critic_error"], scores = redact(str(e))[:300], {}
    kept, repeats = [], 0
    for i, a in enumerate(alts):
        s = scores.get(i) or {}
        same = _index(s.get("same_as"), i)
        if same is not None:
            repeats += 1
            continue
        kept.append(a | {"critic": _critic_total(s)} if s else a)
    meta["n_same_mechanism_dropped"] = repeats
    if scores:
        kept.sort(key=lambda a: -sum((a.get("critic") or {}).values()))
    return kept[:n], hints, meta


def grounded_alternatives(session, task, my_approach, consensus_text, already_tried=None, constraints=None,
                          context="", depth="quick", n=5):
    """Free-text alternatives using the agent's inputs: generation (one call, or deep mode), screening
    against already_tried and constraints, the grounding check and the cost ranking. Returns the fields
    free_text_check adds to its result."""
    from .overlook import extractor
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {list(DEPTHS)}, got {depth!r}")
    logged = logged_as_tried(logged_outcomes(task))
    tried = merge_tried(clean_tried(already_tried), logged)
    rules = clean_constraints(constraints)
    ctx, truncated = clean_context(context)
    blocks = prompt_blocks(tried, rules, ctx)
    prompt = EXTENDED_PROMPT.format(task=task.prompt, approach=my_approach, herd=consensus_text, n=n, **blocks)
    out = {"agent_inputs": {"already_tried": len(tried), "logged_outcomes": len(logged), "constraints": len(rules),
                            "context_chars": len(ctx), "context_truncated": truncated, "depth": depth}}
    if depth == "deep":
        alts, hints, meta = deep_alternatives(session, task, prompt, {
            "task": task.prompt, "approach": my_approach, "herd": consensus_text, "tried": blocks["tried"],
            "constraints": blocks["constraints"]}, n)
        out["deep"] = meta
        authors = meta["models_used"]
    else:
        reply = ask_json(session.llm, extractor(session), prompt, required=("alternatives",))
        alts = [normalize(a, True) for a in reply.get("alternatives") or [] if isinstance(a, dict) and a.get("name")][:n]
        hints, authors = [str(reply.get("relax_hint") or "").strip()], [extractor(session)]
    filtered = []
    if alts and (tried or rules):
        try:
            alts, filtered = screen(session.llm, critic_model(session.cfg, authors), task, alts, tried, rules)
        except BudgetExceeded:
            raise
        except Exception as e:
            out["screen_error"] = redact(str(e))[:300]
    if ctx:
        for a in alts:
            a["ungrounded_numbers"] = ungrounded_numbers(a, ctx)
    alts = rank(alts)
    out.update(alternatives=alts, filtered_out=filtered)
    if rules:
        out["relax_hint"] = next((h for h in hints if h), "")
    if alts:
        first = alts[0]
        out["next_action"] = (f"Test {first['name']} first ({first.get('test_cost', 'unknown')} to test"
                              + (", decisive" if first.get("decisive") else "") + "): "
                              + (f"it is wrong if {first['kill_result']}. " if first.get("kill_result") else "")
                              + "All alternatives are HYPOTHESIZED: DIVERGE has not implemented or scored any of "
                              "them. After testing, call diverge_report_outcome so DIVERGE never offers a failed "
                              "idea again; or call diverge_check with evaluator_cmd to have DIVERGE implement and "
                              "score the families itself.")
    return out


def uses_inputs(already_tried, constraints, context, depth, logged):
    return bool(already_tried or constraints or (context or "").strip() or depth != "quick" or logged)
