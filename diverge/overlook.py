"""Overlooked ideas below the family level (protocol: docs/PREREGISTRATION-overlook.md).

From the consensus samples, per task:
1. decision-input audit: which inputs available at decision time the consensus code actually uses
   (static backward slice first; the extractor model only for cases marked ambiguous);
2. shared assumptions of the top families' code, hard (quoted verbatim from the prompt) vs
   incidental;
3. component grid: each sample decomposed into fixed components; combinations that literature
   occupies or that are coherent but no consensus sample uses;
4. minority report: families and component values produced by only one or two sampler models;
5. latent tail: approaches models name when shown the consensus (vs not), verified.
Everything is written to results/<task>/overlook.json with its evidence."""
import ast
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from .label import EMERGENT, UNLABELED
from .llm import ask_json, message_text, parse_json
from .pipeline import read_json, write_json

OVERLOOKED_SHARE = 0.10

# ---------------------------------------------------------------- 1. inputs (fixed before reading samples)
INPUTS = {
    "online_bin_packing": {
        "item": "size of the arriving item",
        "capacity_values": "remaining capacity of each feasible bin",
        "bin_order": "a bin's position in the list, i.e. when it was opened (bin age)",
        "new_bin_slot": "the last entry is the option of opening a new, empty bin",
        "n_open_bins": "how many feasible bins are offered (len(capacities))",
        "history": "module-level state carried across calls (past items, counts, statistics)",
        "item_distribution": "the stated item-size distribution, Weibull(k=3, scale=45)",
        "bin_capacity": "the bin capacity, 100",
    },
    "cache_eviction": {
        "capacity": "the cache capacity passed to __init__",
        "key_value": "the key's numeric value (order or arithmetic on keys, e.g. detecting sequential scans)",
        "time": "a logical clock: when requests happened (an event counter)",
        "frequency": "how often a key was requested",
        "eviction_history": "keys evicted earlier (ghost entries) consulted on later requests",
        "occupancy": "how full the cache or a segment is, compared with a size",
        "workload_mix": "the stated workload: sequential scans, one-hit-wonder keys, Zipf skew",
    },
}

MUTATORS = {"append", "extend", "insert", "add", "update", "appendleft", "extendleft", "setdefault",
            "move_to_end", "push", "heappush", "__setitem__"}


def names(node):
    """Names and self.attr tokens read anywhere inside node."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
            out.add(f"self.{n.attr}")
    return out


def target_names(t):
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
        return {f"self.{t.attr}"}
    if isinstance(t, (ast.Tuple, ast.List)):
        return set().union(*(target_names(e) for e in t.elts)) if t.elts else set()
    if isinstance(t, (ast.Subscript, ast.Attribute, ast.Starred)):
        return target_names(t.value)
    return set()


def edges_of(body, ctx=frozenset(), out=None, sinks=None):
    """Dependency edges (targets <- sources) and return sources of a function body. ctx carries
    control dependence (names in enclosing conditions and loop iterators)."""
    out = [] if out is None else out
    sinks = set() if sinks is None else sinks
    for st in body:
        if isinstance(st, ast.Assign):
            src = names(st.value) | set(ctx)
            for t in st.targets:
                extra = names(t.slice) if isinstance(t, ast.Subscript) else set()
                out.append((target_names(t), src | extra))
        elif isinstance(st, ast.AugAssign):
            tn = target_names(st.target)
            extra = names(st.target.slice) if isinstance(st.target, ast.Subscript) else set()
            out.append((tn, names(st.value) | tn | set(ctx) | extra))
        elif isinstance(st, ast.AnnAssign) and st.value is not None:
            out.append((target_names(st.target), names(st.value) | set(ctx)))
        elif isinstance(st, (ast.For, ast.AsyncFor)):
            it = names(st.iter)
            out.append((target_names(st.target), it | set(ctx)))
            edges_of(st.body + st.orelse, ctx | it, out, sinks)
        elif isinstance(st, (ast.If, ast.While)):
            edges_of(st.body + st.orelse, ctx | names(st.test), out, sinks)
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            edges_of(st.body, ctx, out, sinks)
        elif isinstance(st, ast.Try):
            edges_of(st.body + st.orelse + st.finalbody + [s for h in st.handlers for s in h.body], ctx, out, sinks)
        elif isinstance(st, ast.Return) and st.value is not None:
            sinks |= names(st.value) | set(ctx)
        elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
            call = st.value
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr in MUTATORS:
                base = target_names(func.value)
                out.append((base, set().union(*[names(a) for a in call.args], set()) | set(ctx)))
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in ("heapq", "bisect"):
                if call.args:
                    base = target_names(call.args[0])
                    out.append((base, set().union(*[names(a) for a in call.args[1:]], set()) | set(ctx)))
        elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = []
            inner_sinks = set()
            edges_of(st.body, frozenset(), inner, inner_sinks)
            out.append(({st.name}, inner_sinks | set().union(*[s for _, s in inner], set())))
    return out, sinks


def relevant(edges, sinks):
    rel = set(sinks)
    changed = True
    while changed:
        changed = False
        for targets, sources in edges:
            if targets & rel and not sources <= rel:
                rel |= sources
                changed = True
    return rel


def module_info(tree):
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    globals_ = set()
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                globals_ |= target_names(t)
    return funcs, classes, globals_


def helper_edges(tree, funcs, exclude):
    """A call to a module-level helper depends on everything the helper reads."""
    out = []
    for name, f in funcs.items():
        if name in exclude:
            continue
        e, s = edges_of(f.body)
        out.append(({name}, s | set().union(*[src for _, src in e], set()) | {a.arg for a in f.args.args}))
    return out


def constants(node):
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))}


def parents(tree):
    p = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            p[child] = node
    return p


def audit_bin_packing(code):
    """{input: (status, evidence)} for a priority(item, capacities) module."""
    tree = ast.parse(code)
    funcs, classes, globals_ = module_info(tree)
    fn = funcs.get("priority")
    if fn is None or len(fn.args.args) < 2:
        return {k: ("ambiguous", "no priority(item, capacities) function") for k in INPUTS["online_bin_packing"]}
    p_item, p_caps = fn.args.args[0].arg, fn.args.args[1].arg
    edges, sinks = edges_of(fn.body)
    edges += helper_edges(tree, funcs, {"priority"})
    rel = relevant(edges, sinks)
    par = parents(tree)
    result = {}
    line = lambda n: getattr(n, "lineno", "?")

    result["item"] = ("used", "item flows into the scores") if p_item in rel else ("unused", "item never affects the scores")

    cap_uses = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == p_caps]
    non_len = [n for n in cap_uses if not (isinstance(par.get(n), ast.Call) and getattr(par[n].func, "id", None) == "len")]
    result["capacity_values"] = (("used", f"capacities read at line {line(non_len[0])}") if p_caps in rel and non_len
                                 else ("unused", "capacity values never affect the scores"))

    # Index variables: loop/comprehension targets over enumerate()/range()/arange().
    index_vars = set()
    for n in ast.walk(fn):
        gens = [n] if isinstance(n, ast.For) else (n.generators if isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)) else [])
        for g in gens:
            it, tgt = g.iter, g.target
            fname = getattr(it.func, "id", None) or getattr(it.func, "attr", None) if isinstance(it, ast.Call) else None
            if fname == "enumerate" and isinstance(tgt, ast.Tuple) and tgt.elts and isinstance(tgt.elts[0], ast.Name):
                index_vars.add(tgt.elts[0].id)
            elif fname in ("range", "arange") and isinstance(tgt, ast.Name):
                index_vars.add(tgt.id)
    positional, new_bin = [], []
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id in index_vars and n.id in rel:
            parent = par.get(n)
            if isinstance(parent, ast.Subscript) and parent.slice is n:
                continue  # only indexing capacities[i]
            if isinstance(parent, ast.Compare) and any(isinstance(c, (ast.BinOp, ast.UnaryOp)) or
                                                       (isinstance(c, ast.Call) and getattr(c.func, "id", "") == "len")
                                                       for c in [parent.left, *parent.comparators]):
                new_bin.append(line(n))  # i == len(capacities) - 1 and the like
            elif isinstance(parent, (ast.Compare, ast.BinOp, ast.UnaryOp, ast.Call, ast.Return, ast.Tuple, ast.List)):
                positional.append(line(n))
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) in ("arange", "range"):
            parent = par.get(n)
            in_iter = isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is n
            if not in_iter and isinstance(parent, (ast.BinOp, ast.UnaryOp, ast.Call)):
                positional.append(line(n))
    result["bin_order"] = ("used", f"bin position used at lines {sorted(set(positional))[:5]}") if positional \
        else ("unused", "bin position never enters a score")

    for n in ast.walk(fn):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == p_caps:
            s = n.slice
            if isinstance(s, ast.UnaryOp) and isinstance(s.op, ast.USub) and getattr(s.operand, "value", None) == 1:
                new_bin.append(line(n))
            if isinstance(s, ast.Slice) and isinstance(s.upper, ast.UnaryOp):
                new_bin.append(line(n))
        if isinstance(n, ast.Compare) and any(isinstance(c, ast.Constant) and c.value in (100, 100.0)
                                             for c in [n.left, *n.comparators]) and names(n) & rel:
            new_bin.append(line(n))
        # Writing the last score (aligned with the new-bin entry) also treats the new bin specially.
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AugAssign) else []
        for t in targets:
            if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.UnaryOp) and isinstance(t.slice.op, ast.USub) \
                    and getattr(t.slice.operand, "value", None) == 1 and target_names(t) & rel:
                new_bin.append(line(n))
    result["new_bin_slot"] = ("used", f"new-bin entry treated specially at lines {sorted(set(new_bin))[:5]}") if new_bin \
        else ("unused", "the new-bin entry is scored like any other bin")

    len_uses = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "len"
                and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == p_caps]
    counted = [n for n in len_uses if isinstance(par.get(n), (ast.BinOp, ast.Compare)) and not (
        isinstance(par.get(n), ast.BinOp) and isinstance(par[n].op, ast.Sub) and getattr(par[n].right, "value", None) == 1)
        and not isinstance(par.get(par.get(n)), ast.Call)]
    result["n_open_bins"] = ("used", f"number of bins enters a score at line {line(counted[0])}") if counted \
        else ("unused", "the number of offered bins is only used for sizing or iteration")

    declared_global = {g for n in ast.walk(fn) if isinstance(n, ast.Global) for g in n.names}
    mutated = set()
    for t, _ in edges:
        mutated |= {x for x in t if x in globals_}
    history = (declared_global | mutated) & rel
    result["history"] = ("used", f"state carried across calls: {sorted(history)}") if history \
        else ("unused", "no state carried across calls affects the scores")

    lits = constants(tree)
    src = code.lower()
    dist = 45 in lits or 45.0 in lits or "weibull" in src
    result["item_distribution"] = ("used", "the Weibull parameters or name appear in the code") if dist \
        else ("unused", "the stated item distribution is never used")
    result["bin_capacity"] = (("used", "the capacity constant 100 appears in the code") if (100 in lits or 100.0 in lits)
                              else ("unused", "the bin capacity is never used"))
    return result


def audit_cache(code):
    """{input: (status, evidence)} for a Policy class; workload_mix is always left to the extractor."""
    tree = ast.parse(code)
    funcs, classes, globals_ = module_info(tree)
    cls = classes.get("Policy")
    if cls is None:
        return {k: ("ambiguous", "no Policy class") for k in INPUTS["cache_eviction"]}
    methods = {m.name: m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
    edges, sinks = [], set()
    for name, m in methods.items():
        e, s = edges_of(m.body)
        edges += e
        if name == "evict":
            sinks |= s
        else:  # helpers that return values feed the methods calling them
            edges.append(({f"self.{name}"}, s))
    edges += helper_edges(tree, funcs, set())
    rel = relevant(edges, sinks)
    par = parents(tree)
    line = lambda n: getattr(n, "lineno", "?")
    params = {a.arg for name, m in methods.items() for a in m.args.args[1:] if name in ("on_hit", "on_miss")}
    cap_param = {a.arg for a in methods["__init__"].args.args[1:2]} if "__init__" in methods else set()
    result = {}

    result["capacity"] = ("used", "capacity flows into eviction decisions") if cap_param & rel \
        else ("unused", "capacity never affects which key is evicted")

    key_value = []
    for n in ast.walk(cls):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)) and names(n) & params:
            key_value.append(line(n))
        if isinstance(n, ast.Compare) and names(n) & params and (
                any(isinstance(o, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)) for o in n.ops) or
                any(isinstance(c, ast.BinOp) and isinstance(c.op, (ast.Add, ast.Sub)) for c in [n.left, *n.comparators])):
            key_value.append(line(n))  # ordering on keys, or key == previous + 1 (sequential scans)
    result["key_value"] = ("used", f"arithmetic or ordering on keys at lines {sorted(set(key_value))[:5]}") if key_value \
        else ("unused", "keys are only used as identities")

    counters, freq = [], []
    for n in ast.walk(cls):
        if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add):
            tn = target_names(n.target)
            if isinstance(n.target, ast.Attribute) and tn & rel:
                counters.append(line(n))
            if isinstance(n.target, ast.Subscript) and tn & rel:
                freq.append(line(n))
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Add):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and target_names(t) & rel and "get" in {
                        getattr(c.func, "attr", "") for c in ast.walk(n.value) if isinstance(c, ast.Call)}:
                    freq.append(line(n))
                if isinstance(t, ast.Attribute) and target_names(t) & rel and target_names(t) <= names(n.value):
                    counters.append(line(n))
    result["time"] = ("used", f"an event counter feeds eviction (lines {sorted(set(counters))[:5]})") if counters \
        else ("ambiguous", "no scalar counter found; ordering structures may still encode time")
    result["frequency"] = ("used", f"per-key counts feed eviction (lines {sorted(set(freq))[:5]})") if freq \
        else ("ambiguous", "no per-key counter found")

    ghosts = []
    if "evict" in methods:
        for n in ast.walk(methods["evict"]):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUTATORS:
                base = target_names(n.func.value)
                read_elsewhere = any(isinstance(c, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in c.ops)
                                     and names(c) & base for m in ("on_miss", "on_hit") if m in methods
                                     for c in ast.walk(methods[m]))
                if base and read_elsewhere and base & rel:
                    ghosts.append(line(n))
    result["eviction_history"] = ("used", f"evicted keys recorded and consulted later (lines {ghosts[:5]})") if ghosts \
        else ("ambiguous", "no ghost structure found statically")

    occ = [line(n) for n in ast.walk(cls) if isinstance(n, ast.Compare) and any(
        isinstance(c, ast.Call) and getattr(c.func, "id", None) == "len" for c in [n.left, *n.comparators]) and names(n) & rel]
    result["occupancy"] = ("used", f"fill level compared with a size at lines {occ[:5]}") if occ \
        else ("unused", "fill levels never drive decisions")
    result["workload_mix"] = ("ambiguous", "needs reading: does the code target scans, one-hit keys or skew?")
    return result


AUDITORS = {"online_bin_packing": audit_bin_packing, "cache_eviction": audit_cache}

RESOLVE_PROMPT = """You are auditing code written for this task:

<task>
{task}
</task>

For each solution below, decide for each listed input whether the solution's DECISION (the
scores it returns, or the key it evicts) actually depends on that input. Judge the code, not its
comments. Inputs:
{inputs}

{solutions}

Return JSON: {{"results": [{{"sample_id": "...", "input": "...", "used": true, "evidence": "line N: why"}}]}}"""


def resolve_ambiguous(session, task, items, codes):
    """items: [(sample_id, input_name)] the static pass could not decide."""
    by_sample = defaultdict(list)
    for sid, inp in items:
        by_sample[sid].append(inp)
    sids = sorted(by_sample)
    out = {}

    def batch(chunk):
        listing = "\n".join(f"- {k}: {v}" for k, v in INPUTS[task.name].items()
                            if any(k in by_sample[s] for s in chunk))
        sols = "\n\n".join(f'<solution id="{s}" inputs="{",".join(by_sample[s])}">\n{codes[s]}\n</solution>' for s in chunk)
        try:
            reply = ask_json(session.llm, extractor(session), RESOLVE_PROMPT.format(task=task.prompt, inputs=listing,
                                                                              solutions=sols), required=("results",))
        except ValueError:
            return {}
        return {(r.get("sample_id"), r.get("input")): r for r in reply["results"] if isinstance(r, dict)}

    chunks = [sids[i:i + 8] for i in range(0, len(sids), 8)]
    with ThreadPoolExecutor(4) as pool:
        for part in pool.map(batch, chunks):
            out.update(part)
    return out


def extractor(session):
    return session.cfg["models"].get("extractor") or session.cfg["models"]["labeler"]


def valid_samples(out):
    samples = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()]
    labels = {json.loads(l)["sample_id"]: json.loads(l)["family"] for l in (out / "labels.jsonl").read_text().splitlines()}
    valid = [s for s in samples if s.get("status") == "valid" and s.get("code_path")]
    for s in valid:
        s["family"] = labels.get(s["sample_id"], UNLABELED)
        s["code"] = (out / s["code_path"]).read_text()
    return valid


def input_audit(session, task, valid):
    auditor = AUDITORS[task.name]
    per_sample, ambiguous = {}, []
    for s in valid:
        try:
            res = auditor(s["code"])
        except SyntaxError:
            res = {k: ("ambiguous", "does not parse") for k in INPUTS[task.name]}
        per_sample[s["sample_id"]] = {k: {"status": v[0], "evidence": v[1], "method": "static"} for k, v in res.items()}
        ambiguous += [(s["sample_id"], k) for k, v in res.items() if v[0] == "ambiguous"]
    if ambiguous:
        resolved = resolve_ambiguous(session, task, ambiguous, {s["sample_id"]: s["code"] for s in valid})
        for sid, k in ambiguous:
            r = resolved.get((sid, k))
            if r is not None and isinstance(r.get("used"), bool):
                per_sample[sid][k] = {"status": "used" if r["used"] else "unused",
                                      "evidence": str(r.get("evidence", ""))[:300], "method": "extractor"}
    model_of = {s["sample_id"]: s["model"] for s in valid}
    rows = []
    for k, desc in INPUTS[task.name].items():
        statuses = {sid: v[k]["status"] for sid, v in per_sample.items()}
        decided = [sid for sid, st in statuses.items() if st != "ambiguous"]
        users = sorted(sid for sid in decided if statuses[sid] == "used")
        share = len(users) / len(decided) if decided else None
        rows.append({"input": k, "description": desc, "used_by": len(users), "decided": len(decided),
                     "undecided": len(statuses) - len(decided), "share": round(share, 3) if share is not None else None,
                     "overlooked": share is not None and share <= OVERLOOKED_SHARE,
                     "used_by_models": dict(Counter(model_of[s] for s in users)),
                     "methods": dict(Counter(per_sample[s][k]["method"] for s in statuses)),
                     "evidence": [{"sample_id": s, "model": model_of[s], "evidence": per_sample[s][k]["evidence"]}
                                  for s in users[:5]]})
    rows.sort(key=lambda r: (r["share"] is None, r["share"] if r["share"] is not None else 1))
    return rows, per_sample


# ---------------------------------------------------------------- 2. shared assumptions
PROPOSE_PROMPT = """These solutions were written by AI models for the task below. They come from the
most common solution families. Read the CODE (not comments or names) and list assumptions that
ALL of these solutions share: what information they use or ignore, what they never do, what they
hold fixed, how they treat time, admission, state, ties and the like.

For each assumption, say whether the task statement FORCES it. If it does, copy the exact
sentence fragment from the task statement that forces it into "prompt_quote" (verbatim, at least
6 words); if nothing in the task forces it, leave "prompt_quote" empty.

<task>
{task}
</task>

{solutions}

Return JSON: {{"assumptions": [{{"assumption": "...", "prompt_quote": "", "evidence": [{{"sample_id": "...", "lines": "..."}}]}}]}}"""

VERIFY_PROMPT = """For each numbered assumption and each solution, does the solution's CODE make that
assumption? Answer for every pair.

Assumptions:
{assumptions}

{solutions}

Keep each evidence under 15 words.

Return JSON: {{"checks": [{{"assumption": 1, "sample_id": "...", "holds": true, "evidence": "line N: ..."}}]}}"""


def normalize_ws(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def classify_assumption(assumption, prompt):
    """Hard only if it quotes the prompt verbatim (whitespace-normalized, at least 6 words)."""
    quote = (assumption.get("prompt_quote") or "").strip().strip('"')
    if quote and len(quote.split()) >= 6 and normalize_ws(quote) in normalize_ws(prompt):
        return "hard", quote
    return "incidental", quote if quote else None


def shared_assumptions(session, task, valid, top3, max_samples=12):
    chosen = []
    for fam in top3:
        chosen += sorted((s for s in valid if s["family"] == fam), key=lambda s: s["sample_id"])[:max_samples // 3]
    chosen = chosen[:max_samples]
    sols = "\n\n".join(f'<solution id="{s["sample_id"]}" family="{s["family"]}">\n{s["code"]}\n</solution>' for s in chosen)
    proposed = ask_json(session.llm, extractor(session), PROPOSE_PROMPT.format(task=task.prompt, solutions=sols),
                        required=("assumptions",))["assumptions"]
    proposed = [a for a in proposed if isinstance(a, dict) and a.get("assumption")]
    listing = "\n".join(f"{i + 1}. {a['assumption']}" for i, a in enumerate(proposed))
    # Verify in batches of 4 solutions: one call for all 12 can exceed the completion limit on long code.
    checks = []
    for start in range(0, len(chosen), 4):
        part = "\n\n".join(f'<solution id="{s["sample_id"]}" family="{s["family"]}">\n{s["code"]}\n</solution>'
                            for s in chosen[start:start + 4])
        checks += ask_json(session.llm, extractor(session), VERIFY_PROMPT.format(assumptions=listing, solutions=part),
                           required=("checks",))["checks"]
    holds = defaultdict(dict)
    for c in checks:
        if isinstance(c, dict) and isinstance(c.get("assumption"), int):
            holds[c["assumption"] - 1][c.get("sample_id")] = (c.get("holds") is True, c.get("evidence"))
    ids = [s["sample_id"] for s in chosen]
    rows = []
    for i, a in enumerate(proposed):
        h = holds.get(i, {})
        held_by = [sid for sid in ids if h.get(sid, (False,))[0]]
        kind, quote = classify_assumption(a, task.prompt)
        rows.append({"assumption": a["assumption"], "kind": kind, "prompt_quote": quote,
                     "shared": len(held_by) == len(ids), "held_by": len(held_by), "checked": len(ids),
                     "evidence": [{"sample_id": sid, "evidence": h[sid][1]} for sid in held_by[:4]],
                     "proposer_evidence": a.get("evidence", [])[:3]})
    return {"checked_samples": ids, "families": top3, "assumptions": rows}


# ---------------------------------------------------------------- 3. component grid
COMPONENTS = {
    "online_bin_packing": {"scoring_term": "the main quantity the score is computed from",
                           "tie_break": "how near-equal bins are ordered",
                           "state_kept": "information kept across calls (or none)",
                           "new_bin_rule": "when the new, empty bin is preferred or avoided"},
    "cache_eviction": {"admission": "which missed keys are kept or filtered on insertion",
                       "eviction_score": "what decides which resident key is evicted",
                       "state_kept": "data structures kept per key or overall",
                       "aging": "how old information decays or is reset"},
}

DECOMPOSE_PROMPT = """Decompose each solution's decision logic into these components. For each
component give a SHORT value (at most 8 words) describing what the CODE does; use "none" if the
solution has no such component.
{components}

<task>
{task}
</task>

{solutions}

Return JSON: {{"items": [{{"id": "...", {fields}}}]}}"""

CANON_PROMPT = """Merge these values of the component "{component}" ({desc}) into at most {k} canonical
labels (2-5 words each). Values that describe the same mechanism get the same label; keep
genuinely different mechanisms apart. Map EVERY value.

Values:
{values}

Return JSON: {{"mapping": {{"<value>": "<label>"}}}}"""

COHERENT_PROMPT = """For the task below, a solution would combine the two component values in each
pair. Is the combination coherent and implementable under the task's interface and rules (not
necessarily good)? Answer for every pair.

<task>
{task}
</task>

{pairs}

Return JSON: {{"pairs": [{{"n": 1, "coherent": true, "reason": "..."}}]}}"""


def decompose(session, task, items, kind):
    comps = COMPONENTS[task.name]
    fields = ", ".join(f'"{c}": "..."' for c in comps)
    listing = "\n".join(f"- {c}: {d}" for c, d in comps.items())
    out = {}

    def batch(chunk):
        body = "\n\n".join(f'<{kind} id="{i}">\n{text}\n</{kind}>' for i, text in chunk)
        try:
            reply = ask_json(session.llm, extractor(session), DECOMPOSE_PROMPT.format(components=listing, task=task.prompt,
                                                                                solutions=body, fields=fields),
                             required=("items",))
        except ValueError:
            return {}
        return {r["id"]: {c: str(r.get(c, "none")).strip().lower() for c in comps}
                for r in reply["items"] if isinstance(r, dict) and r.get("id") is not None}

    chunks = [items[i:i + 8] for i in range(0, len(items), 8)]
    with ThreadPoolExecutor(4) as pool:
        for part in pool.map(batch, chunks):
            out.update(part)
    return out


def canonicalize(session, task, raw, k=8):
    comps = COMPONENTS[task.name]
    maps = {}
    for c, desc in comps.items():
        values = sorted({v[c] for v in raw.values()})
        try:
            reply = ask_json(session.llm, extractor(session), CANON_PROMPT.format(
                component=c, desc=desc, k=k, values="\n".join(f"- {v}" for v in values)), required=("mapping",))
            mapping = {str(a).strip().lower(): str(b).strip().lower() for a, b in reply["mapping"].items()}
        except ValueError:
            mapping = {}
        maps[c] = {v: mapping.get(v, v) for v in values}
    return {i: {c: maps[c][v[c]] for c in comps} for i, v in raw.items()}, maps


def grid_gaps(consensus_cells, literature_cells, min_support=2):
    """Gap candidates: (a) full combinations occupied by literature but no consensus sample;
    (b) value pairs, each value seen in >= min_support consensus samples, never seen together."""
    consensus_tuples = {tuple(sorted(v.items())) for v in consensus_cells.values()}
    lit = [{"family": f, "cell": v} for f, v in literature_cells.items() if tuple(sorted(v.items())) not in consensus_tuples]
    support = Counter((c, v) for cells in consensus_cells.values() for c, v in cells.items())
    seen_pairs = Counter()
    for cells in consensus_cells.values():
        items = sorted(cells.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                seen_pairs[(items[i], items[j])] += 1
    frequent = sorted(((c, v) for (c, v), n in support.items() if n >= min_support and v != "none"))
    pairs = []
    for i in range(len(frequent)):
        for j in range(i + 1, len(frequent)):
            a, b = frequent[i], frequent[j]
            if a[0] == b[0] or (a, b) in seen_pairs:
                continue
            pairs.append({"pair": [list(a), list(b)], "support": [support[a], support[b]]})
    pairs.sort(key=lambda p: -p["support"][0] * p["support"][1])
    return lit, pairs


def component_grid(session, task, valid, families, max_pairs=20):
    raw = decompose(session, task, [(s["sample_id"], s["code"]) for s in valid], "solution")
    lit_items = [(f["family"], f"{f['family']}: {f['one_line_mechanism']}") for f in families
                 if f.get("source_urls") and not f["family"].startswith(EMERGENT)]
    lit_raw = decompose(session, task, lit_items, "family")
    canon, maps = canonicalize(session, task, {**raw, **{f"lit::{k}": v for k, v in lit_raw.items()}})
    consensus_cells = {k: v for k, v in canon.items() if not k.startswith("lit::")}
    lit_cells = {k[5:]: v for k, v in canon.items() if k.startswith("lit::")}
    lit_gaps, pair_gaps = grid_gaps(consensus_cells, lit_cells)
    judged = []
    if pair_gaps:
        top = pair_gaps[:max_pairs]
        listing = "\n".join(f"{i + 1}. {p['pair'][0][0]} = {p['pair'][0][1]}  +  {p['pair'][1][0]} = {p['pair'][1][1]}"
                            for i, p in enumerate(top))
        try:
            reply = ask_json(session.llm, extractor(session), COHERENT_PROMPT.format(task=task.prompt, pairs=listing),
                             required=("pairs",))
            verdicts = {r.get("n"): r for r in reply["pairs"] if isinstance(r, dict)}
        except ValueError:
            verdicts = {}
        for i, p in enumerate(top):
            v = verdicts.get(i + 1, {})
            judged.append({**p, "coherent": v.get("coherent") is True, "reason": v.get("reason")})
    value_samples = defaultdict(list)
    for sid, cells in consensus_cells.items():
        for c, v in cells.items():
            value_samples[f"{c}={v}"].append(sid)
    for g in judged:
        g["evidence"] = {f"{c}={v}": value_samples[f"{c}={v}"][:4] for c, v in g["pair"]}
    return {"components": COMPONENTS[task.name], "vocabulary": {c: sorted(set(m.values())) for c, m in maps.items()},
            "herd_cells": consensus_cells, "literature_cells": lit_cells,
            "literature_only_cells": lit_gaps,
            "unoccupied_pairs": [g for g in judged if g["coherent"]],
            "incoherent_pairs": [g for g in judged if not g["coherent"]],
            "pairs_considered": len(pair_gaps)}


# ---------------------------------------------------------------- 4. minority report
def minority(items_by_model, all_models, max_models=2):
    """items_by_model: {item: {model: [sample ids]}}. Items produced by 1..max_models models only."""
    out = []
    for item, by_model in items_by_model.items():
        models = sorted(m for m, sids in by_model.items() if sids)
        if 1 <= len(models) <= max_models and len(all_models) > len(models):
            out.append({"item": item, "models": models, "absent_from": sorted(set(all_models) - set(models)),
                        "samples": {m: by_model[m][:4] for m in models},
                        "count": sum(len(by_model[m]) for m in models)})
    return sorted(out, key=lambda r: (len(r["models"]), -r["count"], r["item"]))


def minority_report(valid, all_samples, models, grid):
    fam = defaultdict(lambda: defaultdict(list))
    for s in all_samples:
        # A reply with no code (e.g. a sampler that spent its tokens reasoning) can still be labeled, as
        # "emergent:empty-solution"; it is not a family any model produced, so it never becomes a lever.
        if s.get("family") and s["family"] != UNLABELED and s.get("status") != "no_code":
            fam[s["family"]][s["model"]].append(s["sample_id"])
    comp = defaultdict(lambda: defaultdict(list))
    model_of = {s["sample_id"]: s["model"] for s in valid}
    for sid, cells in grid["herd_cells"].items():
        for c, v in cells.items():
            if v != "none":
                comp[f"{c}={v}"][model_of[sid]].append(sid)
    return {"families": minority(fam, models), "components": minority(comp, models)}


# ---------------------------------------------------------------- 5. latent tail
LATENT_WITH = """{task}

---
Most AI-generated solutions to this exact problem use one of these approaches:
{herd}

Do NOT repeat these approaches or minor variants of them. Which OTHER approaches do you know for
this exact problem? List up to 8. For each give a short name, its mechanism (the state it keeps
and its decision rule) and a reference if you know one. Do not write code.

Return JSON: {{"approaches": [{{"name": "...", "mechanism": "...", "reference": "..."}}]}}"""

LATENT_WITHOUT = """{task}

---
Which approaches do you know for this exact problem? List up to 8. For each give a short name, its
mechanism (the state it keeps and its decision rule) and a reference if you know one. Do not
write code.

Return JSON: {{"approaches": [{{"name": "...", "mechanism": "...", "reference": "..."}}]}}"""

MATCH_PROMPT = """Match each proposed approach to the taxonomy family whose MECHANISM it describes, or
"none" if no family fits. Also say whether its mechanism is precise (names the state kept AND the
decision rule).

Taxonomy:
{taxonomy}

Approaches:
{approaches}

Return JSON: {{"matches": [{{"n": 1, "family": "exact family name or none", "precise": true}}]}}"""


def mentions(name, result):
    """Does a search result's title or highlights contain the approach's distinctive words?"""
    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) > 3]
    text = " ".join([result.get("title") or "", *result.get("highlights", [])]).lower()
    return bool(words) and all(w in text for w in words)


def latent_tail(session, task, families, top3, max_exa=3):
    by_name = {f["family"]: f for f in families}
    consensus_list = "\n".join(f"- {n}: {by_name[n]['one_line_mechanism']}" for n in top3 if n in by_name)
    prompts = {"with_herd": LATENT_WITH.format(task=task.prompt, herd=consensus_list),
               "without_herd": LATENT_WITHOUT.format(task=task.prompt)}
    jobs = [(m, cond) for m in session.cfg["models"]["samplers"] for cond in prompts]

    def ask(job):
        model, cond = job
        try:
            entry = session.llm.chat(model, [{"role": "user", "content": prompts[cond]}], temperature=1.0,
                                  max_tokens=session.cfg["sampling"]["max_tokens"], sample_idx=f"latent:{cond}")
            reply = parse_json(message_text(entry))
            approaches = [a for a in reply.get("approaches", []) if isinstance(a, dict) and a.get("name")]
            return model, cond, approaches, None
        except Exception as e:
            return model, cond, [], f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(5) as pool:
        answers = list(pool.map(ask, jobs))
    flat = [(m, c, a) for m, c, aps, _ in answers for a in aps]
    taxonomy = "\n".join(f"- {f['family']}: {f['one_line_mechanism']}" for f in families)
    listing = "\n".join(f"{i + 1}. {a['name']}: {a.get('mechanism', '')}" for i, (_, _, a) in enumerate(flat))
    matches = {}
    if flat:
        try:
            reply = ask_json(session.llm, extractor(session), MATCH_PROMPT.format(taxonomy=taxonomy, approaches=listing),
                             required=("matches",))
            matches = {r.get("n"): r for r in reply["matches"] if isinstance(r, dict)}
        except ValueError:
            pass
    consensus_set = set(top3)
    rows = []
    for i, (m, c, a) in enumerate(flat):
        mt = matches.get(i + 1, {})
        fam = mt.get("family") if mt.get("family") in by_name else None
        rows.append({"model": m, "condition": c, "name": a["name"], "mechanism": a.get("mechanism"),
                     "reference": a.get("reference"), "taxonomy_family": fam, "precise": mt.get("precise") is True,
                     "in_herd": fam in consensus_set})
    # Exa checks for the most-named approaches outside the taxonomy.
    unmatched = Counter(r["name"].strip().lower() for r in rows if r["taxonomy_family"] is None)
    sources = {}
    for name, _ in unmatched.most_common(max_exa):
        try:
            res = session.exa.search(f"{name} {task.name.replace('_', ' ')} algorithm", num_results=3)
            # A result only counts as a source if its title or highlights actually mention the approach.
            sources[name] = [{"url": r["url"], "title": r["title"], "mentions": mentions(name, r)} for r in res]
        except Exception as e:
            sources[name] = [{"error": str(e)}]
    for r in rows:
        r["sources"] = sources.get(r["name"].strip().lower(), [])
        r["verified"] = bool(r["taxonomy_family"]) or r["precise"] or any(s.get("mentions") for s in r["sources"])
    per_model = {}
    for m in session.cfg["models"]["samplers"]:
        per_model[m] = {c: len({(r["taxonomy_family"] or r["name"].lower()) for r in rows
                                if r["model"] == m and r["condition"] == c and r["verified"] and not r["in_herd"]})
                        for c in prompts}
    more = sum(v["with_herd"] > v["without_herd"] for v in per_model.values())
    return {"answers": rows, "errors": {f"{m}|{c}": e for m, c, _, e in answers if e},
            "verified_non_herd_per_model": per_model, "models_with_more_when_shown_herd": more,
            "conclusion": "showing the consensus unlocks more verified non-consensus approaches" if more >= 4 else "no evidence",
            "exa_checks": sources}


# ---------------------------------------------------------------- driver
def build(session, task):
    if task.name not in INPUTS:
        raise ValueError(f"overlook has an input list only for {sorted(INPUTS)}")
    out = session.out_dir(task)
    consensus_map = read_json(out / "herd_map.json")
    families = read_json(out / "families.json")
    valid = valid_samples(out)
    all_samples = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()]
    labels = {json.loads(l)["sample_id"]: json.loads(l)["family"] for l in (out / "labels.jsonl").read_text().splitlines()}
    for s in all_samples:
        s["family"] = labels.get(s["sample_id"], UNLABELED)
    models = session.cfg["models"]["samplers"]
    start = session.ledger.new_spend
    session.progress("overlook: input audit")
    inputs, per_sample = input_audit(session, task, valid)
    session.progress("overlook: shared assumptions")
    assumptions = shared_assumptions(session, task, valid, consensus_map["top3"])
    session.progress("overlook: component grid")
    grid = component_grid(session, task, valid, families)
    minor = minority_report(valid, all_samples, models, grid)
    session.progress("overlook: latent tail")
    latent = latent_tail(session, task, families, consensus_map["top3"])
    result = {"task": task.name, "herd": {"samples": len(all_samples), "valid": len(valid), "models": models,
                                          "top3": consensus_map["top3"], "concentration_top3": consensus_map["concentration_top3"]},
              "paraphrase_test": read_json(out / "paraphrase.json"),
              "inputs": inputs, "assumptions": assumptions, "grid": grid, "minority": minor, "latent_tail": latent,
              "new_spend_usd": round(session.ledger.new_spend - start, 4)}
    write_json(out / "overlook.json", result)
    write_json(out / "overlook_audit_per_sample.json", per_sample)
    return result
