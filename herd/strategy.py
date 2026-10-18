"""Strategy fingerprints: what a piece of code decides from, not how it is written. Two attempts
can differ line by line and still read the same inputs, keep the same state, rank by the same
signal and fail on the same cases; that is the "same strategy" this module measures.

The static parts (inputs, state) come from overlook's AST audit; the decision rule, assumptions
and mechanism family come from an optional extractor (an LLM reading the code); behavior comes
from herd.fingerprint, run in the sandbox; weak regimes come from dev results against the
best baseline."""
import ast
import difflib
import hashlib
import os
import re
import tempfile

from . import fingerprint as behavioral
from .llm import ask_json
from .operators.blind_spot_feedback import gaps
from .overlook import AUDITORS, INPUTS, MUTATORS, extractor

STATE_MUTATORS = MUTATORS | {"pop", "popitem", "popleft", "remove", "discard", "clear", "heappop",
                             "heapreplace", "heappushpop", "insort", "sort", "reverse"}
STOPWORDS = {"a", "an", "the", "of", "to", "and", "or", "is", "are", "in", "on", "for", "by", "it", "its",
             "that", "this", "with", "as", "be", "not", "no", "each", "every", "any", "all", "only"}
EXTRACT_FIELDS = ("primary_signal", "secondary_signal", "assumptions", "mechanism_family")

EXTRACT_PROMPT = """Read the CODE below (not its comments, docstrings or names) and describe the
strategy it implements for this task.

<task>
{task}
</task>

Inputs available at decision time:
{inputs}

<code>
{code}
</code>

Return JSON with:
- "primary_signal": the input or quantity that mainly decides the choice ({naming})
- "secondary_signal": what breaks ties or adjusts the primary signal, or null
- "assumptions": short statements of what the code takes for granted or never does (at most 6)
- "mechanism_family": a short standard name for the mechanism (2-4 words, e.g. "best fit", "LRU")

{{"primary_signal": "...", "secondary_signal": null, "assumptions": ["..."], "mechanism_family": "..."}}"""


# ---------------------------------------------------------------- static parts
def input_usage(task, code, extracted):
    auditor = AUDITORS.get(task.name)
    if auditor is None:
        return {"inputs_used": list(extracted.get("inputs_used") or []), "inputs_unused": [], "inputs_ambiguous": []}
    try:
        audit = auditor(code)
    except SyntaxError:
        audit = {k: ("ambiguous", "does not parse") for k in INPUTS[task.name]}
    return {f"inputs_{status}": [k for k, (s, _) in audit.items() if s == status]
            for status in ("used", "unused", "ambiguous")}


def _base(node):
    """The state a written expression belongs to: a bare name or 'self.attr'."""
    while isinstance(node, (ast.Subscript, ast.Attribute, ast.Starred)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            return f"self.{node.attr}"
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _writes(func):
    """(written state, bare local names) of one function. Bare names count as writes only when
    declared global; subscript, attribute and mutator writes count on their base."""
    declared = {g for n in ast.walk(func) if isinstance(n, ast.Global) for g in n.names}
    written, local = set(), set()
    for n in ast.walk(func):
        targets, binds = [], True
        if isinstance(n, (ast.Assign, ast.Delete)):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in STATE_MUTATORS:
            binds = False
            if isinstance(n.func.value, ast.Name) and n.func.value.id in ("heapq", "bisect"):
                targets = n.args[:1]
            else:
                targets = [n.func.value]
        for t in targets:
            for e in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                if isinstance(e, ast.Name) and binds:
                    (written if e.id in declared else local).add(e.id)
                elif (b := _base(e)) is not None:
                    written.add(b)
    return written, local - declared


def _reads(func):
    out = set()
    for n in ast.walk(func):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) or isinstance(n, ast.AugAssign):
            out.add(n.id if isinstance(n, ast.Name) else _base(n.target))
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
            out.add(f"self.{n.attr}")
    return out - {None}


def state_of(code):
    """Module-level names, or self attributes outside __init__, written in one call and read in a
    later one (any call of the module's functions or the class's methods)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"persistent": False, "features": []}
    module_names = set()
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                module_names |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    methods = [m for c in tree.body if isinstance(c, ast.ClassDef) for m in c.body
               if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name != "__init__"]
    written, read = set(), set()
    for f in funcs + methods:
        w, local = _writes(f)
        written |= {x for x in w if x.startswith("self.") or (x in module_names and x not in local)}
        read |= _reads(f) - local
    features = sorted(written & read)
    return {"persistent": bool(features), "features": features}


def weak_regimes(dev_detail, baseline_detail):
    """Cases where the candidate is worse than the best baseline, worst first, named without
    their scores so they compare across attempts."""
    worse = sorted((g for g in gaps(dev_detail, baseline_detail) if g[0] > 0), key=lambda g: -g[0])
    return [text.rsplit(": ", 1)[0] for _, text in worse]


def behavior_signature(task, code):
    if not behavioral.supported(task):
        return None
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w") as f:
            f.write(code)
        return behavioral.fingerprints(task, {"x": path})["x"]


# ---------------------------------------------------------------- fingerprint
def fingerprint(task, code, extract=None, behavior=False, dev_detail=None, baseline_detail=None):
    extracted = _clean(extract(code)) if extract else _clean({})
    return {**input_usage(task, code, extracted),
            "state": state_of(code),
            "decision_rule": {"primary_signal": extracted["primary_signal"],
                              "secondary_signal": extracted["secondary_signal"]},
            "assumptions": extracted["assumptions"],
            "mechanism_family": extracted["mechanism_family"],
            "behavior_signature": behavior_signature(task, code) if behavior else None,
            "failure_signature": {"weak_regimes": weak_regimes(dev_detail, baseline_detail)}}


def _text(value):
    if value is None or not str(value).strip() or str(value).strip().lower() in ("none", "null", "n/a"):
        return None
    return str(value).strip()


def _clean(reply, inputs=()):
    reply = reply if isinstance(reply, dict) else {}
    canonical = {k.lower(): k for k in inputs}
    signal = lambda v: canonical.get(_text(v).lower(), _text(v)) if _text(v) else None
    assumptions = reply.get("assumptions") or []
    return {**{k: v for k, v in reply.items() if k not in EXTRACT_FIELDS},
            "primary_signal": signal(reply.get("primary_signal")),
            "secondary_signal": signal(reply.get("secondary_signal")),
            "assumptions": [a for a in map(_text, assumptions if isinstance(assumptions, list) else [assumptions]) if a],
            "mechanism_family": _text(reply.get("mechanism_family"))}


def llm_extractor(herd, task):
    """extract(code) backed by the extractor model (the labeler if none is configured)."""
    model = extractor(herd)
    inputs = INPUTS.get(task.name, {})
    listing = "\n".join(f"- {k}: {v}" for k, v in inputs.items()) or "- (not listed; name them in free text)"
    naming = "use one of the input names above where possible" if inputs else "free text"

    def extract(code):
        prompt = EXTRACT_PROMPT.format(task=task.prompt, inputs=listing, code=code, naming=naming)
        key = f"strategy:{hashlib.sha256(code.encode()).hexdigest()[:12]}"
        try:
            reply = ask_json(herd.llm, model, prompt, key=key, required=("primary_signal", "assumptions",
                                                                         "mechanism_family"))
        except ValueError:
            reply = {}
        return {k: v for k, v in _clean(reply, inputs).items() if k in EXTRACT_FIELDS}
    return extract


# ---------------------------------------------------------------- similarity
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else None


def same(a, b):
    if a is None and b is None:
        return None
    return float(a is not None and b is not None and str(a).strip().lower() == str(b).strip().lower())


def tokens(texts):
    return {w for t in texts for w in re.findall(r"[a-z0-9_]+", t.lower())} - STOPWORDS


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def similarity(a, b):
    """Strategy similarity in [0, 1]; parts that neither side has are None and left out."""
    sa, sb = a.get("state") or {}, b.get("state") or {}
    ra, rb = a.get("decision_rule") or {}, b.get("decision_rule") or {}
    fa, fb = a.get("behavior_signature"), b.get("behavior_signature")
    parts = {
        "inputs": jaccard(a.get("inputs_used") or [], b.get("inputs_used") or []),
        "state": mean([float(bool(sa.get("persistent")) == bool(sb.get("persistent"))),
                       jaccard(sa.get("features") or [], sb.get("features") or [])]),
        "decision_rule": mean([same(ra.get("primary_signal"), rb.get("primary_signal")),
                               same(ra.get("secondary_signal"), rb.get("secondary_signal"))]),
        "assumptions": jaccard(tokens(a.get("assumptions") or []), tokens(b.get("assumptions") or [])),
        "mechanism_family": same(a.get("mechanism_family"), b.get("mechanism_family")),
        "behavior": 1 - behavioral.distance(fa, fb) if fa is not None and fb is not None else None,
    }
    return {"overall": mean(parts.values()) or 0.0, "parts": parts}


def code_similarity(code_a, code_b):
    norm = lambda code: [" ".join(l.split()) for l in code.splitlines() if l.strip()]
    return difflib.SequenceMatcher(None, norm(code_a), norm(code_b), autojunk=False).ratio()
