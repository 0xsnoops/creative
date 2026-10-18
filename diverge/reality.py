"""REALITY CHECK: implement a family K times with one shared prompt template, then evaluate."""
import hashlib
import importlib.util
import json
import re
import shlex
import statistics
from concurrent.futures import ThreadPoolExecutor

from .llm import BudgetExceeded, message_text
from .sample import extract_code
from .task import TASKS_DIR

_spec = importlib.util.spec_from_file_location("diverge_sandbox", TASKS_DIR / "_sandbox.py")
sandbox = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sandbox)

# The same template for every family, consensus or blind spot.
IMPLEMENT_TEMPLATE = """{task}

---
Use this specific family of approaches:

Family: {family}
Mechanism: {mechanism}
Background: {source_summary}

Implement this approach faithfully and make it as strong as you can within the family.
Reply with your complete solution in a single Python code block."""

# Hybrid-allowed variant (blind_spot_hybrid and friends). The family is the core mechanism, but
# borrowing from other mechanisms is welcome; the task interface stays as the task states it.
# IMPLEMENT_TEMPLATE itself is untouched so that cached runs replay byte for byte.
HYBRID_TEMPLATE = """{task}

---
Build your solution around this family of approaches:

Family: {family}
Mechanism: {mechanism}
Background: {source_summary}

Make this family the core mechanism of your solution. You may add components from other mechanisms (a second signal, a small auxiliary structure, an admission or tie-break rule borrowed from elsewhere) wherever they make the solution stronger: a hybrid is welcome as long as the family above stays at its heart. Keep the task's interface exactly as the task states it.
Reply with your complete solution in a single Python code block."""

HYBRID_MARKER = "Make this family the core mechanism"


def dev_detail(result):
    """Evaluator detail worth keeping from a dev run: per-instance results for valid code, the
    (already redacted) error for invalid code. Only ever called on dev-split results."""
    detail = result.get("detail") or {}
    if not result["valid"]:
        return {"split": "dev", "error": detail.get("error")}
    return {k: v for k, v in detail.items() if k != "stderr_tail"}


class Evaluator:
    """The task's built-in evaluator, or a command that takes a code path as its last argument
    and prints {"valid": bool, "score": float, ...} as JSON on its last line. Commands run with
    the same isolation as the built-in evaluators: no secrets in the environment, and on macOS
    no network and no writes outside a scratch working directory (so use absolute paths)."""

    def __init__(self, task, command=None, lower_is_better=True, timeout=600):
        if not command and not task.evaluator:
            raise ValueError("this task has no built-in evaluator; pass evaluator_cmd")
        self.task, self.command, self.timeout = task, command, timeout
        self.lower_is_better = lower_is_better if command else True
        # Saved scores are only reused for the same direction, task prompt and evaluator code.
        source = command if command else (TASKS_DIR / task.name / "evaluator.py").read_text()
        inputs = hashlib.sha256("\0".join([task.prompt, source]).encode()).hexdigest()[:12]
        direction = "min" if self.lower_is_better else "max"
        self.key = f"{'cmd:' + command if command else 'builtin:' + task.name}|{direction}|{inputs}"
        self.id = hashlib.sha256(self.key.encode()).hexdigest()[:10]
        self.has_heldout = not command  # external commands have a single data split

    def __call__(self, path, split="dev"):
        """Score on the dev split (all scoring and selection) or, for built-in tasks, on the
        held-out split (final numbers for an already selected candidate)."""
        if not self.command:
            return self.task.evaluate(path, split)
        if split != "dev":
            raise ValueError("external evaluators have no held-out split")
        proc = sandbox.run_isolated(shlex.split(self.command) + [str(path.resolve())], self.timeout,
                                    keep_path=True)
        if proc is None:
            return sandbox.invalid(f"timeout after {self.timeout}s")
        for line in reversed(proc.stdout.strip().splitlines()):
            try:
                result = json.loads(line)
            except ValueError:
                continue
            if isinstance(result, dict) and "valid" in result:
                detail = json.loads(sandbox.redact(json.dumps(result.get("detail", {}))))
                return {"valid": bool(result["valid"]), "score": result.get("score"), "detail": detail}
        return sandbox.invalid(f"no JSON result (exit {proc.returncode})", stderr_tail=proc.stderr[-1000:])

    def final(self, path, dev_score):
        """(score, split) to report for a candidate selected on dev: its held-out score when the
        task has a held-out split (None if it fails there), else its dev score."""
        if not self.has_heldout:
            return dev_score, "dev"
        result = self(path, "heldout")
        return (result["score"] if result["valid"] else None), "heldout"

    def best(self, scores):
        return min(scores) if self.lower_is_better else max(scores)

    def beats(self, a, b):
        return a < b if self.lower_is_better else a > b


def slug(name):
    """File-name-safe family name; the hash suffix keeps long names that share a prefix apart."""
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]
    return f"{base}-{hashlib.sha256(name.encode()).hexdigest()[:6]}"


def implementation_prompt(task, family, extra=""):
    """The shared template; extra (a contrast list or feedback note) goes right before the final
    instruction, so with no extra the prompt is exactly the shared one."""
    prompt = IMPLEMENT_TEMPLATE.format(task=task.prompt, family=family["family"],
                                       mechanism=family["one_line_mechanism"],
                                       source_summary=family["source_summary"])
    if extra:
        marker = "Implement this approach faithfully"
        head, tail = prompt.rsplit(marker, 1)
        prompt = f"{head}{extra.strip()}\n\n{marker}{tail}"
    return prompt


def hybrid_prompt(task, family, extra=""):
    """HYBRID_TEMPLATE filled in the same way as implementation_prompt; extra goes right before
    the final instruction."""
    prompt = HYBRID_TEMPLATE.format(task=task.prompt, family=family["family"],
                                    mechanism=family["one_line_mechanism"],
                                    source_summary=family["source_summary"])
    if extra:
        head, tail = prompt.rsplit(HYBRID_MARKER, 1)
        prompt = f"{head}{extra.strip()}\n\n{HYBRID_MARKER}{tail}"
    return prompt


def reality_check(task, family, evaluator, llm, cfg, out_dir, k, progress=print, extra=""):
    """Return {family, k, n_valid, n_invalid, best_score, median_score, best_code_path, implementations}."""
    prompt = implementation_prompt(task, family, extra)
    code_dir = out_dir / "reality" / evaluator.id / ("contrast" if extra else "plain")
    code_dir.mkdir(parents=True, exist_ok=True)

    def one(i):
        entry = llm.chat(cfg["models"]["implementer"], [{"role": "user", "content": prompt}],
                         temperature=1.0, max_tokens=cfg["reality_check"]["max_tokens"], sample_idx=i)
        code = extract_code(message_text(entry), task.entrypoint)
        if code is None:
            return {"path": None, "valid": False, "score": None, "error": "no code in reply", "dev_detail": None}
        path = code_dir / f"{slug(family['family'])}-{i}.py"
        path.write_text(code)
        result = evaluator(path)
        progress(f"reality {family['family']}: implementation {i + 1}/{k} "
                 f"{'score ' + format(result['score'], '.4g') if result['valid'] else 'invalid'}")
        return {"path": str(path.relative_to(out_dir)), "valid": result["valid"], "score": result["score"],
                "error": result["detail"].get("error"), "dev_detail": dev_detail(result)}

    with ThreadPoolExecutor(k) as pool:
        futures = [pool.submit(one, i) for i in range(k)]
        implementations = []
        for f in futures:
            try:
                implementations.append(f.result())
            except BudgetExceeded:
                raise
            except Exception as e:
                implementations.append({"path": None, "valid": False, "score": None, "error": str(e)})
    valid = [m for m in implementations if m["valid"]]
    best, final_score, final_split = None, None, None
    if valid:
        best_score = evaluator.best([m["score"] for m in valid])
        best = next(m for m in valid if m["score"] == best_score)
        # Selected on dev; the number used to compare families comes from data it was not selected on.
        final_score, final_split = evaluator.final(out_dir / best["path"], best["score"])
    return {"family": family["family"], "k": k, "n_valid": len(valid),
            "n_invalid": len(implementations) - len(valid),
            "best_score": best["score"] if best else None,  # dev score of the selected implementation
            "median_score": statistics.median(m["score"] for m in valid) if valid else None,
            "final_score": final_score, "final_split": final_split,
            "best_code_path": best["path"] if best else None,
            "implementations": implementations}
