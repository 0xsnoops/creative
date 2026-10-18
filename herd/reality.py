"""REALITY CHECK: implement a family K times with one shared prompt template, then evaluate."""
import hashlib
import json
import re
import shlex
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor

from .llm import BudgetExceeded, message_text
from .sample import extract_code

# The same template for every family, herd or blind spot.
IMPLEMENT_TEMPLATE = """{task}

---
Use this specific family of approaches:

Family: {family}
Mechanism: {mechanism}
Background: {source_summary}

Implement this approach faithfully and make it as strong as you can within the family.
Reply with your complete solution in a single Python code block."""


class Evaluator:
    """The task's built-in evaluator, or a shell command that takes a code path as its last
    argument and prints {"valid": bool, "score": float, ...} as JSON on its last line."""

    def __init__(self, task, command=None, lower_is_better=True, timeout=600):
        if not command and not task.evaluator:
            raise ValueError("this task has no built-in evaluator; pass evaluator_cmd")
        self.task, self.command, self.timeout = task, command, timeout
        self.lower_is_better = lower_is_better if command else True
        self.key = f"cmd:{command}" if command else f"builtin:{task.name}"
        self.id = hashlib.sha256(self.key.encode()).hexdigest()[:10]

    def __call__(self, path):
        if not self.command:
            return self.task.evaluate(path)
        try:
            proc = subprocess.run(shlex.split(self.command) + [str(path)], capture_output=True,
                                  text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return {"valid": False, "score": None, "detail": {"error": f"timeout after {self.timeout}s"}}
        for line in reversed(proc.stdout.strip().splitlines()):
            try:
                result = json.loads(line)
            except ValueError:
                continue
            if isinstance(result, dict) and "valid" in result:
                return {"valid": bool(result["valid"]), "score": result.get("score"),
                        "detail": result.get("detail", {})}
        return {"valid": False, "score": None,
                "detail": {"error": f"no JSON result (exit {proc.returncode})", "stderr_tail": proc.stderr[-1000:]}}

    def best(self, scores):
        return min(scores) if self.lower_is_better else max(scores)

    def beats(self, a, b):
        return a < b if self.lower_is_better else a > b


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]


def reality_check(task, family, evaluator, llm, cfg, out_dir, k, progress=print):
    """Return {family, k, n_valid, n_invalid, best_score, median_score, best_code_path, implementations}."""
    prompt = IMPLEMENT_TEMPLATE.format(task=task.prompt, family=family["family"],
                                       mechanism=family["one_line_mechanism"],
                                       source_summary=family["source_summary"])
    code_dir = out_dir / "reality" / evaluator.id
    code_dir.mkdir(parents=True, exist_ok=True)

    def one(i):
        entry = llm.chat(cfg["models"]["implementer"], [{"role": "user", "content": prompt}],
                         temperature=1.0, max_tokens=cfg["reality_check"]["max_tokens"], sample_idx=i)
        code = extract_code(message_text(entry), task.entrypoint)
        if code is None:
            return {"path": None, "valid": False, "score": None, "error": "no code in reply"}
        path = code_dir / f"{slug(family['family'])}-{i}.py"
        path.write_text(code)
        result = evaluator(path)
        progress(f"reality {family['family']}: implementation {i + 1}/{k} "
                 f"{'score ' + format(result['score'], '.4g') if result['valid'] else 'invalid'}")
        return {"path": str(path.relative_to(out_dir)), "valid": result["valid"], "score": result["score"],
                "error": result["detail"].get("error")}

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
    best = None
    if valid:
        best_score = evaluator.best([m["score"] for m in valid])
        best = next(m for m in valid if m["score"] == best_score)
    return {"family": family["family"], "k": k, "n_valid": len(valid),
            "n_invalid": len(implementations) - len(valid),
            "best_score": best["score"] if best else None,
            "median_score": statistics.median(m["score"] for m in valid) if valid else None,
            "best_code_path": best["path"] if best else None,
            "implementations": implementations}
