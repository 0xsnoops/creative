"""SAMPLE: send each model the task prompt verbatim, extract the code, score it if we can."""
import json
import re
from concurrent.futures import ThreadPoolExecutor

from .llm import BudgetExceeded, finish_reason, message_text

FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)


def extract_code(text, entrypoint=None):
    """Longest fenced block that defines the entrypoint (any block if entrypoint is None), else None."""
    blocks = FENCE.findall(text)
    if entrypoint:
        defines = re.compile(rf"^\s*(def|class)\s+{re.escape(entrypoint)}\b", re.M)
        blocks = [b for b in blocks if defines.search(b)]
    return max(blocks, key=len) if blocks else None


def sample_id(model, idx):
    return f"{model.replace('/', '_')}-{idx:03d}"


def run_sampling(task, llm, cfg, out_dir, n_per_model, progress=print):
    models, s = cfg["models"]["samplers"], cfg["sampling"]
    llm.check_models(models)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content": task.prompt}]
    jobs = [(m, i) for m in models for i in range(n_per_model)]
    done = []

    def one(job):
        model, idx = job
        record = {"sample_id": sample_id(model, idx), "model": model, "idx": idx}
        try:
            entry = llm.chat(model, messages, temperature=s["temperature"],
                             max_tokens=s["max_tokens"], sample_idx=idx)
        except BudgetExceeded as e:
            return {**record, "status": "budget_exceeded", "error": str(e)}
        except Exception as e:
            return {**record, "status": "api_error", "error": str(e)}
        finally:
            done.append(job)
            progress(f"sample {len(done)}/{len(jobs)}")
        text = message_text(entry)
        usage = entry["response"].get("usage") or {}
        record.update(cost_usd=entry["cost_usd"], cached=entry["cached"],
                      finish_reason=finish_reason(entry),
                      completion_tokens=usage.get("completion_tokens"),
                      text_path=f"samples/{record['sample_id']}.md")
        (out_dir / record["text_path"]).write_text(text)
        code = extract_code(text, task.entrypoint)
        if code is None:
            return {**record, "status": "no_code"}
        record["code_path"] = f"samples/{record['sample_id']}.py"
        (out_dir / record["code_path"]).write_text(code)
        if not task.evaluator:
            return {**record, "status": "unscored"}
        result = task.evaluate(out_dir / record["code_path"])
        return {**record, "status": "valid" if result["valid"] else "invalid",
                "score": result["score"], "eval_error": result["detail"].get("error")}

    with ThreadPoolExecutor(s["concurrency"]) as pool:
        records = list(pool.map(one, jobs))
    with (out_dir / "samples.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


def summarize(records):
    by_model = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)
    lines = [f"{'model':34} {'n':>3} {'valid':>5} {'no_code':>7} {'invalid':>7} {'err':>3} "
             f"{'best':>8} {'median':>8} {'cost $':>8}"]
    for model, rs in by_model.items():
        scores = sorted(r["score"] for r in rs if r["status"] == "valid")
        count = lambda status: sum(r["status"] == status for r in rs)
        best = f"{scores[0]:.4f}" if scores else "-"
        median = f"{scores[len(scores) // 2]:.4f}" if scores else "-"
        cost = sum(r.get("cost_usd") or 0 for r in rs)
        lines.append(f"{model:34} {len(rs):>3} {count('valid'):>5} {count('no_code'):>7} "
                     f"{count('invalid'):>7} {count('api_error') + count('budget_exceeded'):>3} "
                     f"{best:>8} {median:>8} {cost:>8.4f}")
    return "\n".join(lines)
