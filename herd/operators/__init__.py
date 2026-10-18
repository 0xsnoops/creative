"""Generation operators. Each turns (task, budget_usd) into candidate solutions, each tagged
with the family it was meant to implement (None for plain and hot, which get the bare
prompt). All operators use the same implementer model, implementation template and evaluator.

An operator is a generator of jobs. A job is a zero-argument callable that makes one
implementer call and evaluates the code, or an already finished candidate dict. Planning
calls (search, verbalized sampling) happen inside the generator and are billed to the
same budget."""
import itertools
import json
from concurrent.futures import ThreadPoolExecutor

from ..llm import BudgetExceeded, message_text
from ..pipeline import Herd, ensure_map
from ..reality import IMPLEMENT_TEMPLATE, Evaluator
from ..sample import extract_code


class OperatorContext:
    def __init__(self, name, task, budget_usd, seed, study, progress):
        self.name, self.task, self.seed, self.study, self.progress = name, task, seed, study, progress
        self.herd = Herd(budget_usd, progress, bill_cached=True)
        self.llm, self.exa, self.cfg = self.herd.llm, self.herd.exa, self.herd.cfg
        self.evaluator = Evaluator(task)
        self.out_dir = self.herd.out_dir(task) / "bench" / f"seed{seed}" / name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.notes = {}
        self._next_idx = itertools.count()

    def key(self, *parts):
        """Cache-key tag, so operators and repeats never share responses."""
        return ":".join(str(p) for p in ("bench", self.name, self.seed, *parts))

    def job(self, prompt, temperature=1.0, family=None):
        """One implementer call on prompt, then evaluation. The index is fixed when the job is
        created, so cache keys do not depend on thread timing."""
        idx = next(self._next_idx)

        def run():
            entry = self.llm.chat(self.cfg["models"]["implementer"], [{"role": "user", "content": prompt}],
                                  temperature=temperature, max_tokens=self.cfg["reality_check"]["max_tokens"],
                                  sample_idx=self.key(idx))
            candidate = {"operator": self.name, "seed": self.seed, "idx": idx, "temperature": temperature,
                         "intended_family": family and family["family"],
                         "mechanism": family and family["one_line_mechanism"],
                         "source_urls": family["source_urls"] if family else []}
            code = extract_code(message_text(entry), self.task.entrypoint)
            if code is None:
                return {**candidate, "code_path": None, "valid": False, "score": None, "error": "no code in reply"}
            path = self.out_dir / f"{idx:03d}.py"
            path.write_text(code)
            result = self.evaluator(path)
            return {**candidate, "code_path": str(path.relative_to(self.herd.out_dir(self.task))),
                    "valid": result["valid"], "score": result["score"], "error": result["detail"].get("error")}
        return run

    def family_job(self, family):
        """Implement a named family with the shared template (the reality-check prompt)."""
        prompt = IMPLEMENT_TEMPLATE.format(task=self.task.prompt, family=family["family"],
                                           mechanism=family["one_line_mechanism"],
                                           source_summary=family["source_summary"])
        return self.job(prompt, 1.0, family)

    def run_now(self, jobs):
        """Run jobs in parallel and return their candidates (used for calibration)."""
        with ThreadPoolExecutor(len(jobs)) as pool:
            return list(pool.map(lambda j: j(), jobs))


def run_jobs(ctx, jobs):
    """Pull jobs until the budget or max_candidates runs out. A call refused only because
    other calls were in flight is retried with less parallelism; a call refused with nothing
    else in flight means the budget is spent."""
    max_n, width = ctx.cfg["bench"]["max_candidates"], ctx.cfg["sampling"]["concurrency"]
    candidates, pending = [], []
    while len(candidates) < max_n:
        wave = pending[:width]
        pending = pending[width:]
        while len(wave) < min(width, max_n - len(candidates)):
            try:
                wave.append(next(jobs))
            except (StopIteration, BudgetExceeded):
                break
        if not wave:
            break
        refused = []
        with ThreadPoolExecutor(len(wave)) as pool:
            futures = [(j, pool.submit(j) if callable(j) else None) for j in wave]
            for j, f in futures:
                if f is None:
                    candidates.append(j)
                    continue
                try:
                    candidates.append(f.result())
                except BudgetExceeded:
                    refused.append(j)
                except Exception as e:
                    candidates.append({"operator": ctx.name, "seed": ctx.seed, "valid": False, "score": None,
                                       "code_path": None, "error": f"{type(e).__name__}: {e}"})
        ctx.progress(f"{ctx.name} seed {ctx.seed}: {len(candidates)} candidates, "
                     f"${ctx.herd.ledger.spent_this_run:.3f} spent")
        if refused:
            if len(wave) == 1:
                break  # refused alone: out of budget
            width = max(1, width // 2)
            pending = refused + pending
    return candidates[:max_n]


def run_operator(name, task, budget_usd, seed=0, study=None, progress=print):
    """Run one operator at one budget. Returns {operator, seed, spent_usd, new_spend_usd, notes,
    candidates}; spent_usd counts cache hits at their original cost, so reruns match."""
    from . import blind_spot, hot, plain, transplant, verbalized
    operators = {"plain": plain.jobs, "hot": hot.jobs, "verbalized": verbalized.jobs,
                 "blind_spot": blind_spot.jobs, "transplant": transplant.jobs}
    if study is None:
        study = ensure_map(Herd(progress=progress), task)
    ctx = OperatorContext(name, task, budget_usd, seed, study, progress)
    candidates = run_jobs(ctx, operators[name](ctx))
    run = {"operator": name, "seed": seed, "budget_usd": budget_usd,
           "spent_usd": round(ctx.herd.ledger.spent_this_run, 6),
           "new_spend_usd": round(ctx.herd.ledger.new_spend, 6), "notes": ctx.notes, "candidates": candidates}
    (ctx.out_dir / "run.json").write_text(json.dumps(run, indent=2))
    return run
