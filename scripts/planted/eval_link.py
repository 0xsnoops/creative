"""Evaluator for the planted-answer control (scripts/planted_control.py), used as evaluator_cmd:

    python eval_link.py --data /abs/path/dev.jsonl <candidate.py>

The candidate defines predict(record) -> delay in ms. Score: mean absolute error over all records
(lower is better). detail.cases: one case per link, loss = that link's mean absolute error, so
diverge_escape can see whether attempts fail on the same links."""
import argparse
import importlib.util
import json
import math
import sys


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("candidate")
    args = p.parse_args(argv)
    records = [json.loads(line) for line in open(args.data) if line.strip()]
    try:
        spec = importlib.util.spec_from_file_location("candidate", args.candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        errors, by_link = [], {}
        for r in records:
            got = float(module.predict(dict(r["input"])))
            if not math.isfinite(got):
                raise ValueError(f"non-finite prediction for {r['input']}")
            e = abs(got - r["expected"])
            errors.append(e)
            by_link.setdefault(r["input"]["link_id"], []).append(e)
    except Exception as e:  # an invalid candidate, not a crash
        print(json.dumps({"valid": False, "score": None, "detail": {"error": f"{type(e).__name__}: {e}"[:400]}}))
        return
    cases = [{"id": f"link-{k}", "loss": sum(v) / len(v)} for k, v in sorted(by_link.items())]
    print(json.dumps({"valid": True, "score": sum(errors) / len(errors), "detail": {"cases": cases}}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
