"""An evaluator_cmd template that scores a candidate by replaying recorded data (a backtest, a log
replay, a "replay gate") instead of a live system.

    python /abs/path/examples/replay_evaluator.py --data /abs/path/recorded.jsonl <candidate.py>

Each line of the data file is {"input": ..., "expected": number}. The candidate defines
predict(input) -> number (rename with --entrypoint). The score is the mean absolute error over all
records (lower is better); the last stdout line is {"valid": bool, "score": float, "detail": {...}},
which is what DIVERGE reads. Adapt load/score to your own replay; keep the last-line contract.

Inside DIVERGE's sandbox there is no network, writes only go to a scratch working directory, the
environment is emptied (so pass the data path as a flag, never an environment variable), and .env
files and credential directories cannot be read; any other absolute path is readable. The candidate
runs in this process, so this template is for code you are willing to run, like any evaluator_cmd.
"""
import argparse
import importlib.util
import json
import math
import sys


def result(valid, score=None, **detail):
    print(json.dumps({"valid": valid, "score": score, "detail": detail}))
    sys.stdout.flush()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="absolute path to the recorded JSONL file")
    p.add_argument("--entrypoint", default="predict")
    p.add_argument("candidate")
    args = p.parse_args(argv)
    try:
        records = [json.loads(line) for line in open(args.data) if line.strip()]
    except (OSError, ValueError) as e:
        return result(False, error=f"cannot read --data: {e}")
    if not records:
        return result(False, error="--data has no records")
    try:
        spec = importlib.util.spec_from_file_location("candidate", args.candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, args.entrypoint)
        errors = []
        for r in records:
            got = float(fn(r["input"]))
            if not math.isfinite(got):
                return result(False, error=f"non-finite prediction for input {r['input']!r}")
            errors.append(abs(got - float(r["expected"])))
    except Exception as e:  # any candidate failure is an invalid result, not a crash
        return result(False, error=f"{type(e).__name__}: {e}")
    worst = sorted(range(len(errors)), key=lambda i: -errors[i])[:3]
    result(True, sum(errors) / len(errors), n_records=len(records), worst_records=worst)


if __name__ == "__main__":
    main()
