"""Isolation for running model-written code, shared by the task evaluators and by HERD's
external evaluator_cmd support.

- The child process gets a minimal environment: no API keys or other secrets.
- On macOS it runs under sandbox-exec with no network access and no file writes outside
  a throwaway scratch directory. On other platforms there is no OS sandbox (see README).
- An evaluator's result line must carry a one-time nonce that the parent sends on stdin
  and the child reads before loading the candidate, so candidate code cannot simply print
  a fake result. A candidate that deliberately inspects the evaluator's own stack frames
  could still find it; that is a known limit of running in one process.
"""
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile

RESULT_TAG = "__HERD_RESULT__"

SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{scratch}") (literal "/dev/null") (literal "/dev/stdout")
       (literal "/dev/stderr") (literal "/dev/tty") (literal "/dev/dtracehelper"))
"""


def sandboxed(cmd, scratch):
    """cmd wrapped in the OS sandbox where one is available."""
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return ["sandbox-exec", "-p", SANDBOX_PROFILE.format(scratch=scratch)] + cmd
    return cmd


def clean_env(scratch, keep_path=False):
    env = {"HOME": scratch, "TMPDIR": scratch, "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8",
           "PATH": os.environ.get("PATH", "/usr/bin:/bin") if keep_path else "/usr/bin:/bin"}
    return env


def invalid(error, **detail):
    return {"valid": False, "score": None, "detail": {"error": error, **detail}}


def run_isolated(cmd, timeout, stdin="", keep_path=False):
    """Run cmd sandboxed in a scratch directory. Returns (CompletedProcess or None on timeout)."""
    with tempfile.TemporaryDirectory() as scratch:
        scratch = os.path.realpath(scratch)
        try:
            return subprocess.run(sandboxed(cmd, scratch), input=stdin, capture_output=True, text=True,
                                  timeout=timeout, env=clean_env(scratch, keep_path), cwd=scratch)
        except subprocess.TimeoutExpired:
            return None


def evaluate_isolated(evaluator_script, candidate, timeout):
    """Run `python evaluator_script --inner candidate` in isolation and return its result."""
    nonce = secrets.token_hex(16)
    proc = run_isolated([sys.executable, str(evaluator_script), "--inner", str(os.path.realpath(candidate))],
                        timeout, stdin=nonce + "\n")
    if proc is None:
        return invalid(f"timeout after {timeout}s")
    tag = RESULT_TAG + nonce
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(tag):
            return json.loads(line[len(tag):])
    error = (proc.stderr.strip().splitlines() or ["no result"])[-1]
    return invalid(error, stderr_tail=proc.stderr[-2000:])


def read_nonce():
    """In the child: take the nonce before any candidate code is loaded."""
    nonce = sys.stdin.readline().strip()
    sys.stdin.close()
    return nonce


def emit(nonce, result):
    print(RESULT_TAG + nonce + json.dumps(result), flush=True)
