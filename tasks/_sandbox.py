"""Isolation for running model-written code, shared by the task evaluators and by DIVERGE's
external evaluator_cmd support.

- The child process gets a minimal environment: no API keys or other secrets.
- On macOS it runs under sandbox-exec with no network access and no file writes outside
  a throwaway scratch directory. Built-in evaluators also get an allowlist for reads under
  the home directory (Python, the venv, tasks/ and the candidate file only), so a candidate
  cannot read the repo's .env or ~/.ssh, ~/.config, ~/.aws. External evaluator commands may
  need the user's own files, so they get a denylist instead: any .env file and the usual
  credential directories. On other platforms there is no OS sandbox (see README).
- Error text that leaves the sandbox is redacted: known secret values and key-shaped
  strings are replaced, so a candidate that does find a secret cannot report it back.
- An evaluator's result line must carry a one-time nonce that the parent sends on stdin
  and the child reads before loading the candidate, so candidate code cannot simply print
  a fake result. A candidate that deliberately inspects the evaluator's own stack frames
  could still find it; that is a known limit of running in one process.
"""
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile

RESULT_TAG = "__DIVERGE_RESULT__"

REPO = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
HOME = os.path.realpath(os.path.expanduser("~"))
CREDENTIAL_DIRS = [".ssh", ".config", ".aws", ".gnupg", ".netrc", ".docker", ".kube", ".git-credentials",
                   ".npmrc", ".pypirc", ".azure", ".gcloud", "Library/Keychains"]

SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{scratch}") (literal "/dev/null") (literal "/dev/stdout")
       (literal "/dev/stderr") (literal "/dev/tty") (literal "/dev/dtracehelper"))
{reads}
"""


def _sb(path):
    return path.replace("\\", "\\\\").replace('"', '\\"')


def read_rules(scratch, allow=None):
    """allow=None: deny .env files and credential directories (external commands).
    allow=[paths]: deny everything under HOME except these paths (built-in evaluators)."""
    if allow is None:
        denied = " ".join(f'(subpath "{_sb(os.path.join(HOME, d))}")' for d in CREDENTIAL_DIRS)
        return f'(deny file-read-data {denied} (regex #"/\\.env$"))'
    allowed = " ".join(f'(subpath "{_sb(os.path.realpath(p))}")' for p in [*allow, scratch])
    # Later rules win: deny HOME, re-allow the needed paths, then deny .env files everywhere.
    return (f'(deny file-read-data (subpath "{_sb(HOME)}"))\n(allow file-read-data {allowed})\n'
            f'(deny file-read-data (regex #"/\\.env$"))')


def sandboxed(cmd, scratch, allow=None):
    """cmd wrapped in the OS sandbox where one is available."""
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        profile = SANDBOX_PROFILE.format(scratch=_sb(scratch), reads=read_rules(scratch, allow))
        return ["sandbox-exec", "-p", profile] + cmd
    return cmd


SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)
SECRET_PATTERNS = [r"sk-or-v1-[0-9a-f]{16,}", r"\bsk-[A-Za-z0-9_-]{20,}", r"github_pat_[A-Za-z0-9_]{20,}",
                   r"\bgh[pousr]_[A-Za-z0-9]{20,}", r"\bAKIA[0-9A-Z]{16}\b", r"\bxox[abpr]-[A-Za-z0-9-]{10,}",
                   r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
                   r"(?i)\b(api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?[^\s'\"]{6,}"]


def secret_values():
    """Values of secret-looking environment variables and of the repo's .env (read by the
    trusted parent process, never by the candidate)."""
    values = {v for k, v in os.environ.items() if SECRET_NAME.search(k) and len(v) >= 8}
    env_file = os.path.join(REPO, ".env")
    try:  # unreadable inside an outer sandbox (an evaluator_cmd that calls this evaluator)
        env_lines = open(env_file).read().splitlines() if os.path.exists(env_file) else []
    except OSError:
        env_lines = []
    if env_lines:
        for line in env_lines:
            if "=" in line and not line.lstrip().startswith("#"):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if len(value) >= 8:
                    values.add(value)
    return sorted(values, key=len, reverse=True)


def redact(text):
    if not text:
        return text
    for value in secret_values():
        text = text.replace(value, "[REDACTED]")
    for pattern in SECRET_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


def clean_env(scratch, keep_path=False):
    env = {"HOME": scratch, "TMPDIR": scratch, "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8",
           "PATH": os.environ.get("PATH", "/usr/bin:/bin") if keep_path else "/usr/bin:/bin"}
    return env


def invalid(error, **detail):
    detail = {k: redact(v) if isinstance(v, str) else v for k, v in detail.items()}
    return {"valid": False, "score": None, "detail": {"error": redact(error), **detail}}


def run_isolated(cmd, timeout, stdin="", keep_path=False, allow=None):
    """Run cmd sandboxed in a scratch directory. Returns a CompletedProcess, or None on timeout.
    Callers must pass any text from it through redact() before storing or showing it."""
    with tempfile.TemporaryDirectory() as scratch:
        scratch = os.path.realpath(scratch)
        try:
            return subprocess.run(sandboxed(cmd, scratch, allow), input=stdin, capture_output=True, text=True,
                                  timeout=timeout, env=clean_env(scratch, keep_path), cwd=scratch)
        except subprocess.TimeoutExpired:
            return None


def evaluate_isolated(evaluator_script, candidate, timeout, args=(), payload=None):
    """Run `python evaluator_script --inner candidate *args` in isolation and return its result.
    payload (JSON-serializable, e.g. probe cases) follows the nonce line on stdin; it adds no
    read permissions. The child takes it with read_nonce_and_payload()."""
    nonce = secrets.token_hex(16)
    candidate = os.path.realpath(candidate)
    # Reads under HOME: only Python itself, the venv, tasks/ (evaluators) and the candidate file.
    allow = [sys.prefix, sys.base_prefix, os.path.dirname(os.path.realpath(sys.executable)),
             os.path.dirname(os.path.realpath(evaluator_script)), os.path.dirname(os.path.abspath(__file__)),
             candidate]
    proc = run_isolated([sys.executable, str(evaluator_script), "--inner", candidate, *args], timeout,
                        stdin=nonce + "\n" + (json.dumps(payload) if payload is not None else ""),
                        allow=allow)
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


def read_nonce_and_payload():
    """In the child: take the nonce and the JSON payload after it, before any candidate code is loaded."""
    nonce = sys.stdin.readline().strip()
    payload = json.loads(sys.stdin.read() or "null")
    sys.stdin.close()
    return nonce, payload


def emit(nonce, result):
    print(RESULT_TAG + nonce + json.dumps(result), flush=True)
