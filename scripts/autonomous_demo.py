"""Autonomous tool-selection demo (Armature track): does a coding agent pick HERD on its own?

    uv run python scripts/autonomous_demo.py --run 1 [--max-budget 5]

1. Creates a fresh temp project and runs `python -m diverge onboard --client claude --target <it>`
   into it, so the project holds only the installed skill, the /diverge command and .mcp.json.
2. Copies in the stalled bin-packing trajectory (fixtures/trajectories/strategy_loop.json) as
   attempts/ with their dev scores, the task statement and a small evaluator wrapper.
3. Runs a separate headless agent, `claude -p` with stream-json output, in that project. The
   prompt does not mention HERD.
4. Saves the raw transcript to docs/demo/autonomous-run-<n>.jsonl (after checking it holds no
   secrets) and prints where the agent first selected a HERD tool, if it did.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UV = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
PROMPT = ("Previous attempts at the task in TASK.md and their dev scores are in attempts/. Improve the dev "
          "score using the evaluator (`python evaluate.py <file.py>`). Stop when you can't make progress.")
SECRET = re.compile(r"sk-or-v1-[0-9a-f]{20,}|sk-ant-[A-Za-z0-9_-]{20,}")


def secret_values():
    """Key values from the environment and the repo's .env (never printed), to check transcripts against."""
    values = [v for k, v in os.environ.items() if k.endswith(("_KEY", "_TOKEN")) and len(v) >= 16]
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                v = line.split("=", 1)[1].strip().strip("'\"")
                if len(v) >= 16:
                    values.append(v)
    return values

EVALUATE = '''"""Score a solution on the development instances. Usage: python evaluate.py <file.py>
Prints JSON; "score" is the average % excess bins over the lower bound (lower is better)."""
import json, subprocess, sys
from pathlib import Path
out = subprocess.run([{uv!r}, "--directory", {root!r}, "run", "python", "tasks/online_bin_packing/evaluator.py",
                      str(Path(sys.argv[1]).resolve())], capture_output=True, text=True)
line = (out.stdout.strip().splitlines() or ["{{}}"])[-1]
result = json.loads(line)
print(json.dumps({{"valid": result.get("valid"), "score": result.get("score"),
                  "error": (result.get("detail") or {{}}).get("error")}}))
'''


def build_project(target):
    onboard = subprocess.run([UV, "--directory", str(ROOT), "run", "python", "-m", "diverge", "onboard", "--client",
                              "claude", "--target", str(target), "--json"], capture_output=True, text=True)
    if onboard.returncode != 0:
        sys.exit(f"onboard failed: {onboard.stderr[-2000:]}")
    traj = json.loads((ROOT / "fixtures" / "trajectories" / "strategy_loop.json").read_text())
    (target / "attempts").mkdir()
    scores = {}
    for a in traj["attempts"]:
        (target / "attempts" / f"{a['id']}.py").write_text(a["code"])
        scores[f"{a['id']}.py"] = round(a["dev_score"], 4)
    (target / "attempts" / "scores.json").write_text(json.dumps(
        {"metric": "% excess bins on the development instances, lower is better", "scores": scores}, indent=2))
    task = (ROOT / "tasks" / "online_bin_packing" / "prompt.md").read_text()
    (target / "TASK.md").write_text(task + "\n\n---\nBenchmark name: `online_bin_packing`. Score a file with "
                                    "`python evaluate.py <file.py>` (development instances; lower is better).\n")
    (target / "evaluate.py").write_text(EVALUATE.format(uv=UV, root=str(ROOT)))
    return json.loads(onboard.stdout)


def herd_moment(lines):
    """Index and details of the first HERD tool call, plus the agent text right before it."""
    last_text, tools = None, []
    for i, line in enumerate(lines):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") != "assistant":
            continue
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") == "text":
                last_text = block["text"]
            elif block.get("type") == "tool_use":
                tools.append(block["name"])
                if block["name"].startswith("mcp__diverge__"):
                    return {"line": i, "tool": block["name"], "input": block.get("input"),
                            "text_before": last_text, "tools_before": tools[:-1]}
    return {"line": None, "tool": None, "tools_before": tools}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--max-budget", type=float, default=5.0)
    ap.add_argument("--max-turns", type=int, default=60)
    args = ap.parse_args()
    target = Path(tempfile.mkdtemp(prefix="herd-autonomous-"))
    report = build_project(target)
    print(f"project: {target}\nonboard: {json.dumps(report)[:600]}", flush=True)
    trace_dir = ROOT / "results" / "traces" / f"autonomous-demo-run{args.run}"
    env = {**os.environ, "HERD_TRACE_DIR": str(trace_dir)}
    cmd = ["claude", "-p", PROMPT, "--output-format", "stream-json", "--verbose",
           "--mcp-config", str(target / ".mcp.json"), "--strict-mcp-config",
           "--allowedTools", "Read", "Write", "Edit", "Glob", "Grep", "Bash", "mcp__herd",
           "--max-turns", str(args.max_turns), "--max-budget-usd", str(args.max_budget)]
    started = time.time()
    proc = subprocess.run(cmd, cwd=target, env=env, capture_output=True, text=True, timeout=3600)
    lines = proc.stdout.splitlines()
    raw = "\n".join(lines)
    if SECRET.search(raw) or any(v in raw for v in secret_values()):
        sys.exit("transcript contains something that looks like a secret; not saved")
    out = ROOT / "docs" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"autonomous-run-{args.run}.jsonl").write_text(raw + "\n")
    moment = herd_moment(lines)
    final = next((json.loads(l) for l in reversed(lines) if '"type":"result"' in l.replace(" ", "")), {})
    summary = {"run": args.run, "project": str(target), "prompt": PROMPT, "command": cmd[:1] + ["-p", "<PROMPT>"] + cmd[3:],
               "exit": proc.returncode, "seconds": round(time.time() - started), "herd_selected": moment["tool"] is not None,
               "moment": moment, "agent_cost_usd": final.get("total_cost_usd"), "num_turns": final.get("num_turns"),
               "final_text": final.get("result"), "stderr_tail": proc.stderr[-1500:]}
    (out / f"autonomous-run-{args.run}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()
