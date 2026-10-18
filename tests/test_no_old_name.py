"""The product is DIVERGE: no user-facing surface may still use the old product name (also the old word
for the consensus). Literal saved-data file names and legacy env var names are allowed."""
import asyncio
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD_NAME = re.compile(r"herd", re.I)
ALLOWED = re.compile(r"herd_map\.json|HERD_[A-Z_]+")
DOCS = ["README.md", "AGENTS.md", "skills/diverge/SKILL.md", ".claude/commands/diverge.md", "docs/TRACKS.md"]


def tool_list():
    from diverge.server import mcp
    return asyncio.run(mcp.list_tools())


def test_no_tool_name_says_old_name():
    names = [t.name for t in tool_list()]
    assert "diverge_map" in names
    assert not [n for n in names if OLD_NAME.search(n)]


def test_instructions_and_descriptions_never_say_old_name():
    from diverge.server import INSTRUCTIONS, mcp
    assert not OLD_NAME.search(INSTRUCTIONS) and not OLD_NAME.search(mcp.instructions or "")
    for t in tool_list():
        assert not OLD_NAME.search(t.description or ""), t.name


def test_agent_docs_never_say_old_name():
    for doc in DOCS:
        path = ROOT / doc
        if not path.exists():
            continue
        left = [line for line in path.read_text().splitlines() if OLD_NAME.search(ALLOWED.sub("", line))]
        assert not left, (doc, left[:5])


def run(*args):
    r = subprocess.run([sys.executable, "-m", "diverge", *args], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    return r.stdout + r.stderr


def test_cli_help_never_says_old_name():
    assert not OLD_NAME.search(run("--help"))


def test_offline_demo_never_says_old_name():
    out = run("demo", "strategy_loop")
    assert "VERDICT" in out or "verdict" in out.lower()
    assert not OLD_NAME.search(out), [line for line in out.splitlines() if OLD_NAME.search(line)][:5]
