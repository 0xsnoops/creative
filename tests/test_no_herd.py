"""The product is DIVERGE: no user-facing surface may still say "herd" (the old name and the old word
for the consensus). Literal file names (herd_map.json) and legacy env var names (HERD_*) are allowed."""
import asyncio
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERD = re.compile(r"herd", re.I)
ALLOWED = re.compile(r"herd_map\.json|HERD_[A-Z_]+")
DOCS = ["README.md", "AGENTS.md", "skills/diverge/SKILL.md", ".claude/commands/diverge.md", "docs/TRACKS.md"]


def tool_list():
    from diverge.server import mcp
    return asyncio.run(mcp.list_tools())


def test_no_tool_name_says_herd():
    names = [t.name for t in tool_list()]
    assert "diverge_map" in names
    assert not [n for n in names if HERD.search(n)]


def test_instructions_and_descriptions_never_say_herd():
    from diverge.server import INSTRUCTIONS, mcp
    assert not HERD.search(INSTRUCTIONS) and not HERD.search(mcp.instructions or "")
    for t in tool_list():
        assert not HERD.search(t.description or ""), t.name


def test_agent_docs_never_say_herd():
    for doc in DOCS:
        path = ROOT / doc
        if not path.exists():
            continue
        left = [line for line in path.read_text().splitlines() if HERD.search(ALLOWED.sub("", line))]
        assert not left, (doc, left[:5])


def run(*args):
    r = subprocess.run([sys.executable, "-m", "diverge", *args], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    return r.stdout + r.stderr


def test_cli_help_never_says_herd():
    assert not HERD.search(run("--help"))


def test_offline_demo_never_says_herd():
    out = run("demo", "strategy_loop")
    assert "VERDICT" in out or "verdict" in out.lower()
    assert not HERD.search(out), [line for line in out.splitlines() if HERD.search(line)][:5]
