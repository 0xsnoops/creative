"""`python -m herd onboard`: skill, /herd command and .mcp.json install. Offline, writes only under tmp_path."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from herd import onboard

REPO = Path(__file__).resolve().parent.parent
FAKE_UV = "/opt/fake/bin/uv"
FAKE_KEY = "sk-or-v1-FAKE-test-key-do-not-leak-123456"


def run(target, **kw):
    return onboard.onboard("claude", target, uv=FAKE_UV, **kw)


def paths(target):
    return (target / ".claude" / "skills" / "herd" / "SKILL.md", target / ".claude" / "commands" / "herd.md",
            target / ".mcp.json")


def statuses(report):
    return [a["status"] for a in report["actions"]]


def test_creates_files_and_valid_mcp_json(tmp_path):
    code, report = run(tmp_path)
    assert code == 0 and statuses(report) == ["created"] * 3
    skill, command, mcp = paths(tmp_path)
    assert skill.read_text() == onboard.SKILL_SRC.read_text()
    assert command.read_text() == onboard.COMMAND_SRC.read_text()
    entry = json.loads(mcp.read_text())["mcpServers"]["herd"]
    assert entry == {"type": "stdio", "command": FAKE_UV,
                     "args": ["--directory", str(REPO), "run", "python", "-m", "herd", "serve"]}
    assert report["mcp_server"]["name"] == "herd" and report["next_steps"]


def test_second_run_is_unchanged(tmp_path):
    run(tmp_path)
    before = {p: p.read_text() for p in paths(tmp_path)}
    code, report = run(tmp_path)
    assert code == 0 and statuses(report) == ["unchanged"] * 3
    assert {p: p.read_text() for p in paths(tmp_path)} == before


def test_unrelated_files_and_other_servers_untouched(tmp_path):
    other = {"command": "node", "args": ["server.js"], "env": {"X": "1"}}
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": other}, "extra": [1, 2]}))
    (tmp_path / ".claude" / "commands").mkdir(parents=True)
    mine = tmp_path / ".claude" / "commands" / "mine.md"
    mine.write_text("my command")
    settings = tmp_path / ".claude" / "settings.json"
    settings.write_text("{}")
    code, report = run(tmp_path)
    assert code == 0 and statuses(report) == ["created", "created", "updated"]
    config = json.loads((tmp_path / ".mcp.json").read_text())
    assert config["mcpServers"]["other"] == other and config["extra"] == [1, 2]
    assert "herd" in config["mcpServers"]
    assert mine.read_text() == "my command" and settings.read_text() == "{}"
    created = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert created == {".mcp.json", ".claude/commands/mine.md", ".claude/settings.json",
                       ".claude/commands/herd.md", ".claude/skills/herd/SKILL.md"}


def test_conflicts_are_not_overwritten_unless_forced(tmp_path):
    skill, command, mcp = paths(tmp_path)
    skill.parent.mkdir(parents=True)
    skill.write_text("someone else's skill")
    mcp.write_text(json.dumps({"mcpServers": {"herd": {"command": "python", "args": ["old"]}, "o": {"command": "x"}}}))
    code, report = run(tmp_path)
    assert code == 1 and statuses(report) == ["conflict", "created", "conflict"]
    assert skill.read_text() == "someone else's skill"
    assert json.loads(mcp.read_text())["mcpServers"]["herd"] == {"command": "python", "args": ["old"]}
    assert any("--force" in s for s in report["next_steps"])

    code, report = run(tmp_path, force=True)
    assert code == 0 and statuses(report) == ["updated", "unchanged", "updated"]
    assert skill.read_text() == onboard.SKILL_SRC.read_text()
    servers = json.loads(mcp.read_text())["mcpServers"]
    assert servers["herd"]["command"] == FAKE_UV and servers["o"] == {"command": "x"}


def test_invalid_mcp_json_is_a_conflict_even_with_force(tmp_path):
    (tmp_path / ".mcp.json").write_text("{not json")
    code, report = run(tmp_path, force=True)
    assert code == 1 and report["actions"][2]["status"] == "conflict"
    assert (tmp_path / ".mcp.json").read_text() == "{not json"


def test_dry_run_writes_nothing(tmp_path):
    code, report = run(tmp_path, dry_run=True)
    assert code == 0 and statuses(report) == ["planned"] * 3 and report["dry_run"]
    assert list(tmp_path.iterdir()) == []


def test_unsupported_client(tmp_path):
    code, report = onboard.onboard("cursor", tmp_path, uv=FAKE_UV)
    assert code == 2 and report["supported"] is False and report["actions"] == []
    assert "serve" in report["next_steps"][0] and list(tmp_path.iterdir()) == []


def test_cli_json_report_and_no_secrets(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "OPENROUTER_API_KEY": FAKE_KEY,
           "EXA_API_KEY": FAKE_KEY + "-exa"}
    proc = subprocess.run([sys.executable, "-m", "herd", "onboard", "--client", "claude", "--target", str(tmp_path),
                           "--json"], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert set(report) >= {"client", "target", "actions", "mcp_server", "next_steps"}
    assert {a["status"] for a in report["actions"]} == {"created"}
    assert FAKE_KEY not in proc.stdout and FAKE_KEY not in proc.stderr
    config_text = (tmp_path / ".mcp.json").read_text()
    assert FAKE_KEY not in config_text
    entry = json.loads(config_text)["mcpServers"]["herd"]
    assert "env" not in entry and set(entry) == {"type", "command", "args"}
    assert entry["args"][-3:] == ["-m", "herd", "serve"] and entry["type"] == "stdio"
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert FAKE_KEY not in p.read_text()
    # The real repo .env (if any) must not leak either.
    dotenv = REPO / ".env"
    if dotenv.exists():
        values = [l.split("=", 1)[1].strip().strip("'\"") for l in dotenv.read_text().splitlines()
                  if "=" in l and not l.lstrip().startswith("#")]
        for v in (v for v in values if len(v) >= 8):
            assert v not in proc.stdout and v not in config_text


def test_cli_dry_run_text(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "herd", "onboard", "--target", str(tmp_path), "--dry-run"],
                          cwd=REPO, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and "dry run" in proc.stdout and "planned" in proc.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("path", ["skills/herd/SKILL.md", ".claude/commands/herd.md"])
def test_frontmatter_parses(path):
    text = (REPO / path).read_text()
    assert text.startswith("---\n")
    meta = yaml.safe_load(text.split("---\n")[1])
    assert isinstance(meta["description"], str) and meta["description"]


def test_skill_description_triggers():
    meta = yaml.safe_load((REPO / "skills/herd/SKILL.md").read_text().split("---\n")[1])
    assert meta["name"] == "herd" and len(meta["description"]) <= 1024
    d = meta["description"].lower()
    assert "scored attempts" in d and ("flat" in d or "barely improving" in d)
    assert "escalate" in d and "human" in d
    assert "without being asked" in d
    body = (REPO / "skills/herd/SKILL.md").read_text()
    assert "Do not wait for the user to ask you to use HERD" in body
    assert "Before escalating a scored strategy problem to a human, call `herd_escape`" in body
    for tool in ("herd_status", "herd_check", "herd_escape", "herd_map", "blind_spots", "reality_check"):
        assert tool in body
    assert "python -m herd onboard" in body and "demo strategy_loop" in body
