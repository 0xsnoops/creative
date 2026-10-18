"""Self-onboarding for coding agents: `python -m diverge onboard --client claude --target <dir>`.

Installs the DIVERGE skill and the /diverge command into <target>/.claude/ and registers the DIVERGE MCP
server (stdio, `python -m diverge serve` run by uv from this repo) in <target>/.mcp.json.
Idempotent; never overwrites different existing content without force; touches no other file.
The MCP config never carries secrets: the server reads its keys from this repo's .env
(pipeline.load_env) or the client's environment."""
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL_SRC = REPO / "skills" / "diverge" / "SKILL.md"
COMMAND_SRC = REPO / ".claude" / "commands" / "diverge.md"
SUPPORTED = ("claude",)
SERVER_NAME = "diverge"


def find_uv():
    """Absolute path of the uv executable, or None. Not symlink-resolved, so upgrades keep working."""
    candidates = [shutil.which("uv"), os.environ.get("UV"),
                  str(Path.home() / ".local" / "bin" / "uv"), str(Path.home() / ".cargo" / "bin" / "uv")]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    return None


def server_entry(uv, repo=REPO):
    return {"type": "stdio", "command": uv,
            "args": ["--directory", str(repo), "run", "python", "-m", "diverge", "serve"]}


def _same_entry(a, b):
    """Equal up to the default transport: Claude Code treats a missing "type" as stdio."""
    norm = lambda e: {**e, "type": e.get("type", "stdio")} if isinstance(e, dict) else e
    return norm(a) == norm(b)


def _key_present(name, repo=REPO):
    """Whether a key is set in the environment or in the repo's .env. Reads names only; never values out."""
    if os.environ.get(name):
        return True
    env = repo / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if key.strip() == name and value.strip().strip("'\""):
                    return True
    return False


def _copy_action(src, dest, force, dry_run):
    status, detail = None, ""
    new = src.read_text()
    if not dest.exists():
        status, detail = "created", "copied from " + str(src)
    elif dest.read_text() == new:
        return {"path": str(dest), "status": "unchanged"}
    elif force:
        status, detail = "updated", "different content overwritten (--force)"
    else:
        return {"path": str(dest), "status": "conflict",
                "detail": "exists with different content; not overwritten (rerun with --force to replace it)"}
    if dry_run:
        return {"path": str(dest), "status": "planned", "detail": "would be " + status}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(new)
    return {"path": str(dest), "status": status, "detail": detail}


def _mcp_action(path, entry, force, dry_run):
    if path.exists():
        try:
            config = json.loads(path.read_text() or "{}")
        except json.JSONDecodeError as e:
            return {"path": str(path), "status": "conflict", "detail": f"not valid JSON ({e}); fix it by hand"}
        if not isinstance(config, dict) or not isinstance(config.get("mcpServers", {}), dict):
            return {"path": str(path), "status": "conflict", "detail": "unexpected structure; fix it by hand"}
        servers = config.setdefault("mcpServers", {})
        if SERVER_NAME in servers:
            if _same_entry(servers[SERVER_NAME], entry):
                return {"path": str(path), "status": "unchanged"}
            if not force:
                return {"path": str(path), "status": "conflict",
                        "detail": f'a different "{SERVER_NAME}" server is configured; not overwritten '
                                  "(rerun with --force to replace that entry)"}
            status, detail = "updated", f'"{SERVER_NAME}" entry replaced (--force); other servers kept'
        else:
            status, detail = "updated", f'"{SERVER_NAME}" server added; other servers kept'
    else:
        config, status, detail = {"mcpServers": {}}, "created", f'new file with the "{SERVER_NAME}" server'
    if dry_run:
        return {"path": str(path), "status": "planned", "detail": "would be " + status}
    config["mcpServers"][SERVER_NAME] = entry
    path.write_text(json.dumps(config, indent=2) + "\n")
    return {"path": str(path), "status": status, "detail": detail}


def next_steps(uv, actions, repo=REPO):
    run = f"{uv or 'uv'} --directory {repo} run python -m diverge"
    steps = []
    if any(a["status"] == "conflict" for a in actions):
        steps.append("Resolve the conflicts listed in actions: compare the existing files with this repo's "
                     "versions, or rerun with --force to overwrite them.")
    if uv is None:
        steps.append("uv was not found; install it (https://docs.astral.sh/uv/) and rerun onboard so .mcp.json "
                     "gets its absolute path.")
    steps += [
        "Restart or reload Claude Code in the target directory so it reads .mcp.json, and approve the project "
        "MCP server 'diverge' when asked (or check it with /mcp).",
        "Check readiness with the diverge_status MCP tool (no cost), or run: " + f"{run} preflight",
        ("API keys are read from the DIVERGE repo's .env (" + str(repo / ".env") + ") or the environment; "
         + ("OPENROUTER_API_KEY is present." if _key_present("OPENROUTER_API_KEY", repo) else
            "OPENROUTER_API_KEY is NOT set: ask the user to set it there. Never invent a key.")),
        "Try the offline demo (no API calls, $0): " + f"{run} demo strategy_loop",
        "A first consensus map can take several minutes; start Claude Code with MCP_TOOL_TIMEOUT=1800000 so the "
        "client does not give up.",
    ]
    return steps


def onboard(client="claude", target=".", dry_run=False, force=False, uv=None):
    """Return (exit code, report). Exit 0 when nothing conflicts, 1 on a conflict, 2 if unsupported."""
    target = Path(target).expanduser().resolve()
    uv = uv or find_uv()
    entry = server_entry(uv or "uv")
    report = {"client": client, "target": str(target), "dry_run": dry_run, "actions": [],
              "mcp_server": {"name": SERVER_NAME, **entry}}
    if client not in SUPPORTED:
        report.update(supported=False, detail=f"client {client!r} is not supported by onboard (only: "
                      + ", ".join(SUPPORTED) + "). Register this stdio MCP server in your client by hand.",
                      next_steps=[f"Add a stdio MCP server named 'diverge' running: {entry['command']} "
                                  + " ".join(entry["args"])])
        return 2, report
    if not target.is_dir():
        report.update(supported=True, detail=f"target {target} is not a directory", next_steps=[])
        return 1, report
    actions = [
        _copy_action(SKILL_SRC, target / ".claude" / "skills" / "diverge" / "SKILL.md", force, dry_run),
        _copy_action(COMMAND_SRC, target / ".claude" / "commands" / "diverge.md", force, dry_run),
        _mcp_action(target / ".mcp.json", entry, force, dry_run),
    ]
    report.update(supported=True, actions=actions, next_steps=next_steps(uv, actions))
    return (1 if any(a["status"] == "conflict" for a in actions) else 0), report


def render(report):
    lines = [f"DIVERGE onboard: client={report['client']} target={report['target']}"
             + (" (dry run: nothing written)" if report.get("dry_run") else "")]
    if report.get("detail"):
        lines.append(report["detail"])
    for a in report["actions"]:
        lines.append(f"  {a['status']:9} {a['path']}" + (f"  ({a['detail']})" if a.get("detail") else ""))
    m = report["mcp_server"]
    lines.append(f"MCP server '{m['name']}' ({m['type']}): {m['command']} {' '.join(m['args'])}")
    if report.get("next_steps"):
        lines.append("Next steps:")
        lines += [f"  {i}. {s}" for i, s in enumerate(report["next_steps"], 1)]
    return "\n".join(lines)


def main(args, out=print):
    code, report = onboard(args.client, args.target, args.dry_run, args.force)
    out(json.dumps(report, indent=2) if args.json else render(report))
    return code
