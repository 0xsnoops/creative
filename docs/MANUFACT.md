# Deploying DIVERGE to Manufact Cloud

Research date: 2026-10-18. Nothing was deployed: we have no Manufact credentials. Everything below marked "verified" was run locally at $0.

## Findings

| Question | Answer | Confidence |
| --- | --- | --- |
| Can DIVERGE deploy as-is (stdio)? | **No.** Every Manufact Python example serves streamable HTTP at `/mcp` from an ASGI `app` that uvicorn runs on a port. DIVERGE's `python -m diverge serve` is stdio-only (`mcp.run("stdio")` in `diverge/server.py`). No Manufact doc describes hosting a stdio server. | High for "HTTP is the documented path". Medium for "stdio is unsupported": no doc says so explicitly. |
| Is mcp-use required? | **Optional.** The official Python guide and example repo use only the official `mcp` SDK (FastMCP). Manufact detects `mcp` in `pyproject.toml` and labels the repo `mcp-python`. The `mcp-use` npm CLI is just one way to deploy (`npx mcp-use deploy`); the dashboard "Import from GitHub" flow does the same job. | High |
| Is a thin adapter enough? | **Yes.** In the installed SDK (`mcp` 2.3.0), `MCPServer` has `streamable_http_app(stateless_http=..., transport_security=...)`, which returns a Starlette app. `diverge_mcp_http.py` wraps the existing `diverge.server.mcp` object, so no tool code changes. | Verified locally |
| Manifest, Dockerfile or entry point required? | **No manifest.** Runtime is detected from `pyproject.toml` / `requirements.txt` (uv, poetry, pipenv, pip). If there is no `uv.lock`, Manufact runs `pip install .`. If `uv.lock` is committed, it runs `uv sync --frozen`. DIVERGE commits `uv.lock`, and `uv lock --check` passes. You set the port and start command in the dashboard. A Dockerfile is optional and only needed for unsupported runtimes. | High (docs + example repo) |
| Proxy gotcha | Behind Manufact's reverse proxy, the SDK's DNS-rebinding check returns **421 Misdirected Request**. The guide disables it (`enable_dns_rebinding_protection=False`) and notes that `allowed_hosts=["*"]` does not work. The adapter disables the check unless `HERD_ALLOWED_HOSTS` is set. | Docs; not verified against the real proxy |
| Credentials | You need a Manufact Cloud account with an organization, plus GitHub access through the Manufact/mcp-use GitHub App. The CLI alternative is `npx mcp-use login`, then `npx mcp-use deploy` (`--no-github` uploads a tarball instead). The Free plan covers the first server. DIVERGE's own runtime keys (`OPENROUTER_API_KEY`, `EXA_API_KEY`) go in the server's environment variables in the dashboard, never in the repo. | High |
| Local testing without credentials | Yes. Run uvicorn locally and connect with the official `mcp` client (done below), or with MCP Inspector (`npx @modelcontextprotocol/inspector`, transport "Streamable HTTP", URL `http://127.0.0.1:8000/mcp`, header `Authorization: Bearer <token>`). Inspector was not run here. | Verified with the mcp client |

### Sources (fetched 2026-10-18)
- Manufact blog, "MCP app with the MCP Python SDK", Enrico Toniato, dated 2026-05-10: https://manufact.com/blog/mcp-app-with-mcp-python-sdk. Covers port 8000, the `uvicorn 'pkg:app' --host 0.0.0.0 --port 8000` start command, `uv sync --frozen` versus `pip install .`, the 421/DNS-rebinding issue and `stateless_http=True`.
- Official example repo (last commit 2026-07-06): https://github.com/manufacts/manufact-for-mcp-python. Its README has the one-click deploy URL with `runtime=python`, `base-image=python:3.12`, `port=8000`.
- Dashboard quickstart: https://docs.manufact.com/dashboard/quickstart. Covers org, Import from GitHub, env vars, the `https://<slug>.run.mcp-use.com/mcp` URL and CLI login/deploy.
- Build & runtimes: https://docs.manufact.com/dashboard/build-runtimes. Python is detected from `pyproject.toml`; a committed Dockerfile is used as-is.
- Server settings: https://docs.manufact.com/dashboard/settings. Framework presets fill in the build command, start command and port.
- CLI deploy: https://docs.manufact.com/cli/deploy. Covers `--runtime python`, `--port` (default **3000**, so pass 8000), `--start-command`, `--env`, `--no-github`.
- mcp-use docs call Manufact Cloud the recommended deployment target for mcp-use servers: https://docs.mcp-use.com/typescript/server/deployment/mcp-use

## What was implemented

- `diverge_mcp_http.py` (repo root) exports the ASGI `app` that Manufact's start command runs. It serves the same `diverge.server.mcp` object as the stdio server (`streamable_http_app(stateless_http=True)`), so the tools are identical, at `/mcp`. The local entry `python diverge_mcp_http.py [--port N]` binds 127.0.0.1 only.
- **Exposure guard.** DIVERGE's tools spend the owner's OpenRouter/Exa budget, and `reality_check` / `diverge_check` / `diverge_escape` run the caller's `evaluator_cmd` as a command. On a public URL, that is remote command execution. The adapter therefore requires `Authorization: Bearer $DIVERGE_HTTP_TOKEN` (the older `HERD_HTTP_TOKEN` is still accepted), which uses constant-time comparison:
  - wrong or missing token: **401**
  - `DIVERGE_HTTP_TOKEN` (and `HERD_HTTP_TOKEN`) unset: **503** (fails closed), unless `HERD_HTTP_ALLOW_OPEN=1`
  - the local loopback entry sets `HERD_HTTP_ALLOW_OPEN=1` by default
- `tests/test_manufact.py` starts the app on a free loopback port. It checks that the HTTP tool list equals the stdio server's tool list, that a missing or wrong token gets 401, and that an unset token gets 503. It makes no API calls.
- No new dependencies. `uvicorn` 0.54.0 and `starlette` 1.8.0 already come in through `mcp[cli]`. mcp-use was not added. No files outside the allowed set were touched.

## Run locally

```bash
# exactly what Manufact will run, on loopback, with a token
DIVERGE_HTTP_TOKEN=dev ~/.local/bin/uv run uvicorn 'diverge_mcp_http:app' --host 127.0.0.1 --port 8000
# or, open on loopback only:
~/.local/bin/uv run python diverge_mcp_http.py --port 8000
# tests
~/.local/bin/uv run pytest tests/test_manufact.py -q
```

Claude Code over HTTP: `claude mcp add --transport http diverge http://127.0.0.1:8000/mcp --header "Authorization: Bearer dev"`.

### Local verification output (2026-10-18)

Run with `OPENROUTER_API_KEY=` and `EXA_API_KEY=` blanked and `budget_usd=0`, so no paid call was possible. The `diverge_map` result came from the local cache (the transcript below was recorded before the rename and is shown with the current server and tool names).

```
no token -> HTTP 401
server: diverge
tools: ['diverge_map', 'blind_spots', 'reality_check', 'diverge_check', 'diverge_escape']
diverge_map(budget 0): {"task":"online_bin_packing","n_samples":25,"n_unlabeled":0,"low_confidence":true,"concentration_top3":0.68,"top3":["Tight-fit-or-spacious gap avoidance","Best Fit (tightest residual)","Sum-of-Squares gap-count balancing"],...
server stopped
```

`pytest tests/test_manufact.py`: 3 passed.

## Manual deployment steps (for a human with Manufact credentials)

1. Push this branch to a GitHub repo that the Manufact GitHub App can access.
2. In the Manufact dashboard, go to **Servers → New Server → Import from GitHub**, pick the repo and branch, and set:
   - **Port:** `8000`
   - **Build command:** leave empty (the committed `uv.lock` triggers `uv sync --frozen`)
   - **Start command:** `uvicorn 'diverge_mcp_http:app' --host 0.0.0.0 --port 8000`. If `uvicorn` is not on PATH after `uv sync`, use `.venv/bin/uvicorn ...` or `uv run uvicorn ...`.
   - **Environment variables:**
     - `DIVERGE_HTTP_TOKEN`: a long random string
     - `OPENROUTER_API_KEY`
     - `EXA_API_KEY`
     - optionally `HERD_ALLOWED_HOSTS=<slug>.run.mcp-use.com`, but only after confirming the Host header the proxy forwards
3. Click **Deploy**. Then smoke-test:
   ```bash
   curl -s -X POST https://<slug>.run.mcp-use.com/mcp \
     -H 'content-type: application/json' -H 'accept: application/json, text/event-stream' \
     -H "Authorization: Bearer $DIVERGE_HTTP_TOKEN" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
   ```
   Equivalent CLI route: `npx -y mcp-use@latest login`, then `npx -y mcp-use@latest deploy --runtime python --port 8000 --start-command "uvicorn 'diverge_mcp_http:app' --host 0.0.0.0 --port 8000" --env DIVERGE_HTTP_TOKEN=... --env-file .env`. The CLI defaults to port 3000, so `--port` is required. Check what `--env-file` uploads before using it.

## Unverified / caveats

- **No real deploy was done.** The proxy behaviour (421 vs. OK), the PATH of `uvicorn` after `uv sync --frozen`, and health-check expectations are all untested on Manufact.
- **Packaging.** DIVERGE's `pyproject.toml` has no `[build-system]`. With `uv.lock`, `uv sync` treats it as a virtual project and installs only the dependencies. The start command then imports `diverge_mcp_http` and `diverge` from the repo root, because uvicorn adds the working directory to `sys.path`. This worked locally. If Manufact falls back to `pip install .` (no lockfile), a project without a build-system may fail. Keep `uv.lock` committed.
- **Cold cache.** `cache/` and `results/` are gitignored, so a fresh deploy has no cached maps. The first `diverge_map` call per task makes paid OpenRouter calls, bounded by `budget_usd` and the global cap in `config.yaml`. The ledger is also local disk, so a redeploy or a second replica resets the global cap's history.
- **Long tool calls.** Calls can run for minutes. Whether Manufact's gateway enforces request timeouts on streamable-HTTP responses is not documented in the pages read.
- **Stateless sessions.** `stateless_http=True` follows Manufact's example. Progress notifications stream on the POST's SSE response, but this was not exercised against the proxy.
- **Bearer auth.** Static bearer auth is our own guard, not a Manufact feature. Manufact may offer gateway-level auth, but none was found in the pages read. MCP-spec OAuth is not implemented.
- **Unchecked docs.** The `docs.mcp-use.com/python/...` server docs were not checked, since they were not needed for the official SDK path.
