"""The streamable-HTTP entry (diverge_mcp_http) serves DIVERGE's tools and guards exposure. No API calls."""
import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

import diverge_mcp_http

TOKEN = "test-token-not-a-secret"


@pytest.fixture(scope="module")
def running():
    # One server per module: the SDK's session manager can only start once per app object.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(diverge_mcp_http.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    assert srv.started
    yield f"http://127.0.0.1:{port}/mcp"
    srv.should_exit = True
    thread.join(5)


@pytest.fixture
def server(running, monkeypatch):
    monkeypatch.setenv("DIVERGE_HTTP_TOKEN", TOKEN)  # read per request, so each test sets its own
    return running


async def list_tool_names(url, token):
    async with create_mcp_http_client(headers={"Authorization": f"Bearer {token}"}) as http:
        async with streamable_http_client(url, http_client=http) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return sorted(t.name for t in (await session.list_tools()).tools)


def test_lists_same_tools_as_stdio_server(server):
    names = asyncio.run(list_tool_names(server, TOKEN))
    expected = sorted(t.name for t in asyncio.run(diverge_mcp_http.mcp.list_tools()))
    assert names == expected and "diverge_map" in names


def test_rejects_missing_or_wrong_token(server):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    assert httpx.post(server, json=body).status_code == 401
    assert httpx.post(server, json=body, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_refuses_when_unconfigured(server, monkeypatch):
    monkeypatch.delenv("DIVERGE_HTTP_TOKEN")
    monkeypatch.delenv("DIVERGE_HTTP_ALLOW_OPEN", raising=False)
    assert httpx.post(server, json={}).status_code == 503
