"""DIVERGE over streamable HTTP, for hosted deployment (e.g. Manufact Cloud) and local HTTP testing.

Serves the same `mcp` server object as `python -m diverge serve` (stdio); no tool code is duplicated.

    uvicorn 'diverge_mcp_http:app' --host 0.0.0.0 --port 8000      # hosted; endpoint is /mcp
    uv run python diverge_mcp_http.py [--port 8000]                # local, binds 127.0.0.1

Exposure guard: DIVERGE's tools spend the owner's API budget and run `evaluator_cmd` as a command,
so a public endpoint must not be open. When DIVERGE_HTTP_TOKEN is set, every request needs
`Authorization: Bearer <token>`. When it is unset, requests are refused (503) unless
DIVERGE_HTTP_ALLOW_OPEN=1 (the local entry point sets this, since it only binds loopback).
Each DIVERGE_* variable here falls back to its legacy HERD_* name (HERD_HTTP_TOKEN,
HERD_HTTP_ALLOW_OPEN, HERD_ALLOWED_HOSTS) when unset.
"""
import argparse
import hmac
import json
import os

from mcp.server.transport_security import TransportSecuritySettings

from diverge.server import mcp


def _env(name, default=""):
    """DIVERGE_<name>, else the legacy HERD_<name>, else default."""
    return os.environ.get(f"DIVERGE_{name}") or os.environ.get(f"HERD_{name}") or default


# Behind a managed reverse proxy the Host header is the public hostname, which the SDK's
# DNS-rebinding check rejects with 421 (per Manufact's Python guide). Set DIVERGE_ALLOWED_HOSTS
# (comma-separated host[:port]) to re-enable the check for known hostnames.
_hosts = [h.strip() for h in _env("ALLOWED_HOSTS").split(",") if h.strip()]
_security = TransportSecuritySettings(enable_dns_rebinding_protection=bool(_hosts), allowed_hosts=_hosts)


def _reject(status, error, detail):
    body = json.dumps({"error": error, "detail": detail}).encode()

    async def respond(send):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})
    return respond


class BearerGuard:
    """ASGI middleware: require the DIVERGE_HTTP_TOKEN (or HERD_HTTP_TOKEN) bearer token on every HTTP request."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":  # lifespan events pass through
            return await self.inner(scope, receive, send)
        token = _env("HTTP_TOKEN")
        if not token:
            if _env("HTTP_ALLOW_OPEN") == "1":
                return await self.inner(scope, receive, send)
            return await _reject(503, "unconfigured", "set DIVERGE_HTTP_TOKEN (or DIVERGE_HTTP_ALLOW_OPEN=1)")(send)
        sent = dict(scope.get("headers") or []).get(b"authorization", b"")
        if not hmac.compare_digest(sent, f"Bearer {token}".encode()):
            return await _reject(401, "unauthorized", "missing or wrong bearer token")(send)
        return await self.inner(scope, receive, send)


# stateless_http: each POST is self-contained, so replicas behind a load balancer need no
# sticky sessions; progress notifications still stream back on the request's SSE response.
app = BearerGuard(mcp.streamable_http_app(stateless_http=True, transport_security=_security))


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="Serve DIVERGE over streamable HTTP on loopback.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ.setdefault("DIVERGE_HTTP_ALLOW_OPEN", "1")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
