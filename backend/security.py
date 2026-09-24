"""Request-origin hardening for a server that's only meant to be reached from
its own frontend on localhost.

Binding to 127.0.0.1 alone doesn't stop a web page open in the user's normal
browser from talking to this server: any site can POST to
http://127.0.0.1:<port>/api/... (a "simple" cross-origin request needs no CORS
preflight), open the /ws WebSocket (WebSockets aren't covered by CORS at all),
or - via DNS rebinding - make its own hostname resolve to 127.0.0.1 and then
read responses as same-origin. Two checks close that:

- TrustedHostMiddleware (wired in main.py with ALLOWED_HOSTS) rejects any
  request whose Host header isn't a loopback name, which defeats DNS
  rebinding (the rebound request still carries the attacker's hostname).
- LocalOriginMiddleware (below) rejects state-changing HTTP requests and
  WebSocket handshakes whose Origin header isn't this server's own origin.
  Requests with no Origin at all (curl, scripts, tests) are allowed: browsers
  always send Origin on cross-origin POST/PUT/PATCH/DELETE and on WebSocket
  handshakes, so a missing header can't be a cross-site page.
"""

from typing import Optional
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# Hostnames (without port) this server answers to: the dev web app and the
# desktop window both load http://127.0.0.1:<port>; localhost is accepted for
# people who type that instead.
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

STATE_CHANGING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _header(scope: Scope, name: bytes) -> Optional[str]:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def is_allowed_origin(origin: Optional[str], host: Optional[str]) -> bool:
    """True if `origin` is absent, or is this server's own origin: http(s),
    a loopback hostname from ALLOWED_HOSTS, and the same host:port the
    request was sent to (its Host header)."""
    if origin is None:
        return True
    if not host:
        return False
    parts = urlsplit(origin)
    if parts.scheme not in ("http", "https") or parts.hostname not in ALLOWED_HOSTS:
        return False
    return parts.netloc.lower() == host.lower()


class LocalOriginMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in STATE_CHANGING_METHODS:
            if not is_allowed_origin(_header(scope, b"origin"), _header(scope, b"host")):
                response = PlainTextResponse("Cross-origin request rejected", status_code=403)
                await response(scope, receive, send)
                return
        elif scope["type"] == "websocket":
            if not is_allowed_origin(_header(scope, b"origin"), _header(scope, b"host")):
                # Closing before accept() makes the server refuse the
                # handshake (an HTTP 403 to the client).
                await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)
