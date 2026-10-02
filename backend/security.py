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


class MaxBodySizeMiddleware:
    """Rejects request bodies over a size limit with 413 (report item R6):
    up front from Content-Length, and otherwise by reading the (chunked)
    body up to the limit before the app sees it - the endpoints read the
    whole body anyway, and an exception raised mid-read would be turned
    into a generic 400 by FastAPI. `limits` maps a path to a callable
    returning that path's limit (looked up per request, so a patched
    MAX_UPLOAD_BYTES applies); everything else gets `default_limit`."""

    def __init__(self, app: ASGIApp, default_limit: int, limits=None) -> None:
        self.app = app
        self.default_limit = default_limit
        self.limits = dict(limits or {})

    def _limit(self, path: str) -> int:
        getter = self.limits.get(path)
        return int(getter()) if getter is not None else self.default_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in STATE_CHANGING_METHODS:
            await self.app(scope, receive, send)
            return
        limit = self._limit(scope.get("path", ""))
        too_large = PlainTextResponse(f"Request body exceeds {limit} bytes", status_code=413)
        length = _header(scope, b"content-length")
        if length is not None and length.isdigit() and int(length) > limit:
            await too_large(scope, receive, send)
            return

        messages = []
        received = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] != "http.request":
                break
            received += len(message.get("body", b""))
            if received > limit:
                await too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay():
            if messages:
                return messages.pop(0)
            return await receive()

        await self.app(scope, replay, send)
