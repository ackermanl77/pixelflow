import threading

import pytest

pytest.importorskip("webview")  # requirements-desktop.txt only; not installed in CI
import desktop  # noqa: E402


class _Response:
    def __init__(self, payload, ok=True):
        self.ok = ok
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def test_default_port_is_not_comfyui():
    assert desktop.PORT != 8188


def test_wait_for_server_accepts_only_pixelflow_health(monkeypatch):
    monkeypatch.setattr(desktop.requests, "get", lambda url, timeout: _Response({"app": "pixelflow", "status": "ok"}))
    alive = threading.Thread(target=threading.Event().wait, args=(5,), daemon=True)
    alive.start()
    desktop._wait_for_server(alive, timeout_s=2)


def test_wait_for_server_ignores_another_app_on_the_port(monkeypatch):
    """Another server (e.g. ComfyUI) answering 200 must not count as ours;
    once our own server thread dies (failed bind), fail with its error."""
    monkeypatch.setattr(desktop.requests, "get", lambda url, timeout: _Response(None))
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    monkeypatch.setattr(desktop, "_server_error", ["port 8288 is most likely already in use"])
    with pytest.raises(RuntimeError, match="already in use"):
        desktop._wait_for_server(dead, timeout_s=2)
