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


# ---- free-port fallback / second instance (report item R5) ----


def test_choose_port_uses_the_preferred_port_when_free(monkeypatch):
    monkeypatch.setattr(desktop, "_port_is_free", lambda port: True)
    assert desktop.choose_port(8288) == (8288, True)


def test_choose_port_attaches_to_a_running_pixelflow(monkeypatch):
    monkeypatch.setattr(desktop, "_port_is_free", lambda port: False)
    monkeypatch.setattr(desktop, "_pixelflow_already_running", lambda port: True)
    assert desktop.choose_port(8288) == (8288, False)


def test_choose_port_falls_back_when_another_app_owns_it(monkeypatch):
    monkeypatch.setattr(desktop, "_port_is_free", lambda port: False)
    monkeypatch.setattr(desktop, "_pixelflow_already_running", lambda port: False)
    monkeypatch.setattr(desktop, "_free_port", lambda: 50123)
    assert desktop.choose_port(8288) == (50123, True)


def test_free_port_is_bindable():
    port = desktop._free_port()
    assert desktop._port_is_free(port)


# ---- graceful shutdown (report item R4) ----


def test_shutdown_interrupts_and_stops_the_server(monkeypatch):
    from backend import main

    class _Server:
        should_exit = False

    server = _Server()
    monkeypatch.setitem(desktop._server, "server", server)
    stopped = threading.Event()
    interrupted = []
    monkeypatch.setattr(main.queue, "interrupt", lambda *a: interrupted.append(True))
    thread = threading.Thread(target=lambda: stopped.wait(5), daemon=True)
    thread.start()
    stopped.set()
    desktop._shutdown(thread, timeout_s=2)
    assert server.should_exit and interrupted and not thread.is_alive()
