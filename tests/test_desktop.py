import sys
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


def test_a_second_launch_focuses_the_running_window_instead_of_opening_one(monkeypatch):
    monkeypatch.setattr(desktop, "setup_logging", lambda: None, raising=False)
    monkeypatch.setattr(desktop, "choose_port", lambda preferred: (8288, False))
    monkeypatch.setattr(desktop, "_focus_existing_window", lambda: True)
    opened = []
    monkeypatch.setattr(desktop.webview, "create_window", lambda *a, **k: opened.append(a))
    monkeypatch.setattr(desktop.webview, "start", lambda **k: opened.append("start"))
    desktop.main()
    assert opened == []


def test_without_a_window_to_focus_a_second_launch_opens_one(monkeypatch):
    monkeypatch.setattr(desktop, "choose_port", lambda preferred: (8288, False))
    monkeypatch.setattr(desktop, "_focus_existing_window", lambda: False)
    monkeypatch.setattr(desktop, "_configure_webview2_runtime", lambda: None)
    opened = []
    monkeypatch.setattr(desktop.webview, "create_window", lambda title, url, **k: opened.append((title, url)))
    monkeypatch.setattr(desktop.webview, "start", lambda **k: None)
    desktop.main()
    assert opened == [("PixelFlow", "http://127.0.0.1:8288")]


@pytest.mark.skipif(sys.platform != "win32", reason="window lookup is Windows-only")
def test_focus_existing_window_finds_nothing_for_an_unknown_title():
    assert desktop._focus_existing_window("PixelFlow test window that does not exist 7f3a") is False


def test_focus_existing_window_is_a_no_op_off_windows(monkeypatch):
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    assert desktop._focus_existing_window() is False


@pytest.mark.skipif(sys.platform != "win32", reason="window lookup is Windows-only")
def test_focus_existing_window_finds_another_processes_window():
    """A real top-level window from another process (a tiny Tk window parked off-screen)."""
    import subprocess
    import time

    title = f"PixelFlow focus test {time.time_ns()}"
    script = f"import tkinter as tk; r = tk.Tk(); r.title({title!r}); r.geometry('80x40+-20000+-20000'); r.after(15000, r.destroy); r.mainloop()"
    proc = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.monotonic() + 10
        found = False
        while time.monotonic() < deadline and not found:
            found = desktop._focus_existing_window(title)
            time.sleep(0.1)
        assert found
    finally:
        proc.kill()
        proc.wait(5)
