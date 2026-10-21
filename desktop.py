from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import requests
import uvicorn
import webview

from backend.config import APP_ID, HOST, port as configured_port

# Not 8188: that's ComfyUI's default port, and people who run PixelFlow are
# likely to have ComfyUI running too. PIXELFLOW_PORT overrides it.
PORT = configured_port()

logger = logging.getLogger(__name__)


def _url(port: int) -> str:
    return f"http://{HOST}:{port}"


def _health_url(port: int) -> str:
    return f"{_url(port)}/api/health"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _webview2_runtime_folder() -> Path:
    """A machine without the Microsoft Edge WebView2 Runtime installed can't open this app's
    window at all (pywebview's edgechromium backend needs it). scripts/build_release.ps1 bundles a
    pinned "Fixed Version" WebView2 Runtime as a sibling "WebView2Runtime" folder next to the
    packaged .exe specifically so the release doesn't depend on the target machine having it."""
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else _project_root()
    return base / "WebView2Runtime"


def _configure_webview2_runtime() -> None:
    """Point the WebView2 loader at the bundled Fixed Version runtime, if present.

    pywebview 5.x's edgechromium backend calls CoreWebView2Environment.CreateAsync()
    without an explicit browser-executable-folder argument (no webview.settings key for
    it exists in this version -- an older API some references still mention). Microsoft's
    WebView2Loader.dll falls back to the WEBVIEW2_BROWSER_EXECUTABLE_FOLDER environment
    variable in that case, so that's set here instead. In dev mode this folder doesn't
    exist (it's only bundled by the release build), so the env var is left unset and the
    system-installed Evergreen runtime is used instead.
    """
    runtime_folder = _webview2_runtime_folder()
    if runtime_folder.exists():
        os.environ["WEBVIEW2_BROWSER_EXECUTABLE_FOLDER"] = str(runtime_folder)


def _server_log_path() -> Path:
    # Next to the other logs in the data folder (writable even when the exe
    # is installed under Program Files - see backend/paths.py).
    from backend.paths import LOG_DIR

    return LOG_DIR / "server-error.log"


_server_error: list[str] = []
_server: dict = {}


def _run_server(port: int = PORT) -> None:
    # A windowed (console=False) build has no stderr, so an exception on this background thread --
    # e.g. an import failing inside the frozen bundle -- used to vanish without a trace: the main
    # thread just timed out with "server did not start". Keep the traceback, and write it to the
    # log folder so it can actually be diagnosed.
    try:
        from backend.main import app

        # log_config=None: don't let uvicorn install its own stderr-only
        # handlers (stderr is None in the windowed build); its loggers then
        # propagate to the root logger, i.e. to logs/pixelflow.log.
        config = uvicorn.Config(app, host=HOST, port=port, log_level="warning", log_config=None)
        server = uvicorn.Server(config)
        _server["server"] = server
        # uvicorn only installs OS signal handlers on the main thread, so running its
        # server loop here (a background thread) is safe alongside pywebview's own loop.
        server.run()
    except BaseException as exc:
        detail = traceback.format_exc()
        if isinstance(exc, SystemExit):
            # uvicorn logs a failed bind and calls sys.exit(1) -- by far the most common cause.
            detail = f"uvicorn exited (code {exc.code}); port {port} is most likely already in use.\n\n{detail}"
        _server_error.append(detail)
        logging.getLogger(__name__).error("embedded server failed: %s", detail)
        try:
            path = _server_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(detail, encoding="utf-8")
        except OSError:
            pass
        raise


def _wait_for_server(thread: threading.Thread, timeout_s: float = 120.0, port: int = PORT) -> None:
    # The server itself comes up in a second or two (torch is imported lazily, see
    # backend/nodes/runtime.py), but the budget stays generous for slow first launches -- and
    # stop immediately if the server thread has already died.
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not thread.is_alive():
            reason = _server_error[-1] if _server_error else "the server thread exited unexpectedly"
            raise RuntimeError(f"The embedded PixelFlow server failed to start (see {_server_log_path()}):\n{reason}")
        try:
            response = requests.get(_health_url(port), timeout=0.5)
        except requests.RequestException:
            time.sleep(0.25)
            continue
        if _is_pixelflow_health(response):
            return
        # Something answered, but it isn't PixelFlow's health endpoint -- another app owns the
        # port (our own uvicorn will fail to bind and the thread will die shortly). Keep polling
        # until then so the thread-death branch above reports the real bind error.
        time.sleep(0.25)
    raise RuntimeError(f"The embedded PixelFlow server did not start within {timeout_s:.0f}s")


def _is_pixelflow_health(response: requests.Response) -> bool:
    if not response.ok:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("app") == APP_ID


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((HOST, port))
        except OSError:
            return False
    return True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def _pixelflow_already_running(port: int) -> bool:
    try:
        return _is_pixelflow_health(requests.get(_health_url(port), timeout=1.0))
    except requests.RequestException:
        return False


def choose_port(preferred: int = PORT) -> tuple[int, bool]:
    """(port, start_own_server). The preferred port when it's free; when
    another PixelFlow already serves it, attach to that one (a second window
    on the same server instead of a half-started second instance); when
    some other app owns it, a free port picked by the OS. Everything in the
    frontend uses relative URLs and the Origin check compares against the
    Host header, so any port works."""
    if _port_is_free(preferred):
        return preferred, True
    if _pixelflow_already_running(preferred):
        return preferred, False
    fallback = _free_port()
    logger.warning("port %d is used by another program; using port %d instead", preferred, fallback)
    return fallback, True


WINDOW_TITLE = "PixelFlow"


def _focus_existing_window(title: str = WINDOW_TITLE) -> bool:
    """Bring another process's top-level window titled exactly `title` (the
    running PixelFlow) to the front, restoring it when minimized. True when
    one was found. Windows only; elsewhere (and on any failure) False, and
    the caller opens a window of its own instead."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        own_pid = os.getpid()
        found: list[int] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def visit(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if buffer.value == title:
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value != own_pid:
                    found.append(hwnd)
                    return False
            return True

        user32.EnumWindows(visit, 0)
        if not found:
            return False
        hwnd = found[0]
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        return True
    except Exception:  # noqa: BLE001 - never stop the launch over this
        logger.debug("could not focus the running PixelFlow window", exc_info=True)
        return False


def _shutdown(thread: threading.Thread | None, timeout_s: float = 10.0) -> None:
    """Window closed: stop the running prompt at its next step, let uvicorn
    finish (so a PNG being written isn't cut off), and wait for it."""
    if thread is None:
        return
    try:
        from backend.main import queue

        queue.interrupt()
    except Exception:  # noqa: BLE001 - best effort on the way out
        pass
    server = _server.get("server")
    if server is not None:
        server.should_exit = True
    thread.join(timeout_s)


def main() -> int | None:
    from backend.selftest import run_self_test, wants_self_test

    if wants_self_test(sys.argv[1:]):
        # Frontend, WebView2 runtime, torch/CUDA, diffusers and a real prompt through the
        # embedded server (throwaway data folder), then exit -- no window (backend/selftest.py).
        return run_self_test(desktop=sys.modules[__name__])
    if "--mcp" in sys.argv[1:]:
        # F24: an MCP server on stdio for assistants, driving the PixelFlow that's already
        # running (backend/mcp_server.py) -- no window, no server of its own.
        from backend import mcp_server

        return mcp_server.main()
    # Before anything else, so even an import failure of the backend in the
    # frozen app lands in logs/pixelflow.log (the build has no console).
    from backend.logging_setup import setup_logging

    setup_logging()
    port, own_server = choose_port(PORT)
    thread = None
    if own_server:
        thread = threading.Thread(target=_run_server, args=(port,), daemon=True)
        thread.start()
        _wait_for_server(thread, port=port)
        from backend import mcp_server

        mcp_server.record_server(port)  # so `PixelFlow.exe --mcp` finds a fallback port too
    elif _focus_existing_window():
        # A second launch (R5): bring the running PixelFlow's window forward instead
        # of opening another window on the same server.
        logger.info("PixelFlow is already running on port %d; focused its window", port)
        return
    else:
        logger.info("PixelFlow is already running on port %d (no window of it found); opening a window on it", port)

    _configure_webview2_runtime()
    webview.create_window(WINDOW_TITLE, _url(port), width=1440, height=960, min_size=(1024, 700))
    try:
        webview.start(gui="edgechromium")
    finally:
        _shutdown(thread)
        if own_server:
            from backend import mcp_server

            mcp_server.forget_server(port)


if __name__ == "__main__":
    sys.exit(main())
