from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

import requests
import uvicorn
import webview

HOST = "127.0.0.1"
# Not 8188: that's ComfyUI's default port, and people who run PixelFlow are
# likely to have ComfyUI running too. Keep in sync with run.py.
PORT = 8288
URL = f"http://{HOST}:{PORT}"
HEALTH_URL = f"{URL}/api/health"
# What backend/main.py's /api/health reports as "app" -- anything else
# answering on PORT is some other program.
APP_ID = "pixelflow"


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
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else _project_root()
    return base / "server-error.log"


_server_error: list[str] = []


def _run_server() -> None:
    # A windowed (console=False) build has no stderr, so an exception on this background thread --
    # e.g. an import failing inside the frozen bundle -- used to vanish without a trace: the main
    # thread just timed out with "server did not start". Keep the traceback, and write it next to
    # the .exe so it can actually be diagnosed.
    try:
        from backend.main import app

        config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
        server = uvicorn.Server(config)
        # uvicorn only installs OS signal handlers on the main thread, so running its
        # server loop here (a background thread) is safe alongside pywebview's own loop.
        server.run()
    except BaseException as exc:
        detail = traceback.format_exc()
        if isinstance(exc, SystemExit):
            # uvicorn logs a failed bind and calls sys.exit(1) -- by far the most common cause.
            detail = f"uvicorn exited (code {exc.code}); port {PORT} is most likely already in use.\n\n{detail}"
        _server_error.append(detail)
        try:
            _server_log_path().write_text(detail, encoding="utf-8")
        except OSError:
            pass
        raise


def _wait_for_server(thread: threading.Thread, timeout_s: float = 120.0) -> None:
    # Loading the CUDA build of torch + diffusers cold can take well over 30s on first launch, so
    # the budget is generous -- but stop immediately if the server thread has already died.
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not thread.is_alive():
            reason = _server_error[-1] if _server_error else "the server thread exited unexpectedly"
            raise RuntimeError(f"The embedded PixelFlow server failed to start (see {_server_log_path()}):\n{reason}")
        try:
            response = requests.get(HEALTH_URL, timeout=0.5)
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


def main() -> None:
    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()
    _wait_for_server(thread)

    _configure_webview2_runtime()
    webview.create_window("PixelFlow", URL, width=1440, height=960, min_size=(1024, 700))
    webview.start(gui="edgechromium")


if __name__ == "__main__":
    main()
