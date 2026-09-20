from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import requests
import uvicorn
import webview

HOST = "127.0.0.1"
PORT = 8188
URL = f"http://{HOST}:{PORT}"


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


def _run_server() -> None:
    from backend.main import app

    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    # uvicorn only installs OS signal handlers on the main thread, so running its
    # server loop here (a background thread) is safe alongside pywebview's own loop.
    server.run()


def _wait_for_server() -> None:
    for _ in range(120):
        try:
            if requests.get(URL, timeout=0.5).ok:
                return
        except requests.RequestException:
            time.sleep(0.25)
    raise RuntimeError("The embedded PixelFlow server did not start")


def main() -> None:
    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()
    _wait_for_server()

    _configure_webview2_runtime()
    webview.create_window("PixelFlow", URL, width=1440, height=960, min_size=(1024, 700))
    webview.start(gui="edgechromium")


if __name__ == "__main__":
    main()
