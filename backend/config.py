"""Constants shared by run.py, desktop.py and the server (one place instead
of three copies)."""

import os

# Returned by /api/health so desktop.py can tell this server apart from some
# other app (e.g. ComfyUI) already listening on the same port.
APP_ID = "pixelflow"
HOST = "127.0.0.1"
# Not ComfyUI's default 8188: people who run PixelFlow are likely to have
# ComfyUI running too. Override with PIXELFLOW_PORT.
DEFAULT_PORT = 8288


def port() -> int:
    try:
        value = int(os.environ.get("PIXELFLOW_PORT", DEFAULT_PORT))
    except ValueError:
        return DEFAULT_PORT
    return value if 0 < value < 65536 else DEFAULT_PORT


def app_version() -> str:
    """The version scripts/build_release.ps1 stamps into backend/_version.py,
    or "dev" when running from source."""
    try:
        from ._version import __version__  # type: ignore[import-not-found]
    except ImportError:
        return "dev"
    return str(__version__)
