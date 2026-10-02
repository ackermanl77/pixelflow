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


def test_mode() -> bool:
    """PIXELFLOW_TEST_MODE=1: a lightweight server for the browser smoke
    test (tests/e2e, report item T3). torch/diffusers are never imported (no
    background warm-up, /api/system reports no device) and loading a model
    fails with a clear error, so the server starts in about a second and
    needs only the web dependencies. Torch-free graphs (Empty Image -> Save
    Image, ...) still run normally."""
    return os.environ.get("PIXELFLOW_TEST_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def app_version() -> str:
    """The version scripts/build_release.ps1 stamps into backend/_version.py,
    or "dev" when running from source."""
    try:
        from ._version import __version__  # type: ignore[import-not-found]
    except ImportError:
        return "dev"
    return str(__version__)
