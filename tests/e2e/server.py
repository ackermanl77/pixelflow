"""Starts the PixelFlow backend for the Playwright smoke test (report item
T3): test mode (no torch, no model loads) with a throwaway data folder
(<temp>/pixelflow-e2e-<port>, emptied at every start - Playwright kills the
server, on Windows without a chance to clean up), so the run never touches
the real outputs/, workflows/ or history.
Playwright's webServer (playwright.config.js) starts and stops it.

    python tests/e2e/server.py [port]
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8299
    home = Path(tempfile.gettempdir()) / f"pixelflow-e2e-{port}"
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir(parents=True)
    os.environ.update(
        PIXELFLOW_TEST_MODE="1",
        PIXELFLOW_HOME=str(home),
        PIXELFLOW_MODELS_DIR=str(home / "models"),
        PIXELFLOW_PORT=str(port),
        PIXELFLOW_COMFYUI_MODEL_PATHS="0",  # never the real ComfyUI library (B25)
    )
    sys.path.insert(0, str(ROOT))
    import uvicorn  # after the environment: backend.paths reads it at import

    uvicorn.run("backend.main:app", host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
