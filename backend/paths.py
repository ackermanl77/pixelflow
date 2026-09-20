import sys
from pathlib import Path


def _bundle_root() -> Path:
    """Where read-only bundled resources (the frontend/ static files) live.
    Under PyInstaller these are extracted to a temp dir (sys._MEIPASS)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    """Where writable user data (outputs/uploads/workflows) lives. This must
    be a stable, persistent location next to the actual .exe, not the
    temp extraction dir _bundle_root() uses when frozen."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BUNDLE_DIR = _bundle_root()
ROOT_DIR = _data_root()

FRONTEND_DIR = BUNDLE_DIR / "frontend"
OUTPUT_DIR = ROOT_DIR / "outputs"
UPLOAD_DIR = ROOT_DIR / "uploads"
WORKFLOW_DIR = ROOT_DIR / "workflows"

for _dir in (OUTPUT_DIR, UPLOAD_DIR, WORKFLOW_DIR):
    _dir.mkdir(exist_ok=True)
