import re
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
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"

for _dir in (OUTPUT_DIR, UPLOAD_DIR, WORKFLOW_DIR, CHECKPOINT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


class PathTraversalError(ValueError):
    """Raised when a user-supplied name would resolve outside its base directory."""


def safe_path(base: Path, name: str) -> Path:
    """Resolve `name` against `base`, rejecting any result that escapes `base`
    (via '..' segments or an absolute path, e.g. from an untrusted request)."""
    base_resolved = base.resolve()
    candidate = (base_resolved / name).resolve()
    if candidate != base_resolved and not candidate.is_relative_to(base_resolved):
        raise PathTraversalError(f"'{name}' resolves outside {base}")
    return candidate


_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.\-]+")
MAX_FILENAME_STEM = 100


def sanitize_filename(name: str, default: str = "image") -> str:
    """Whitelist-sanitize an untrusted filename (or filename prefix) down to
    word characters, '.', and '-'. Keeping only the final path component
    isn't enough on Windows: a name like 'out.png:evil' or 'a?b' survives
    Path(...).name but writes an NTFS alternate data stream or fails with
    OSError. Leading/trailing dots are stripped too (so the result can never
    be '.', '..', or a hidden dotfile). Falls back to `default` if nothing
    usable is left."""
    last = re.split(r"[\\/]", name or "")[-1]
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", last).strip("._")
    return cleaned[:MAX_FILENAME_STEM] or default
