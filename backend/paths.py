"""Where PixelFlow reads its bundled files and keeps its writable data.

Writable data (outputs, uploads, workflows, models, history, logs) lives
under ROOT_DIR, resolved in this order:

1. the PIXELFLOW_HOME environment variable, if set;
2. running from source: the project folder (unchanged dev layout);
3. the packaged app with a `portable.txt` marker next to PixelFlow.exe, or
   an existing pre-0.2 data folder there (an `outputs` folder or
   `history.jsonl`): the exe's own folder ("portable" mode);
4. otherwise %LOCALAPPDATA%\\PixelFlow (~/.local/share/pixelflow elsewhere),
   so an install under C:\\Program Files (read-only) still starts.

Models can live somewhere else (e.g. a big non-synced drive): set
PIXELFLOW_MODELS_DIR, or {"models_dir": "D:/sd-models"} in
ROOT_DIR/settings.json. Nothing is created at import time; ensure_data_dirs()
does that (best effort) at server startup.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

APP_DIR_NAME = "PixelFlow"
PORTABLE_MARKER = "portable.txt"
SETTINGS_FILENAME = "settings.json"


def _bundle_root() -> Path:
    """Where read-only bundled resources (the frontend/ static files) live.
    Under PyInstaller these are extracted to a temp dir (sys._MEIPASS)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def _user_data_root() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_DIR_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_DIR_NAME.lower()


def resolve_data_root(environ=None, frozen=None, executable=None) -> Path:
    """See the module docstring. Parameters exist for the tests."""
    environ = os.environ if environ is None else environ
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    home = environ.get("PIXELFLOW_HOME")
    if home:
        return Path(home).expanduser().resolve()
    if not frozen:
        return Path(__file__).resolve().parent.parent
    exe_dir = Path(executable or sys.executable).resolve().parent
    legacy = (exe_dir / "outputs").is_dir() or (exe_dir / "history.jsonl").is_file()
    if (exe_dir / PORTABLE_MARKER).is_file() or legacy:
        return exe_dir
    return _user_data_root()


def load_settings(root: Path) -> Dict[str, Any]:
    """ROOT_DIR/settings.json as a dict ({} when missing or unreadable)."""
    try:
        data = json.loads((root / SETTINGS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_models_dir(root: Path, environ=None) -> Path:
    environ = os.environ if environ is None else environ
    configured = environ.get("PIXELFLOW_MODELS_DIR") or load_settings(root).get("models_dir")
    if isinstance(configured, str) and configured.strip():
        return Path(configured.strip()).expanduser().resolve()
    return root / "models"


BUNDLE_DIR = _bundle_root()
ROOT_DIR = resolve_data_root()

FRONTEND_DIR = BUNDLE_DIR / "frontend"
OUTPUT_DIR = ROOT_DIR / "outputs"
UPLOAD_DIR = ROOT_DIR / "uploads"
WORKFLOW_DIR = ROOT_DIR / "workflows"
# Preview Image writes here; emptied at every server start (see main.py).
TEMP_DIR = ROOT_DIR / "temp"
MODELS_DIR = resolve_models_dir(ROOT_DIR)
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
LORA_DIR = MODELS_DIR / "loras"
EMBEDDING_DIR = MODELS_DIR / "embeddings"
CONTROLNET_DIR = MODELS_DIR / "controlnet"
UPSCALE_MODEL_DIR = MODELS_DIR / "upscale_models"
# Split-file models (B26), ComfyUI's layout: the diffusion model (UNet/transformer),
# its text encoders and its VAE as separate single files.
DIFFUSION_MODEL_DIR = MODELS_DIR / "diffusion_models"
TEXT_ENCODER_DIR = MODELS_DIR / "text_encoders"
VAE_DIR = MODELS_DIR / "vae"
# IP-Adapter (B20): the adapters and the CLIP vision image encoders they use.
IPADAPTER_DIR = MODELS_DIR / "ipadapter"
CLIP_VISION_DIR = MODELS_DIR / "clip_vision"
# Finished prompts, one JSON object per line (see engine/queue.py).
HISTORY_PATH = ROOT_DIR / "history.jsonl"
# The output cache's copy on disk (engine.disk_cache): the last run's node outputs.
OUTPUT_CACHE_DIR = ROOT_DIR / "cache" / "outputs"
# Rotating log files (see backend/logging_setup.py); created on first use.
LOG_DIR = ROOT_DIR / "logs"
# Prompt wildcards: __name__ picks a line of wildcards/name.txt (nodes/wildcards.py).
WILDCARD_DIR = ROOT_DIR / "wildcards"
# Custom node plugins, imported at startup (backend/plugins.py).
CUSTOM_NODES_DIR = ROOT_DIR / "custom_nodes"


def ensure_data_dirs() -> None:
    """Create the data folders. Failures are logged, not raised: a
    read-only location must not stop the server from starting."""
    for folder in (OUTPUT_DIR, UPLOAD_DIR, WORKFLOW_DIR, TEMP_DIR, CHECKPOINT_DIR, LORA_DIR, EMBEDDING_DIR, CONTROLNET_DIR,
                   UPSCALE_MODEL_DIR, DIFFUSION_MODEL_DIR, TEXT_ENCODER_DIR, VAE_DIR, IPADAPTER_DIR, CLIP_VISION_DIR,
                   WILDCARD_DIR, CUSTOM_NODES_DIR):
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("could not create data folder %s", folder, exc_info=True)


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
