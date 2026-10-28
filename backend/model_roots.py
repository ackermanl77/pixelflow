"""Extra model folders shared with ComfyUI (B25).

PixelFlow's own models/<kind>/ folders come first. After them come the folders a
ComfyUI `extra_model_paths.yaml` assigns to the same kind, so a library that
ComfyUI already uses (here D:/models, 321 GB) isn't duplicated.

Which yaml files are read, in order:

1. PIXELFLOW_COMFYUI_MODEL_PATHS: yaml paths separated by os.pathsep; "0" or
   "off" turns the feature off;
2. otherwise "comfyui_model_paths" in ROOT_DIR/settings.json: a path, a list
   of paths, or false to turn it off;
3. otherwise the ones found automatically: every Comfy Desktop install's
   ComfyUI/extra_model_paths.yaml under
   %LOCALAPPDATA%/Comfy-Desktop/ComfyUI-Installs/*/.

The yaml uses ComfyUI's format. Each top-level section has an optional
`base_path` (relative to the yaml's folder) and `<kind>: <folder>` entries;
a `|` block lists one folder per line. ComfyUI kinds map onto PixelFlow kinds
through COMFY_KINDS, e.g. diffusion_models also reads ComfyUI's `unet`
folders. The result is rescanned at most every 30 s.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import paths

logger = logging.getLogger(__name__)

# PixelFlow kind -> the ComfyUI yaml keys whose folders it reads.
COMFY_KINDS: Dict[str, tuple] = {
    "checkpoints": ("checkpoints",),
    "loras": ("loras",),
    "embeddings": ("embeddings",),
    "controlnet": ("controlnet",),
    "upscale_models": ("upscale_models",),
    "diffusion_models": ("diffusion_models", "unet"),
    "text_encoders": ("text_encoders", "clip"),
    "vae": ("vae",),
    "ipadapter": ("ipadapter",),
    "clip_vision": ("clip_vision",),
}

_TTL_S = 30.0
_cache: Dict[str, object] = {"at": -1e9, "roots": {}, "sources": []}


def _own_dir(kind: str) -> Path:
    return {
        "checkpoints": paths.CHECKPOINT_DIR,
        "loras": paths.LORA_DIR,
        "embeddings": paths.EMBEDDING_DIR,
        "controlnet": paths.CONTROLNET_DIR,
        "upscale_models": paths.UPSCALE_MODEL_DIR,
        "diffusion_models": paths.DIFFUSION_MODEL_DIR,
        "text_encoders": paths.TEXT_ENCODER_DIR,
        "vae": paths.VAE_DIR,
        "ipadapter": paths.IPADAPTER_DIR,
        "clip_vision": paths.CLIP_VISION_DIR,
    }[kind]


def _disabled(value) -> bool:
    return value is False or (isinstance(value, str) and value.strip().lower() in ("0", "off", "false", "no"))


def config_files(environ=None, settings=None, local_appdata: Optional[str] = None) -> List[Path]:
    """The yaml files to read (see the module docstring). Parameters exist for the tests."""
    environ = os.environ if environ is None else environ
    settings = paths.load_settings(paths.ROOT_DIR) if settings is None else settings
    configured = environ.get("PIXELFLOW_COMFYUI_MODEL_PATHS")
    if configured is not None:
        if _disabled(configured):
            return []
        return [Path(p).expanduser() for p in configured.split(os.pathsep) if p.strip()]
    if "comfyui_model_paths" in settings:
        value = settings["comfyui_model_paths"]
        if _disabled(value) or value is None:
            return []
        values = value if isinstance(value, list) else [value]
        return [Path(str(p)).expanduser() for p in values if str(p).strip()]
    base = local_appdata if local_appdata is not None else environ.get("LOCALAPPDATA")
    if not base:
        return []
    installs = Path(base) / "Comfy-Desktop" / "ComfyUI-Installs"
    try:
        return sorted(installs.glob("*/ComfyUI/extra_model_paths.yaml"))
    except OSError:
        return []


def parse_extra_model_paths(text: str, yaml_dir: Path) -> Dict[str, List[Path]]:
    """ComfyUI yaml text -> {ComfyUI kind: [folders]}, folders resolved the way
    ComfyUI resolves them (base_path relative to the yaml, entries relative to
    base_path). Malformed sections are skipped."""
    import yaml

    data = yaml.safe_load(text) or {}
    found: Dict[str, List[Path]] = {}
    if not isinstance(data, dict):
        return found
    for section in data.values():
        if not isinstance(section, dict):
            continue
        base = section.get("base_path")
        base_path = None
        if isinstance(base, str) and base.strip():
            base_path = Path(os.path.expandvars(base.strip())).expanduser()
            if not base_path.is_absolute():
                base_path = yaml_dir / base_path
        for key, value in section.items():
            if key in ("base_path", "is_default") or not isinstance(value, str):
                continue
            for line in value.splitlines():
                entry = line.strip()
                if not entry:
                    continue
                folder = Path(os.path.expandvars(entry)).expanduser()
                if not folder.is_absolute():
                    folder = (base_path or yaml_dir) / folder
                found.setdefault(str(key), []).append(folder)
    return found


def _scan() -> Dict[str, List[Path]]:
    roots: Dict[str, List[Path]] = {kind: [] for kind in COMFY_KINDS}
    sources = []
    for config in config_files():
        try:
            parsed = parse_extra_model_paths(config.read_text(encoding="utf-8"), config.parent)
        except (OSError, ValueError, ImportError) as error:
            logger.warning("could not read ComfyUI model paths from %s: %s", config, error)
            continue
        except Exception as error:  # noqa: BLE001 - yaml.YAMLError and friends
            logger.warning("could not parse ComfyUI model paths in %s: %s", config, error)
            continue
        sources.append(str(config))
        for kind, comfy_keys in COMFY_KINDS.items():
            for key in comfy_keys:
                for folder in parsed.get(key, []):
                    if folder not in roots[kind]:
                        roots[kind].append(folder)
    _cache["sources"] = sources
    return roots


def extra_roots(kind: str) -> List[Path]:
    """ComfyUI's existing folders for `kind` (not PixelFlow's own)."""
    now = time.monotonic()
    if now - float(_cache["at"]) >= _TTL_S:
        _cache["roots"] = _scan()
        _cache["at"] = now
    own = _own_dir(kind).resolve()
    result = []
    for folder in _cache["roots"].get(kind, []):  # type: ignore[union-attr]
        try:
            if folder.is_dir() and folder.resolve() != own:
                result.append(folder)
        except OSError:
            continue
    return result


def reset_cache() -> None:
    _cache.update(at=-1e9, roots={}, sources=[])


def sources() -> List[str]:
    """The yaml files the last scan read (GET /api/system shows them)."""
    extra_roots("checkpoints")
    return list(_cache["sources"])  # type: ignore[arg-type]


def model_dirs(kind: str) -> List[Path]:
    """Where models of `kind` are looked for, in priority order."""
    return [_own_dir(kind), *extra_roots(kind)]


def find(kind: str, name: str) -> Optional[Path]:
    """`name` (a '/'-separated relative name) in the first folder of `kind`
    that has it, or None. safe_path() keeps '..' or absolute names inside
    each folder."""
    if not name:
        return None
    for folder in model_dirs(kind):
        try:
            candidate = paths.safe_path(folder, name)
        except paths.PathTraversalError:
            return None
        if candidate.exists():
            return candidate
    return None


def merge_names(lists: Iterable[List[str]]) -> List[str]:
    """Names from several folders, the first folder's entry winning a clash."""
    seen, merged = set(), []
    for names in lists:
        for name in names:
            if name.lower() not in seen:
                seen.add(name.lower())
                merged.append(name)
    return sorted(merged, key=str.lower)
