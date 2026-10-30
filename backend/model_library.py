"""The Model Library panel (UI13): installed checkpoints, LoRAs, VAEs and ControlNets, each with its
size and preview image, and the loader node that uses it (dragging one onto the canvas creates that
node with the model chosen).

A preview image is the one model managers (ComfyUI-Manager, Civitai helpers) put beside a model:
<name>.preview.png, or <name>.png / .jpg / .jpeg / .webp with the model's file name minus its
extension; for a diffusers folder, preview.png (etc.) inside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from . import model_roots
from .nodes import models
from .nodes.controlnet import list_controlnets

# kind -> (heading, loader node type, the loader's input that takes the model)
LIBRARY_KINDS = {
    "checkpoints": ("Checkpoints", "LoadCheckpoint", "ckpt_name"),
    "loras": ("LoRAs", "LoraLoader", "lora_name"),
    "vae": ("VAEs", "LoadSplitModel", "vae"),
    "controlnet": ("ControlNets", "ControlNetLoader", "control_net_name"),
}
PREVIEW_SUFFIXES = (".preview.png", ".png", ".jpg", ".jpeg", ".webp")


def _names(kind: str) -> List[str]:
    return list_controlnets() if kind == "controlnet" else models.list_models(kind)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def preview_path(kind: str, name: str) -> Optional[Path]:
    """The preview image of an installed model, or None (none there, or not a local model)."""
    if kind not in LIBRARY_KINDS:
        return None
    path = model_roots.find(kind, name)
    if path is None:
        return None
    if path.is_dir():
        candidates = [path / f"preview{suffix}" for suffix in PREVIEW_SUFFIXES[1:]]
    else:
        stem = path.with_suffix("")
        candidates = [stem.parent / f"{stem.name}{suffix}" for suffix in PREVIEW_SUFFIXES]
    return next((c for c in candidates if c.is_file()), None)


def library() -> Dict[str, Any]:
    """GET /api/model-library: per kind, its heading, loader and installed models (name, size in
    bytes or None for a Hub model, whether it has a preview). Hub repo ids (not on disk) are listed
    too: the loader fetches them if they aren't cached."""
    out: Dict[str, Any] = {}
    for kind, (label, node, input_name) in LIBRARY_KINDS.items():
        items = []
        for name in _names(kind):
            if not name:
                continue
            path = model_roots.find(kind, name)
            items.append({
                "name": name,
                "size": _size(path) if path is not None else None,
                "preview": preview_path(kind, name) is not None,
            })
        out[kind] = {"label": label, "node": node, "input": input_name, "items": items}
    return out
