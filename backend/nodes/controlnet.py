"""ControlNet (report item B7): guide generation with an edge, depth or pose
image. Load ControlNet reads a model from models/controlnet/ (a diffusers
folder with config.json, or a single .safetensors file) or a Hugging Face
repo id; Apply ControlNet attaches it, with its control image and strength,
to a CONDITIONING. Every sampler then runs the matching diffusers ControlNet
pipeline, derived with from_pipe() from the loaded checkpoint so the base
weights are shared (see generation.py). SD 1.x/2.x and SDXL ControlNets are
supported (diffusers' ControlNetModel); the model must match the checkpoint's
family.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

from .. import paths
from .base import NodeBase, as_image_list, register_node

logger = logging.getLogger(__name__)

CONTROLNET_EXTENSIONS = (".safetensors",)
MAX_CACHED_CONTROLNETS = 2


def controlnet_dir() -> Path:
    return paths.CONTROLNET_DIR


def list_controlnets() -> List[str]:
    """Names under models/controlnet/: .safetensors files and diffusers
    folders (a config.json next to the weights), '/'-separated."""
    folder = controlnet_dir()
    if not folder.is_dir():
        return []
    found = []
    for current, dirnames, filenames in os.walk(folder):
        current_path = Path(current)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if current_path != folder and (current_path / "config.json").is_file():
            found.append(current_path.relative_to(folder).as_posix())
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            if not name.startswith(".") and name.lower().endswith(CONTROLNET_EXTENSIONS):
                found.append((current_path / name).relative_to(folder).as_posix())
    return sorted(found, key=str.lower)


def _local_path(name: str) -> Path | None:
    try:
        path = paths.safe_path(controlnet_dir(), name)
    except paths.PathTraversalError:
        return None
    return path if path.exists() else None


_cache: "OrderedDict[str, object]" = OrderedDict()
_cache_lock = threading.Lock()


def load_controlnet(name: str):
    """A diffusers ControlNetModel for `name` (cached, most recent
    MAX_CACHED_CONTROLNETS), loaded on the CPU in fp32; the sampler moves it
    to the pipeline's device and dtype."""
    with _cache_lock:
        if name in _cache:
            _cache.move_to_end(name)
            return _cache[name]
    from diffusers import ControlNetModel

    local = _local_path(name)
    try:
        if local is not None and local.is_file():
            model = ControlNetModel.from_single_file(str(local))
        else:
            model = ControlNetModel.from_pretrained(str(local) if local is not None else name)
    except OSError as error:
        raise RuntimeError(
            f"ControlNet '{name}' couldn't be loaded: put a diffusers ControlNet folder or a .safetensors file in "
            f"models/controlnet/, or use a Hugging Face repo id ({error})"
        ) from error
    with _cache_lock:
        _cache[name] = model
        while len(_cache) > MAX_CACHED_CONTROLNETS:
            _cache.popitem(last=False)
    return model


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


@register_node("Load ControlNet")
class ControlNetLoader(NodeBase):
    """Loads a ControlNet from models/controlnet/ or the Hugging Face Hub
    (e.g. lllyasviel/sd-controlnet-canny for SD 1.5 checkpoints,
    thibaud/controlnet-sd21-canny-diffusers for SD 2.1 / sd-turbo)."""

    CATEGORY = "loaders"
    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load"
    DESCRIPTION = "Loads a ControlNet model (models/controlnet/ or a Hub repo id)."

    @classmethod
    def INPUT_TYPES(cls):
        names = list_controlnets()
        return {
            "required": {
                "control_net_name": (
                    names or [""],
                    {"allow_custom": True, "tooltip": "A model in models/controlnet/, or a Hugging Face repo id"},
                ),
            }
        }

    def load(self, control_net_name):
        if not control_net_name:
            raise ValueError("choose a ControlNet: put one in models/controlnet/ or type a Hugging Face repo id")
        return ({"name": control_net_name, "model": load_controlnet(control_net_name)},)


@register_node("Apply ControlNet")
class ControlNetApply(NodeBase):
    """Attaches a ControlNet and its control image (e.g. Canny Edges of a
    photo) to the positive CONDITIONING. Chain several for several
    ControlNets. `strength` scales its influence; `start`/`end` limit it to
    part of the sampling (0-1 of the steps)."""

    CATEGORY = "conditioning"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    DESCRIPTION = "Guides sampling with a ControlNet and a control image."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def apply(self, conditioning, control_net, image, strength=1.0, start=0.0, end=1.0):
        if float(end) <= float(start):
            raise ValueError(f"end ({float(end):g}) must be after start ({float(start):g})")
        control = {
            "net": control_net,
            "image": as_image_list(image)[0].convert("RGB"),
            "strength": float(strength),
            "start": float(start),
            "end": float(end),
        }
        return ({**conditioning, "controls": list(conditioning.get("controls", [])) + [control]},)


# ---- used by the samplers (generation.py) ----


def controls_of(positive) -> List[Dict]:
    return list((positive or {}).get("controls", []))


def control_pipeline(pipe, kind: str, controls):
    """`pipe` re-wired with the ControlNet(s) of `controls`: the diffusers
    ControlNet variant of `kind` ("text2img", "img2img" or "inpaint"),
    sharing every base component. The ControlNets are moved to the
    pipeline's device and dtype."""
    import diffusers

    unet = getattr(pipe, "unet", None)
    if unet is None:
        raise ValueError("ControlNet works with SD 1.x/2.x and SDXL checkpoints (this one has no UNet)")
    nets = []
    for control in controls:
        net = control["net"]["model"]
        nets.append(net.to(device=unet.device, dtype=unet.dtype))
    factory = {
        "text2img": diffusers.AutoPipelineForText2Image,
        "img2img": diffusers.AutoPipelineForImage2Image,
        "inpaint": diffusers.AutoPipelineForInpainting,
    }[kind]
    derived = factory.from_pipe(pipe, controlnet=nets[0] if len(nets) == 1 else nets)
    derived.set_progress_bar_config(disable=True)
    return derived


def control_kwargs(kind: str, controls, size) -> Dict:
    """The ControlNet call arguments: the control image(s) at the output
    size, their scales and step windows. A txt2img ControlNet pipeline takes
    the control image as `image`; img2img and inpaint take it as
    `control_image` (their `image` is the picture being re-sampled)."""
    from PIL import Image

    images = [control["image"].resize(size, Image.LANCZOS) for control in controls]
    single = len(controls) == 1

    def one_or_list(values):
        return values[0] if single else values

    return {
        ("image" if kind == "text2img" else "control_image"): one_or_list(images),
        "controlnet_conditioning_scale": one_or_list([c["strength"] for c in controls]),
        "control_guidance_start": one_or_list([c["start"] for c in controls]),
        "control_guidance_end": one_or_list([c["end"] for c in controls]),
    }
