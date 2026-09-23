import logging
import threading
import uuid
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageColor

from .base import NodeBase, register_node
from ..paths import CHECKPOINT_DIR, OUTPUT_DIR, PathTraversalError, UPLOAD_DIR, safe_path

logger = logging.getLogger(__name__)

# Constructing a pipeline is expensive (multi-second, multi-GB of weights), so
# loaded pipelines are cached in-process per (ckpt_name, device) for the life
# of the server rather than reloaded on every node execution.
_pipeline_cache: dict[tuple[str, str], StableDiffusionPipeline] = {}
_pipeline_cache_lock = threading.Lock()


def _resolve_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _local_checkpoint_path(ckpt_name: str) -> Path | None:
    """Resolve `ckpt_name` to a local file/folder under CHECKPOINT_DIR, or
    None if it isn't one (e.g. it's a Hugging Face Hub repo id instead).
    Uses safe_path so a malicious ckpt_name can't read files outside
    CHECKPOINT_DIR via '..' or an absolute path."""
    try:
        candidate = safe_path(CHECKPOINT_DIR, ckpt_name)
    except PathTraversalError:
        return None
    return candidate if candidate.exists() else None


def _load_pipeline(ckpt_name: str) -> StableDiffusionPipeline:
    device, dtype = _resolve_device_and_dtype()
    cache_key = (ckpt_name, device)
    with _pipeline_cache_lock:
        cached = _pipeline_cache.get(cache_key)
        if cached is not None:
            return cached

        local_path = _local_checkpoint_path(ckpt_name)
        if local_path is not None and local_path.is_file():
            logger.info("loading checkpoint from local file %s", local_path)
            pipe = StableDiffusionPipeline.from_single_file(str(local_path), torch_dtype=dtype)
        elif local_path is not None and local_path.is_dir():
            logger.info("loading checkpoint from local directory %s", local_path)
            pipe = StableDiffusionPipeline.from_pretrained(str(local_path), torch_dtype=dtype)
        else:
            logger.info("loading checkpoint '%s' from the Hugging Face Hub", ckpt_name)
            pipe = StableDiffusionPipeline.from_pretrained(ckpt_name, torch_dtype=dtype)

        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        _pipeline_cache[cache_key] = pipe
        return pipe


@register_node("Load Checkpoint")
class LoadCheckpoint(NodeBase):
    """Loads a real Stable Diffusion pipeline via diffusers. `ckpt_name` is
    resolved, in order: a local file (.safetensors/.ckpt) under
    models/checkpoints/, a local diffusers-format folder under the same
    directory, or otherwise a Hugging Face Hub repo id (downloaded and
    cached on first use - requires internet access the first time).
    """

    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": ("STRING", {"default": "stabilityai/sd-turbo"})}}

    def load(self, ckpt_name):
        pipe = _load_pipeline(ckpt_name)
        return ({"name": ckpt_name, "pipeline": pipe},)


@register_node("Empty Image")
class EmptyImage(NodeBase):
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "color": ("STRING", {"default": "#202020"}),
            }
        }

    def create(self, width, height, color):
        rgb = ImageColor.getrgb(color)
        return (Image.new("RGB", (int(width), int(height)), rgb),)


@register_node("Load Image")
class LoadImage(NodeBase):
    CATEGORY = "loaders"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filename": ("STRING", {"default": ""})}}

    def load(self, filename):
        path = safe_path(UPLOAD_DIR, filename)
        return (Image.open(path).convert("RGB"),)


def _save(image, prefix, folder):
    folder.mkdir(exist_ok=True)
    safe_prefix = Path(prefix).name or "image"
    name = f"{safe_prefix}_{uuid.uuid4().hex[:8]}.png"
    path = safe_path(folder, name)
    image.save(path)
    return path.name


@register_node("Preview Image")
class PreviewImage(NodeBase):
    CATEGORY = "output"
    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def preview(self, image):
        name = _save(image, "preview", OUTPUT_DIR)
        return {"images": [name]}


@register_node("Save Image")
class SaveImage(NodeBase):
    CATEGORY = "output"
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "PixelFlow"}),
            }
        }

    def save(self, image, filename_prefix):
        name = _save(image, filename_prefix, OUTPUT_DIR)
        return {"images": [name]}
