import gc
import json
import logging
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageColor
from PIL.PngImagePlugin import PngInfo

from .base import NodeBase, as_image_list, register_node
from ..paths import CHECKPOINT_DIR, OUTPUT_DIR, PathTraversalError, UPLOAD_DIR, safe_path, sanitize_filename

logger = logging.getLogger(__name__)

# Constructing a pipeline is expensive (multi-second, multi-GB of weights), so
# loaded pipelines are cached in-process per (ckpt_name, device) rather than
# reloaded on every node execution. The cache is a small LRU: keeping every
# checkpoint ever used resident would pin all of them in VRAM for the life of
# the server, so loading a new one past MAX_CACHED_PIPELINES evicts the least
# recently used and frees its memory.
MAX_CACHED_PIPELINES = 1
_pipeline_cache: "OrderedDict[tuple[str, str], StableDiffusionPipeline]" = OrderedDict()
_pipeline_cache_lock = threading.Lock()

# torch >= 2.5 exposes torch.OutOfMemoryError; older builds only have the
# CUDA-specific name (the same class).
OutOfMemoryError = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def free_memory() -> None:
    """Release unreferenced tensors and hand cached CUDA blocks back to the
    driver. A no-op on CPU-only machines beyond the gc pass."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evict_lru_pipelines(keep: int) -> None:
    """Drop least-recently-used pipelines until at most `keep` remain.
    Caller must hold _pipeline_cache_lock."""
    evicted = False
    while len(_pipeline_cache) > keep:
        key, _pipe = _pipeline_cache.popitem(last=False)
        logger.info("evicting cached pipeline %s to free memory", key)
        del _pipe
        evicted = True
    if evicted:
        free_memory()


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
            _pipeline_cache.move_to_end(cache_key)
            return cached

        # Free room *before* loading, so the old and new weights never need
        # to fit in (V)RAM at the same time.
        _evict_lru_pipelines(MAX_CACHED_PIPELINES - 1)

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

        out_of_memory = False
        try:
            pipe = pipe.to(device)
        except OutOfMemoryError:
            out_of_memory = True  # freed outside the except block (see KSampler)
        if out_of_memory:
            del pipe
            free_memory()
            raise RuntimeError(
                f"Out of GPU memory while loading checkpoint '{ckpt_name}'. Close other GPU "
                "applications or use a smaller checkpoint."
            )
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


# Hidden inputs (never shown as widgets/sockets) the engine injects into the
# output nodes, the same way ComfyUI embeds the graph into saved PNGs.
_METADATA_HIDDEN_INPUTS = {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}


def _png_metadata(prompt, extra_pnginfo):
    """PNG tEXt chunks for a saved image: `prompt` is the executed graph
    (with any random seed replaced by the seed actually used, so it can be
    re-queued to reproduce the image), plus each `extra_pnginfo` entry
    (the frontend sends its litegraph `workflow` there)."""
    if not prompt and not extra_pnginfo:
        return None
    info = PngInfo()
    if prompt:
        info.add_text("prompt", json.dumps(prompt))
    for key, value in (extra_pnginfo or {}).items():
        info.add_text(str(key), json.dumps(value))
    return info


def _save(image, prefix, folder, pnginfo=None):
    folder.mkdir(exist_ok=True)
    name = f"{sanitize_filename(prefix)}_{uuid.uuid4().hex[:8]}.png"
    path = safe_path(folder, name)
    image.save(path, pnginfo=pnginfo)
    return path.name


@register_node("Preview Image")
class PreviewImage(NodeBase):
    CATEGORY = "output"
    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}, "hidden": dict(_METADATA_HIDDEN_INPUTS)}

    def preview(self, image, prompt=None, extra_pnginfo=None):
        pnginfo = _png_metadata(prompt, extra_pnginfo)
        names = [_save(img, "preview", OUTPUT_DIR, pnginfo) for img in as_image_list(image)]
        return {"images": names}


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
            },
            "hidden": dict(_METADATA_HIDDEN_INPUTS),
        }

    def save(self, image, filename_prefix, prompt=None, extra_pnginfo=None):
        pnginfo = _png_metadata(prompt, extra_pnginfo)
        names = [_save(img, filename_prefix, OUTPUT_DIR, pnginfo) for img in as_image_list(image)]
        return {"images": names}
