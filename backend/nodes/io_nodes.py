import copy
import datetime
import json
import logging
import re
import threading
import uuid
from pathlib import Path

from PIL import Image, ImageColor, ImageOps
from PIL.PngImagePlugin import PngInfo

from . import models, runtime
from .base import NodeBase, as_image_list, register_node
from ..paths import OUTPUT_DIR, TEMP_DIR, UPLOAD_DIR, PathTraversalError, safe_path, sanitize_filename

logger = logging.getLogger(__name__)

# Kept under their old names for callers/tests that import them from here.
free_memory = runtime.free_memory
enable_vae_slicing = models.enable_vae_slicing


def __getattr__(name):
    # torch's OOM exception class, resolved lazily so importing this module
    # doesn't import torch (report item P6).
    if name == "OutOfMemoryError":
        return runtime.oom_error()
    raise AttributeError(name)


def _load_pipeline(ckpt_name: str, dtype: str = "auto", on_progress=None):
    """The cached (or freshly loaded) pipeline for a checkpoint name."""
    return models.PIPELINES.get(ckpt_name, dtype, on_progress).pipe


@register_node("Load Checkpoint")
class LoadCheckpoint(NodeBase):
    """Loads a diffusers pipeline. `ckpt_name` is a file (.safetensors /
    .ckpt) or diffusers folder under models/checkpoints/, or a Hugging Face
    Hub repo id (downloaded on first use, with progress). The architecture
    (SD 1.x/2.x, SDXL, SD3, Flux) is detected from the files. `dtype` picks
    the precision; "auto" is fp16 on CUDA for SD/SDXL, bf16 for SD3/Flux,
    fp32 on CPU."""

    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    DESCRIPTION = "Loads a Stable Diffusion checkpoint (local file/folder or Hugging Face Hub id)."
    # The pipeline already lives in models.PIPELINES; keeping another
    # reference in the engine's output cache would stop an evicted pipeline
    # from actually being freed. (Re-running this node is a cheap cache hit.)
    CACHE_OUTPUTS = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # A combo of what's installed, but any other value (a hub id
                # typed via "Other…") is accepted too: allow_custom.
                "ckpt_name": (
                    models.list_models("checkpoints"),
                    {
                        "default": models.DEFAULT_CHECKPOINT,
                        "allow_custom": True,
                        "tooltip": "A file or diffusers folder in models/checkpoints/, or a Hugging Face Hub repo id",
                    },
                ),
            },
            "optional": {
                "dtype": (
                    list(models.DTYPE_CHOICES),
                    {"default": "auto", "tooltip": "Precision; auto = fp16 (SD/SDXL) or bf16 (SD3/Flux) on GPU"},
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, ckpt_name="", **_):
        # A local checkpoint file replaced under the same name must not be
        # served from downstream nodes' cached outputs.
        local_path = models.local_model_path("checkpoints", ckpt_name)
        if local_path is not None and local_path.is_file():
            stat = local_path.stat()
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        return ""

    @classmethod
    def VALIDATE_INPUTS(cls, ckpt_name=None, **_):
        if ckpt_name is not None and not str(ckpt_name).strip():
            return "ckpt_name is empty - pick a checkpoint or type a Hugging Face Hub id"
        return True

    def load(self, ckpt_name, dtype="auto", on_progress=None):
        entry = models.PIPELINES.get(ckpt_name, dtype, on_progress)
        return ({"name": ckpt_name, "pipeline": entry.pipe, "arch": entry.arch, "dtype": entry.dtype, "loras": []},)


def _validate_color(color):
    try:
        ImageColor.getrgb(color)
    except (ValueError, AttributeError):
        return f"'{color}' is not a colour (use e.g. #202020, rgb(32,32,32) or a name like 'black')"
    return True


@register_node("Empty Image")
class EmptyImage(NodeBase):
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create"
    DESCRIPTION = "A blank image; as a sampler's latent_image it only sets the output size."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "color": ("STRING", {"default": "#202020", "tooltip": "Any CSS-style colour"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, color=None, **_):
        return True if color is None else _validate_color(color)

    def create(self, width, height, color):
        rgb = ImageColor.getrgb(color)
        return (Image.new("RGB", (int(width), int(height)), rgb),)


def list_uploads() -> list:
    """Filenames in UPLOAD_DIR (sorted, dotfiles like .gitkeep excluded) -
    the choices of LoadImage/LoadMask's `filename` combo and GET /api/uploads.
    Only verified images get in there (see POST /api/upload)."""
    if not UPLOAD_DIR.is_dir():
        return []
    return sorted(p.name for p in UPLOAD_DIR.iterdir() if p.is_file() and not p.name.startswith("."))


def _upload_path(filename: str) -> Path:
    path = safe_path(UPLOAD_DIR, filename)
    if not filename or not path.is_file():
        raise FileNotFoundError(f"'{filename}' is not in the uploads folder - upload it first")
    return path


def _upload_signature(filename: str):
    """IS_CHANGED helper: re-run when the file is replaced; always re-run
    (NaN) when it can't be read, so the node reports the real error."""
    try:
        stat = _upload_path(filename).stat()
    except (OSError, PathTraversalError):
        return float("nan")
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _upload_input_types():
    # A combo of the current uploads, rebuilt on every call (INPUT_TYPES()
    # is re-read by /api/nodes and by validate_prompt), plus the
    # `image_upload` flag that makes the frontend add an "Upload..." button.
    return (list_uploads(), {"image_upload": True})


def _validate_upload(filename=None, **_):
    if filename is None:
        return True
    try:
        _upload_path(filename)
    except (FileNotFoundError, PathTraversalError) as exc:
        return str(exc)
    return True


# MASK socket values are single-channel PIL images ("L" mode): white (255)
# marks the region to repaint, black keeps the original pixels.


@register_node("Load Image")
class LoadImage(NodeBase):
    """Loads an uploaded image. `mask` comes from the image's alpha channel
    the way ComfyUI does it: transparent pixels become white (repaint) mask
    pixels, and an image without alpha gives an all-black mask. EXIF
    orientation is applied, so phone photos come in the right way up."""

    CATEGORY = "loaders"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load"
    DESCRIPTION = "An uploaded image, plus a mask from its transparency."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filename": _upload_input_types()}}

    @classmethod
    def IS_CHANGED(cls, filename="", **_):
        return _upload_signature(filename)

    @classmethod
    def VALIDATE_INPUTS(cls, **inputs):
        return _validate_upload(**inputs)

    def load(self, filename):
        with Image.open(_upload_path(filename)) as opened:
            img = ImageOps.exif_transpose(opened)
            rgb = img.convert("RGB")
            if "A" in img.getbands() or "transparency" in img.info:
                mask = ImageOps.invert(img.convert("RGBA").getchannel("A"))
            else:
                mask = Image.new("L", img.size, 0)
        return (rgb, mask)


@register_node("Load Mask")
class LoadMask(NodeBase):
    """Loads an uploaded image as a MASK: its luminance (white = repaint),
    or its alpha channel (transparent = repaint, like Load Image's mask)."""

    CATEGORY = "loaders"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": _upload_input_types(),
                "channel": (["luminance", "alpha"], {"default": "luminance"}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, filename="", **_):
        return _upload_signature(filename)

    @classmethod
    def VALIDATE_INPUTS(cls, **inputs):
        return _validate_upload(**inputs)

    def load(self, filename, channel="luminance", invert=False):
        with Image.open(_upload_path(filename)) as opened:
            img = ImageOps.exif_transpose(opened)
            if channel == "alpha":
                mask = ImageOps.invert(img.convert("RGBA").getchannel("A"))
            else:
                mask = img.convert("L")
        if invert:
            mask = ImageOps.invert(mask)
        return (mask,)


# Hidden inputs (never shown as widgets/sockets) the engine injects into the
# output nodes, the same way ComfyUI embeds the graph into saved PNGs.
_METADATA_HIDDEN_INPUTS = {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}

# Only these extra_pnginfo keys are embedded in saved PNGs (report item R6):
# a client can't bloat every output with arbitrary metadata.
EMBEDDED_PNGINFO_KEYS = ("workflow",)

# Keys samplers put into an output image's .info (PIL carries .info through
# most operations) so SaveImage can embed the image's *own* seed (B8).
SEED_INFO_KEY = "pixelflow_seed"
NODE_INFO_KEY = "pixelflow_node"
MODEL_INFO_KEY = "pixelflow_model"


def _prompt_for_image(prompt, image):
    """The prompt embedded into one saved image: when the image came out of
    a sampler batch, that sampler's seed is the image's own seed and its
    batch_size 1, so re-queueing the embedded prompt reproduces just it."""
    seed = image.info.get(SEED_INFO_KEY) if hasattr(image, "info") else None
    node_id = image.info.get(NODE_INFO_KEY) if hasattr(image, "info") else None
    if not prompt or seed is None or node_id not in prompt:
        return prompt
    inputs = prompt[node_id].get("inputs", {})
    if inputs.get("seed") == seed and inputs.get("batch_size", 1) == 1:
        return prompt
    per_image = copy.deepcopy(prompt)
    per_image[node_id]["inputs"]["seed"] = seed
    if "batch_size" in per_image[node_id]["inputs"]:
        per_image[node_id]["inputs"]["batch_size"] = 1
    return per_image


def _png_metadata(prompt, extra_pnginfo):
    """PNG tEXt chunks for a saved image: `prompt` is the executed graph
    (with any random seed replaced by the seed actually used, so it can be
    re-queued to reproduce the image), plus the known `extra_pnginfo`
    entries (the frontend sends its litegraph `workflow` there)."""
    extras = {k: v for k, v in (extra_pnginfo or {}).items() if k in EMBEDDED_PNGINFO_KEYS}
    if not prompt and not extras:
        return None
    info = PngInfo()
    if prompt:
        info.add_text("prompt", json.dumps(prompt))
    for key, value in extras.items():
        info.add_text(str(key), json.dumps(value))
    return info


def _strip_internal_info(image):
    for key in (SEED_INFO_KEY, NODE_INFO_KEY, MODEL_INFO_KEY):
        image.info.pop(key, None)


def _save_temp(image, prefix, folder, pnginfo=None):
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{sanitize_filename(prefix)}_{uuid.uuid4().hex[:8]}.png"
    path = safe_path(folder, name)
    image = image.copy()
    _strip_internal_info(image)
    # Throwaway previews: fast compression.
    image.save(path, pnginfo=pnginfo, compress_level=1)
    return path.name


@register_node("Preview Image")
class PreviewImage(NodeBase):
    """Shows images in the gallery without keeping them: they go to the
    temp/ folder, which is emptied whenever the server starts."""

    CATEGORY = "output"
    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}, "hidden": dict(_METADATA_HIDDEN_INPUTS)}

    def preview(self, image, prompt=None, extra_pnginfo=None):
        names = [
            _save_temp(img, "preview", TEMP_DIR, _png_metadata(_prompt_for_image(prompt, img), extra_pnginfo))
            for img in as_image_list(image)
        ]
        return {"images": names, "type": "temp"}


# ---- Save Image naming (report item B17) ----

_save_lock = threading.Lock()


def expand_prefix_tokens(prefix: str, image=None, now=None) -> str:
    """%date% (YYYY-MM-DD), %time% (HHMMSS), %seed% and %model% (from the
    sampler that made the image; empty when unknown)."""
    now = now or datetime.datetime.now()
    info = getattr(image, "info", {}) or {}
    model = str(info.get(MODEL_INFO_KEY, "")).replace("/", "_").replace("\\", "_")
    values = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H%M%S"),
        "seed": str(info.get(SEED_INFO_KEY, "")),
        "model": model,
    }
    return re.sub(r"%(date|time|seed|model)%", lambda m: values[m.group(1)], prefix or "")


def split_prefix(prefix: str):
    """'portraits/2026/hero' -> (['portraits', '2026'], 'hero'). Each folder
    segment is sanitised like a filename; '.', '..' and empty segments are
    dropped, so the result always stays inside the output folder."""
    segments = [s for s in (prefix or "").split("/")]
    stem = sanitize_filename(segments[-1] if segments else "", default="")
    folders = []
    for segment in segments[:-1]:
        if segment.strip() in ("", ".", ".."):
            continue
        cleaned = sanitize_filename(segment, default="")
        if cleaned:
            folders.append(cleaned)
    if not stem:
        stem = "image"
    return folders, stem


def next_counter(folder: Path, stem: str) -> int:
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)_\.png$", re.IGNORECASE)
    highest = 0
    if folder.is_dir():
        for path in folder.iterdir():
            match = pattern.match(path.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def save_numbered(image, prefix: str, base: Path, pnginfo=None) -> str:
    """Save as <folders>/<stem>_00001_.png (ComfyUI's naming: sorts in
    creation order), returning the path relative to `base` with '/'."""
    folders, stem = split_prefix(expand_prefix_tokens(prefix, image))
    folder = safe_path(base, "/".join(folders)) if folders else base.resolve()
    image = image.copy()
    _strip_internal_info(image)
    with _save_lock:
        folder.mkdir(parents=True, exist_ok=True)
        counter = next_counter(folder, stem)
        while True:
            path = safe_path(folder, f"{stem}_{counter:05d}_.png")
            try:
                handle = open(path, "xb")
            except FileExistsError:
                counter += 1
                continue
            break
    with handle:
        image.save(handle, format="PNG", pnginfo=pnginfo)
    return path.relative_to(base.resolve()).as_posix()


@register_node("Save Image")
class SaveImage(NodeBase):
    """Saves to the outputs folder as <prefix>_00001_.png, ... A prefix may
    contain subfolders ('portraits/hero') and the tokens %date%, %time%,
    %seed% and %model%."""

    CATEGORY = "output"
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": (
                    "STRING",
                    {"default": "PixelFlow", "tooltip": "Subfolders with '/'; tokens %date% %time% %seed% %model%"},
                ),
            },
            "hidden": dict(_METADATA_HIDDEN_INPUTS),
        }

    def save(self, image, filename_prefix, prompt=None, extra_pnginfo=None):
        names = [
            save_numbered(img, filename_prefix, OUTPUT_DIR, _png_metadata(_prompt_for_image(prompt, img), extra_pnginfo))
            for img in as_image_list(image)
        ]
        return {"images": names, "type": "output"}
