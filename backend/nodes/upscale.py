"""Model upscaling (report item B7, the optional part): ESRGAN-family and other
single-image super-resolution models (.pth / .safetensors in
models/upscale_models/, e.g. RealESRGAN_x4plus.pth or 4x-UltraSharp.pth),
loaded with the `spandrel` package, which recognises the architecture from the
weights. Upscale Image (model) runs the model in overlapping tiles so a big
image fits in VRAM.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import List

from PIL import Image

from .. import model_roots, paths
from . import runtime
from .base import NodeBase, map_images, register_node

UPSCALE_EXTENSIONS = (".pth", ".pt", ".safetensors")
TILE = 512
OVERLAP = 32

_cache: dict = {}
_cache_lock = threading.Lock()


def upscale_dir() -> Path:
    return paths.UPSCALE_MODEL_DIR


def list_upscale_models() -> List[str]:
    """Upscalers in models/upscale_models/ and the ComfyUI library's folders (B25)."""
    return model_roots.merge_names(_list_folder(folder) for folder in [upscale_dir(), *model_roots.extra_roots("upscale_models")])


def _list_folder(folder: Path) -> List[str]:
    if not folder.is_dir():
        return []
    return sorted(
        (p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in UPSCALE_EXTENSIONS and not p.name.startswith(".")),
        key=str.lower,
    )


def _find(name: str) -> Path | None:
    for folder in [upscale_dir(), *model_roots.extra_roots("upscale_models")]:
        try:
            path = paths.safe_path(folder, name)
        except paths.PathTraversalError:
            return None
        if path.is_file():
            return path
    return None


def load_upscale_model(name: str):
    """A spandrel ImageModelDescriptor for models/upscale_models/<name> (one cached)."""
    path = _find(name)
    if path is None:
        raise FileNotFoundError(f"upscale model '{name}' is not in models/upscale_models/")
    key = (str(path), path.stat().st_mtime_ns)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    try:
        import spandrel
    except ImportError as error:
        raise RuntimeError("model upscaling needs the 'spandrel' package (pip install -r requirements.txt)") from error
    descriptor = spandrel.ModelLoader(device="cpu").load_from_file(str(path))
    if not isinstance(descriptor, spandrel.ImageModelDescriptor):
        raise ValueError(f"'{name}' is a {descriptor.architecture.name} model that needs a mask, not an upscaler")
    descriptor.model.eval()
    with _cache_lock:
        _cache.clear()
        _cache[key] = descriptor
    return descriptor


def upscale_image(descriptor, image: Image.Image, tile: int = TILE, overlap: int = OVERLAP) -> Image.Image:
    """Run `descriptor` over `image` in tiles of `tile` px overlapping by
    `overlap`; each tile's inner part is kept, so seams fall inside overlaps."""
    import numpy as np

    torch = runtime.torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" and descriptor.supports_half else torch.float32
    descriptor.to(device)
    model = descriptor.model.to(dtype=dtype)
    scale = int(descriptor.scale)
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = source.shape[:2]
    out = np.zeros((height * scale, width * scale, 3), dtype=np.float32)
    step = max(1, tile - 2 * overlap)
    with torch.inference_mode():
        for top in range(0, height, step):
            for left in range(0, width, step):
                y0, x0 = max(0, top - overlap), max(0, left - overlap)
                y1, x1 = min(height, top + step + overlap), min(width, left + step + overlap)
                patch = torch.from_numpy(source[y0:y1, x0:x1]).permute(2, 0, 1)[None].to(device=device, dtype=dtype)
                result = model(patch).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                # Keep this tile's own region (its step-sized square), dropping the overlap margins.
                inner_y1, inner_x1 = min(height, top + step), min(width, left + step)
                ry, rx = (top - y0) * scale, (left - x0) * scale
                out[top * scale:inner_y1 * scale, left * scale:inner_x1 * scale] = result[
                    ry:ry + (inner_y1 - top) * scale, rx:rx + (inner_x1 - left) * scale
                ]
    return Image.fromarray(np.round(out * 255.0).astype(np.uint8))


@register_node("Load Upscale Model")
class UpscaleModelLoader(NodeBase):
    """An upscaling model from models/upscale_models/ (ESRGAN, Real-ESRGAN,
    SwinIR, HAT, ... -- anything spandrel recognises), as an UPSCALE_MODEL."""

    CATEGORY = "loaders"
    RETURN_TYPES = ("UPSCALE_MODEL",)
    RETURN_NAMES = ("upscale_model",)
    FUNCTION = "load"
    DESCRIPTION = "Loads a super-resolution model (models/upscale_models/)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (list_upscale_models() or [""], {"tooltip": "A .pth/.safetensors file in models/upscale_models/"})}}

    def load(self, model_name):
        if not model_name:
            raise ValueError("put an upscaling model (.pth or .safetensors) in models/upscale_models/ first")
        return (load_upscale_model(model_name),)


@register_node("Upscale Image (model)")
class ImageUpscaleWithModel(NodeBase):
    """Enlarges an IMAGE with an UPSCALE_MODEL (2x or 4x, as trained), tiled
    so large images fit in GPU memory. Add Resize Image after it for another
    size; or re-sample the result at low denoise for a hires fix."""

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    DESCRIPTION = "Upscales an image with a super-resolution model."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"upscale_model": ("UPSCALE_MODEL",), "image": ("IMAGE",)}}

    def run(self, upscale_model, image):
        return (map_images(image, lambda img: upscale_image(upscale_model, img)),)
