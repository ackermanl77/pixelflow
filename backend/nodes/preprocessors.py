"""ControlNet preprocessors beyond Canny Edges: Depth Map (a Depth Anything V2
model through transformers), and three classical ones in NumPy -- Soft Edges,
Lineart (XDoG) and Scribble. Each turns an IMAGE into the control image a
matching ControlNet expects: white-on-black edges or lines, or a depth map
with near = bright.

The classical edge maps approximate the learned HED/lineart annotators that
some ControlNets were trained on; they are close enough to guide those models,
but a learned annotator gives cleaner lines on busy photos.
"""
from __future__ import annotations

import threading
from typing import Dict, Tuple

from PIL import Image, ImageFilter

from .base import NodeBase, map_images, register_node

#: Depth models offered in the node's list. Depth Anything V2 Small is
#: Apache-2.0; the Base and Large checkpoints are licensed non-commercially, so
#: they're not listed (any transformers depth model id can still be typed).
DEPTH_MODELS = ["depth-anything/Depth-Anything-V2-Small-hf"]

#: How far below its surroundings (0-1 grey levels) a pixel must dip before
#: Lineart draws a line there: keeps sensor noise and fine texture out.
LINEART_NOISE_FLOOR = 0.02

_depth_cache: Dict[Tuple[str, str], tuple] = {}
_depth_lock = threading.Lock()


def _gray(image: Image.Image):
    import numpy as np

    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _to_rgb(values) -> Image.Image:
    """A 0..1 float array -> an RGB image (the control-image format)."""
    import numpy as np

    return Image.fromarray(np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)).convert("RGB")


def _blur(values, sigma: float):
    import numpy as np

    if sigma <= 0:
        return values
    # Separable Gaussian in float (Pillow only blurs 8-bit images, and XDoG
    # needs the small differences 8 bits would round away).
    radius = max(1, int(3 * sigma + 0.5))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    height, width = values.shape
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    rows = sum(weight * padded[:, i:i + width] for i, weight in enumerate(kernel))
    padded = np.pad(rows, ((radius, radius), (0, 0)), mode="edge")
    return sum(weight * padded[i:i + height] for i, weight in enumerate(kernel)).astype(np.float32)


def _sobel_magnitude(values):
    import numpy as np

    padded = np.pad(values, 1, mode="edge")
    gx = (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]) - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    gy = (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]) - (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
    return np.hypot(gx, gy)


def soft_edges(image: Image.Image, detail: float = 1.0) -> Image.Image:
    """Soft (HED-like) edges: the gradient strength of the smoothed image,
    normalized to its 99th percentile and eased, so strong contours are white
    and texture fades to grey. Higher `detail` keeps finer edges."""
    import numpy as np

    gray = _blur(_gray(image), max(0.5, 2.0 / max(detail, 0.1)))
    magnitude = _sobel_magnitude(gray)
    scale = np.percentile(magnitude, 99) or 1.0
    return _to_rgb(np.sqrt(np.clip(magnitude / scale, 0.0, 1.0)))


def lineart(image: Image.Image, sigma: float = 1.0, sharpness: float = 60.0) -> Image.Image:
    """Line art by XDoG (eXtended Difference of Gaussians): thin white lines on
    black where the image has edges, from fine (`sigma` small) to bold."""
    import numpy as np

    gray = _gray(image)
    small = _blur(gray, sigma)
    large = _blur(gray, sigma * 1.6)
    dog = small - 0.98 * large  # a sharpened image: dips below zero along the dark side of edges
    # A line only where the dip is deeper than noise (LINEART_NOISE_FLOOR), ramping up with `sharpness`.
    lines = np.tanh(sharpness * np.maximum(0.0, -dog - LINEART_NOISE_FLOOR))
    return _to_rgb(lines)


def scribble(image: Image.Image, threshold: float = 0.35, thickness: int = 3) -> Image.Image:
    """Scribble: the soft edges above `threshold`, thickened to `thickness`
    pixels and made pure white on black, like a quick hand sketch."""
    import numpy as np

    edges = np.asarray(soft_edges(image, detail=0.6).convert("L"), dtype=np.float32) / 255.0
    mask = Image.fromarray(((edges >= threshold) * 255).astype(np.uint8))
    size = max(1, int(thickness)) | 1  # MaxFilter needs an odd size
    if size > 1:
        mask = mask.filter(ImageFilter.MaxFilter(size))
    return mask.convert("RGB")


def load_depth_model(repo_id: str, device: str):
    """(image processor, model) for a transformers depth-estimation model, on
    `device` (cached; the model is downloaded once into the Hugging Face cache)."""
    key = (repo_id, device)
    with _depth_lock:
        if key in _depth_cache:
            return _depth_cache[key]
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    try:
        processor = AutoImageProcessor.from_pretrained(repo_id)
        model = AutoModelForDepthEstimation.from_pretrained(repo_id).to(device).eval()
    except OSError as error:
        raise RuntimeError(
            f"Depth model '{repo_id}' couldn't be loaded: it's downloaded from the Hugging Face Hub on first use, "
            f"so connect to the internet once ({error})"
        ) from error
    entry = (processor, model)
    with _depth_lock:
        _depth_cache.clear()  # one depth model in memory at a time
        _depth_cache[key] = entry
    return entry


def depth_map(image: Image.Image, processor, model, device: str) -> Image.Image:
    """Relative depth of `image` as a grey image of the same size, near =
    white (what depth ControlNets expect)."""
    from . import runtime

    torch = runtime.torch()
    rgb = image.convert("RGB")
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.inference_mode():
        predicted = model(**inputs).predicted_depth  # (1, h, w), larger = nearer
        resized = torch.nn.functional.interpolate(
            predicted[:, None].float(), size=(rgb.height, rgb.width), mode="bicubic", align_corners=False
        )[0, 0]
    low, high = resized.min(), resized.max()
    values = ((resized - low) / (high - low)).cpu().numpy() if float(high - low) > 1e-6 else resized.cpu().numpy() * 0
    return _to_rgb(values)


@register_node("Depth Map")
class DepthMap(NodeBase):
    """Estimates depth with a Depth Anything V2 model (transformers), for a
    depth ControlNet: near is white, far is black. The model (~100 MB) is
    downloaded from the Hugging Face Hub on first use and runs on the GPU when
    there is one."""

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    DESCRIPTION = "Depth map, near = white (for a depth ControlNet)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (DEPTH_MODELS, {"allow_custom": True, "tooltip": "A transformers depth-estimation model id"}),
            }
        }

    def run(self, image, model=DEPTH_MODELS[0]):
        from . import runtime

        torch = runtime.torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor, depth_model = load_depth_model(model or DEPTH_MODELS[0], device)
        return (map_images(image, lambda img: depth_map(img, processor, depth_model, device)),)


@register_node("Soft Edges")
class SoftEdges(NodeBase):
    """Soft, HED-like edges: strong contours white, texture faint grey (for a
    SoftEdge/HED ControlNet). Classical (gradient strength), no model."""

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    DESCRIPTION = "Soft edge map (for a SoftEdge / HED ControlNet)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "detail": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05})}}

    def run(self, image, detail=1.0):
        return (map_images(image, lambda img: soft_edges(img, float(detail))),)


@register_node("Lineart")
class Lineart(NodeBase):
    """Line art by XDoG: white lines on black (for a lineart ControlNet).
    `sigma` sets the line scale: small for fine detail, larger for bold strokes."""

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    DESCRIPTION = "Line art, white on black (for a lineart ControlNet)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 5.0, "step": 0.1}),
                "sharpness": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 200.0, "step": 1.0}),
            }
        }

    def run(self, image, sigma=1.0, sharpness=60.0):
        return (map_images(image, lambda img: lineart(img, float(sigma), float(sharpness))),)


@register_node("Scribble")
class Scribble(NodeBase):
    """Thick, pure white strokes on black from the image's main contours (for
    a scribble ControlNet)."""

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    DESCRIPTION = "Scribble strokes, white on black (for a scribble ControlNet)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05}),
                "thickness": ("INT", {"default": 3, "min": 1, "max": 15}),
            }
        }

    def run(self, image, threshold=0.35, thickness=3):
        return (map_images(image, lambda img: scribble(img, float(threshold), int(thickness))),)
