"""Remove Background (report item F28): the subject kept, the background made
transparent (or a MASK of it), like ComfyUI's Remove Background node.

The subject is every "thing" instance a COCO panoptic Mask2Former finds
(people, animals, vehicles, furniture...), as opposed to "stuff" (sky, wall,
grass, road). It is a native transformers architecture, so no remote code
runs; the model (~190 MB) is downloaded into the Hugging Face cache on first
use and runs on the GPU when there is one. ComfyUI's dedicated matting models
(BiRefNet, RMBG) need trust_remote_code and are left out.
"""

import threading
from typing import Callable, Optional

from PIL import Image, ImageFilter

from .base import NodeBase, as_image_list, register_node

BACKGROUND_MODELS = ["facebook/mask2former-swin-tiny-coco-panoptic"]
OUTPUT_MODES = ["transparent", "white", "black", "green"]
_FILLS = {"white": (255, 255, 255), "black": (0, 0, 0), "green": (0, 177, 64)}

_lock = threading.Lock()
_cache: dict = {}


def load_segmenter(repo_id: str, device: str):
    """(image processor, model) for a transformers panoptic segmentation model (one cached)."""
    key = (repo_id, device)
    with _lock:
        if key in _cache:
            return _cache[key]
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    try:
        processor = AutoImageProcessor.from_pretrained(repo_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(repo_id).to(device).eval()
    except OSError as error:
        raise RuntimeError(
            f"Background model '{repo_id}' couldn't be loaded: it's downloaded from the Hugging Face Hub on first use, "
            f"so connect to the internet once ({error})"
        ) from error
    with _lock:
        _cache.clear()
        _cache[key] = (processor, model)
    return processor, model


def subject_mask(image: Image.Image, processor, model, device: str, threshold: float = 0.5) -> Image.Image:
    """An "L" mask of `image` (white = subject): the union of the "thing" segments."""
    from . import runtime

    torch = runtime.torch()
    rgb = image.convert("RGB")
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_panoptic_segmentation(
        outputs, threshold=float(threshold), target_sizes=[(rgb.height, rgb.width)], label_ids_to_fuse=set()
    )[0]
    import numpy as np

    segmentation = result["segmentation"].cpu().numpy()
    things = [info["id"] for info in result["segments_info"] if int(info["label_id"]) < _COCO_THINGS]
    mask = np.isin(segmentation, things) if things else np.zeros(segmentation.shape, dtype=bool)
    return Image.fromarray((mask * 255).astype("uint8"), mode="L")


# COCO panoptic: label ids below 80 are the 80 "thing" classes, the rest "stuff".
_COCO_THINGS = 80


def cut_out(image: Image.Image, mask: Image.Image, mode: str = "transparent", feather: float = 1.0) -> Image.Image:
    """`image` with everything outside `mask` removed: transparent (RGBA) or a flat colour."""
    alpha = mask.convert("L").resize(image.size, Image.BILINEAR)
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(float(feather)))
    rgb = image.convert("RGB")
    if mode == "transparent":
        result = rgb.copy()
        result.putalpha(alpha)
        return result
    background = Image.new("RGB", image.size, _FILLS[mode])
    return Image.composite(rgb, background, alpha)


@register_node("Remove Background")
class RemoveBackground(NodeBase):
    """Keeps the subject (people, animals, objects) and removes the rest,
    with a MASK of the subject for inpainting or compositing."""

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "run"
    DESCRIPTION = "Cut the subject out: transparent or a flat background, plus its mask."

    # Tests replace this with a stand-in: (image) -> "L" mask.
    segment: Optional[Callable[[Image.Image], Image.Image]] = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "background": (OUTPUT_MODES, {"tooltip": "transparent keeps an alpha channel (save as PNG)"}),
                "feather": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.5,
                                      "tooltip": "Softens the cut-out's edge, in pixels"}),
                "model": (BACKGROUND_MODELS, {"allow_custom": True, "tooltip": "A transformers Mask2Former panoptic model id"}),
            }
        }

    def _segmenter(self, model: str) -> Callable[[Image.Image], Image.Image]:
        if RemoveBackground.segment is not None:
            return RemoveBackground.segment
        from . import runtime

        torch = runtime.torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor, segmenter = load_segmenter(model or BACKGROUND_MODELS[0], device)
        return lambda img: subject_mask(img, processor, segmenter, device)

    def run(self, image, background="transparent", feather=1.0, model=BACKGROUND_MODELS[0]):
        if background not in OUTPUT_MODES:
            raise ValueError(f"background must be one of {', '.join(OUTPUT_MODES)}")
        segment = self._segmenter(model)
        images, masks = [], []
        for img in as_image_list(image):
            mask = segment(img)
            images.append(cut_out(img, mask, background, float(feather)))
            masks.append(mask)
        if isinstance(image, (list, tuple)):
            return (images, masks)
        return (images[0], masks[0])
