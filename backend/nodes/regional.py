"""Regional prompting (F19): a prompt that only applies inside a mask or a box.

"Regional Prompt" adds a region -- its prompt text plus a mask (a MASK input) or a box (fractions of
the image) -- to a CONDITIONING; chain several for several regions. The sampler then runs the UNet
once more per region at every step, with the region's prompt in place of the main one, and blends
the noise predictions by the regions' masks in latent space: the main prompt where no region
covers, each region's prompt inside its own (a MultiDiffusion-style composition, as ComfyUI's
Conditioning (Set Area / Set Mask) does). Only the positive (conditional) half of a CFG batch takes
part -- the negative prompt is shared -- so a region's prompt may tokenize to any length.

SD 1.x/2.x and SDXL (UNet) models, including their img2img, inpaint and ControlNet variants (they
share the UNet); a transformer model (SD3, Flux, Z-Image) is refused with a clear message.
"""

import contextlib
import logging
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import runtime
from .base import NodeBase, first_image, register_node

logger = logging.getLogger(__name__)

MAX_REGIONS = 8


def region_mask(region: Dict, width: int, height: int) -> np.ndarray:
    """A region's weight map (height, width) in 0..1: its MASK resized, or its box drawn with a
    feathered edge (feather: a fraction of the image's shorter side)."""
    if region.get("mask") is not None:
        mask = region["mask"].convert("L").resize((width, height), Image.BILINEAR)
    else:
        x, y, w, h = region["box"]
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).rectangle(
            [round(x * width), round(y * height), round((x + w) * width) - 1, round((y + h) * height) - 1], fill=255)
        radius = float(region.get("feather", 0.0)) * min(width, height)
        if radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius / 2))
    return np.asarray(mask, dtype=np.float32) / 255.0 * float(region.get("strength", 1.0))


def blend_weights(masks: List[np.ndarray]) -> List[np.ndarray]:
    """The regions' weights where they overlap: scaled down so they never add up to more than 1
    (the main prompt keeps 1 - their sum)."""
    total = np.sum(masks, axis=0)
    scale = np.where(total > 1.0, 1.0 / np.maximum(total, 1e-6), 1.0)
    return [mask * scale for mask in masks]


def _split(value, half: int, batch: int):
    """`value`'s conditional half: a tensor (or a list/tuple of them) batched like the UNet input."""
    if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == batch:
        return value[half:]
    if isinstance(value, (list, tuple)):
        return type(value)(_split(v, half, batch) for v in value)
    return value


@contextlib.contextmanager
def regional_unet(pipe, regions: List[Dict], cfg: bool):
    """For the duration of one pipeline call: the pipeline's UNet blends each region's noise
    prediction in by its mask. regions: [{"embeds": prompt_embeds [1, seq, dim], "pooled":
    pooled_prompt_embeds [1, dim] or None, "mask": (h, w) weights at image size}]. cfg: the
    pipeline runs classifier-free guidance (its UNet batch is [negative; positive])."""
    if not regions:
        yield
        return
    torch = runtime.torch()
    import torch.nn.functional as F

    unet = getattr(pipe, "unet", None)
    if unet is None:
        raise RuntimeError(f"Regional prompts need an SD 1.x/2.x or SDXL model; {type(pipe).__name__} has no UNet")
    original = unet.forward
    weights = blend_weights([region["mask"] for region in regions])
    latent_weights: Dict[tuple, list] = {}

    def forward(sample, timestep, encoder_hidden_states, *args, added_cond_kwargs=None, return_dict=True, **kwargs):
        out = original(sample, timestep, encoder_hidden_states, *args, added_cond_kwargs=added_cond_kwargs,
                       return_dict=False, **kwargs)[0]
        batch = sample.shape[0]
        half = batch // 2 if cfg else 0
        rows = batch - half
        key = tuple(sample.shape[-2:])
        if key not in latent_weights:
            latent_weights[key] = [
                F.interpolate(torch.from_numpy(w)[None, None], size=key, mode="bilinear", align_corners=False)
                .to(device=out.device, dtype=out.dtype) for w in weights
            ]
        cond_sample = sample[half:]
        cond_args = [_split(a, half, batch) for a in args]
        cond_kwargs = {name: _split(value, half, batch) for name, value in kwargs.items()}
        blended = out[half:].clone()
        for region, weight in zip(regions, latent_weights[key]):
            embeds = region["embeds"].to(device=out.device, dtype=encoder_hidden_states.dtype).expand(rows, -1, -1)
            added = None
            if added_cond_kwargs is not None:
                added = {name: _split(value, half, batch) for name, value in added_cond_kwargs.items()}
                if region.get("pooled") is not None and "text_embeds" in added:
                    added["text_embeds"] = region["pooled"].to(device=out.device, dtype=added["text_embeds"].dtype).expand(rows, -1)
            region_out = original(cond_sample, timestep, embeds, *cond_args, added_cond_kwargs=added,
                                  return_dict=False, **cond_kwargs)[0]
            blended = blended + weight * (region_out - out[half:])  # = base * (1 - w) + region * w
        out = torch.cat([out[:half], blended]) if half else blended
        if return_dict:
            from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

            return UNet2DConditionOutput(sample=out)
        return (out,)

    unet.forward = forward
    try:
        yield
    finally:
        del unet.forward  # back to the class's own forward


@register_node("Regional Prompt")
class RegionalPrompt(NodeBase):
    """A prompt for one part of the image (F19): `text` applies inside the `mask` (white = the
    region) or, without a mask, inside the box x/y/width/height (fractions of the image), with a
    soft edge of `feather`. The incoming conditioning stays the prompt for everywhere else. Chain
    several for several regions; overlapping ones share their overlap. SD 1.x/2.x and SDXL."""

    CATEGORY = "conditioning"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    DESCRIPTION = "Adds a prompt that only applies inside a mask or a box (regional prompting)."

    @classmethod
    def INPUT_TYPES(cls):
        fraction = {"min": 0.0, "max": 1.0, "step": 0.01}
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": "The main prompt (applies outside the region)"}),
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "The region's prompt"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Where the region is (white); without it, the box below"}),
                "x": ("FLOAT", {"default": 0.0, **fraction}),
                "y": ("FLOAT", {"default": 0.0, **fraction}),
                "width": ("FLOAT", {"default": 0.5, **fraction}),
                "height": ("FLOAT", {"default": 1.0, **fraction}),
                "feather": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                                      "tooltip": "Soft edge of the box, as a fraction of the image's shorter side"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "1 = the region's prompt alone inside it; lower mixes in the main prompt"}),
            },
        }

    def apply(self, conditioning, text, mask=None, x=0.0, y=0.0, width=0.5, height=1.0, feather=0.05, strength=1.0):
        regions = list(conditioning.get("regions", []))
        if len(regions) >= MAX_REGIONS:
            raise ValueError(f"at most {MAX_REGIONS} regional prompts per conditioning")
        region: Dict = {"text": str(text), "strength": float(strength), "feather": float(feather)}
        if mask is not None:
            region["mask"] = first_image(mask).convert("L")
        else:
            x, y = min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)
            w, h = min(max(float(width), 0.0), 1.0 - x), min(max(float(height), 0.0), 1.0 - y)
            if w <= 0 or h <= 0:
                raise ValueError("the region's box is empty: give it a width and a height")
            region["box"] = (x, y, w, h)
        return ({**conditioning, "regions": regions + [region]},)


def region_conditioning(conditioning) -> Optional[List[Dict]]:
    """The regions a CONDITIONING carries, or None."""
    regions = (conditioning or {}).get("regions")
    return list(regions) if regions else None
