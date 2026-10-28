"""IP-Adapter (B20): an image used as a prompt, through diffusers' IP-Adapter support.

"Apply IP-Adapter" adds a reference image, an adapter (models/ipadapter) and the CLIP vision
image encoder it was trained with (models/clip_vision) to a MODEL, the way Load LoRA adds a LoRA:
nothing changes in the shared, cached pipeline until a sampler runs. The sampler then makes
exactly this MODEL's adapter active (activate()): loaded once onto the pipeline's UNet and kept
while the same adapter is used, unloaded for a run without one (a UNet with an IP-Adapter loaded
needs an image on every call).

The image encoder is read from ComfyUI's clip_vision file, a transformers CLIP checkpoint (the
full CLIPModel or its vision half); its configuration comes from the tensor shapes, so nothing is
downloaded. Supported: SD 1.5 and SDXL adapters, including the "plus" ones (diffusers chooses the
image projection from the adapter's keys).
"""

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from . import models, runtime
from .base import NodeBase, first_image, register_node

logger = logging.getLogger(__name__)

_lock = threading.Lock()
# Attention heads by hidden size: OpenAI ViT-L/14, LAION ViT-H/14 and ViT-bigG/14 all use 16.
_HEADS = {1024: 16, 1280: 16, 1664: 16}


def _file(kind: str, name: str) -> Path:
    path = models.local_model_path(kind, name)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"'{name}' is not a file in models/{kind}/")
    return path


def vision_config_from_state(state) -> Dict:
    """CLIPVisionConfig arguments read off a CLIP vision state dict's shapes."""
    try:
        hidden = state["vision_model.embeddings.class_embedding"].shape[0]
        patch = state["vision_model.embeddings.patch_embedding.weight"].shape[-1]
        positions = state["vision_model.embeddings.position_embedding.weight"].shape[0]
        intermediate = state["vision_model.encoder.layers.0.mlp.fc1.weight"].shape[0]
    except KeyError as missing:
        raise ValueError(f"not a CLIP vision checkpoint in transformers' layout (no {missing})") from None
    if hidden not in _HEADS:
        raise ValueError(f"unsupported CLIP vision width {hidden} (ViT-L/14, ViT-H/14 and ViT-bigG/14 are supported)")
    layers = len({key.split(".")[3] for key in state if key.startswith("vision_model.encoder.layers.")})
    projection = state["visual_projection.weight"].shape[0] if "visual_projection.weight" in state else hidden
    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_attention_heads": _HEADS[hidden],
        "patch_size": patch,
        "image_size": int(round((positions - 1) ** 0.5)) * patch,
        "projection_dim": projection,
        # OpenAI's ViT-L/14 uses quick GELU; the LAION-trained ViT-H and bigG use GELU.
        "hidden_act": "quick_gelu" if hidden == 1024 else "gelu",
        "layer_norm_eps": 1e-5,
    }


@lru_cache(maxsize=2)
def load_image_encoder(path_text: str, stamp: str, dtype_name: str):
    """CLIPVisionModelWithProjection from a clip_vision file (`stamp` = its mtime/size, so a
    replaced file is read again). Built on the meta device and given the file's tensors."""
    import torch
    import transformers
    from safetensors.torch import load_file

    # position_ids: older checkpoints store it; transformers now computes it (rebuilt below).
    state = {key: value for key, value in load_file(path_text).items()
             if (key.startswith("vision_model.") or key == "visual_projection.weight") and not key.endswith("position_ids")}
    config = transformers.CLIPVisionConfig(**vision_config_from_state(state))
    with torch.device("meta"):
        encoder = transformers.CLIPVisionModelWithProjection(config)
    dtype = models.torch_dtype(dtype_name)
    state = {key: value.to(dtype) for key, value in state.items()}
    if "visual_projection.weight" not in state:
        state["visual_projection.weight"] = torch.eye(config.hidden_size, dtype=dtype)
    missing, unexpected = encoder.load_state_dict(state, strict=False, assign=True)
    missing = [key for key in missing if not key.endswith("position_ids")]
    if missing or unexpected:
        raise ValueError(f"'{Path(path_text).name}' doesn't fit a CLIP vision model: missing {missing[:4]}, unexpected {unexpected[:4]}")
    embeddings = encoder.vision_model.embeddings
    if embeddings.position_ids.is_meta:
        embeddings.register_buffer("position_ids", torch.arange(embeddings.num_positions).expand((1, -1)), persistent=False)
    return encoder.eval()


def _unet_owner(pipe):
    """Where the loaded adapter is recorded: the UNet, shared by a pipeline and every pipeline
    derived from it with from_pipe() (img2img, inpaint, ControlNet)."""
    unet = getattr(pipe, "unet", None)
    if unet is None:
        raise RuntimeError(f"{type(pipe).__name__} has no UNet: IP-Adapter supports SD 1.5 and SDXL pipelines")
    return unet


def activate(pipe, adapter: Optional[Dict]) -> Dict:
    """Make exactly `adapter` (a MODEL's "ip_adapter", or None) active on `pipe` for one run, and
    return the call arguments it needs ({"ip_adapter_image": ...} or {})."""
    if adapter is None:
        unet = getattr(pipe, "unet", None)
        if unet is not None and getattr(unet, "_pixelflow_ip_adapter", None):
            with _lock:
                logger.info("unloading the IP-Adapter (this run has none)")
                pipe.unload_ip_adapter()
                unet._pixelflow_ip_adapter = None
        return {}
    if not hasattr(pipe, "load_ip_adapter"):
        raise RuntimeError(f"{type(pipe).__name__} does not support IP-Adapters")
    unet = _unet_owner(pipe)
    adapter_path = _file("ipadapter", adapter["ipadapter"])
    encoder_path = _file("clip_vision", adapter["clip_vision"])
    key = (str(adapter_path), adapter_path.stat().st_mtime_ns, str(encoder_path))
    stat = encoder_path.stat()
    dtype_name = "fp32" if unet.dtype == runtime.torch().float32 else ("bf16" if unet.dtype == runtime.torch().bfloat16 else "fp16")
    encoder = load_image_encoder(str(encoder_path), f"{stat.st_mtime_ns}:{stat.st_size}", dtype_name)
    with _lock:
        device = getattr(pipe, "_execution_device", None) or unet.device
        encoder.to(device)
        if getattr(pipe, "image_encoder", None) is not encoder:
            from transformers import CLIPImageProcessor

            pipe.register_modules(image_encoder=encoder, feature_extractor=CLIPImageProcessor())
        if getattr(unet, "_pixelflow_ip_adapter", None) != key:
            if getattr(unet, "_pixelflow_ip_adapter", None):
                pipe.unload_ip_adapter()
                pipe.register_modules(image_encoder=encoder, feature_extractor=CLIPImageProcessor())
            logger.info("loading IP-Adapter %s with image encoder %s", adapter_path.name, encoder_path.name)
            pipe.load_ip_adapter(str(adapter_path.parent), subfolder="", weight_name=adapter_path.name, image_encoder_folder=None)
            unet._pixelflow_ip_adapter = key
        pipe.set_ip_adapter_scale(float(adapter["weight"]))
    return {"ip_adapter_image": adapter["image"]}


@register_node("Apply IP-Adapter")
class ApplyIPAdapter(NodeBase):
    """Uses an image as a prompt: its look (style, subject, colours) steers what the sampler
    makes, at `weight`. Needs an adapter from models/ipadapter and the CLIP vision image encoder it
    was trained with from models/clip_vision (e.g. ip-adapter-plus_sdxl_vit-h with
    CLIP-ViT-H-14-laion2B). SD 1.5 and SDXL checkpoints."""

    CATEGORY = "conditioning"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    DESCRIPTION = "Uses an image as a prompt (IP-Adapter): adds it to the model for the sampler."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE", {"tooltip": "The reference image (the first of a batch)"}),
                "ipadapter": (models.list_models("ipadapter"), {"tooltip": "An IP-Adapter in models/ipadapter/, e.g. ip-adapter-plus_sdxl_vit-h.safetensors"}),
                "clip_vision": (models.list_models("clip_vision"), {"tooltip": "Its image encoder in models/clip_vision/ (ViT-H for the *_vit-h adapters)"}),
                "weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                                     "tooltip": "How strongly the image steers the result; 0.5-0.8 keeps the text prompt in charge"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, ipadapter="", clip_vision="", **_):
        stamps = []
        for kind, name in (("ipadapter", ipadapter), ("clip_vision", clip_vision)):
            path = models.local_model_path(kind, str(name or ""))
            if path is not None and path.is_file():
                stat = path.stat()
                stamps.append(f"{stat.st_mtime_ns}:{stat.st_size}")
        return "|".join(stamps)

    def apply(self, model, image, ipadapter, clip_vision, weight=0.7):
        _file("ipadapter", ipadapter)
        _file("clip_vision", clip_vision)
        adapter = {"ipadapter": ipadapter, "clip_vision": clip_vision, "image": first_image(image).convert("RGB"), "weight": float(weight)}
        return ({**model, "ip_adapter": adapter},)
