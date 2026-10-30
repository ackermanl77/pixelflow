"""Split-file models (B26): a diffusion model, a text encoder and a VAE as separate single files,
the way ComfyUI's "Load Diffusion Model" / "Load CLIP" / "Load VAE" nodes take them (the
models/diffusion_models, text_encoders and vae folders, or a ComfyUI library's).

The diffusion model's state-dict keys say which family it is (diffusers'
infer_diffusers_model_type()); the family (SPLIT_FAMILIES) names the diffusers pipeline, the
Hugging Face repo whose small config/tokenizer/scheduler files describe it (fetched once, a few
hundred KB, never its weights), and how each component file is read:

- the transformer through diffusers' own single-file converter (from_single_file);
- the text encoder, a transformers model, built from the repo's config and filled from the file
  (ComfyUI's text encoder files keep the causal-LM's "model." prefix);
- the VAE through diffusers' single-file loader.

Supported: Z-Image Turbo (6B transformer, Qwen3-4B text encoder, the Flux VAE) and Flux (Flux
transformer, CLIP-L and T5-XXL text encoders, Flux VAE). The pipeline cache
(models.PipelineCache) holds the result under split_model_name(), like any checkpoint, and falls
back to model CPU offload when it doesn't fit the GPU on its own.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from .models import ARCH_FLUX, ARCH_ZIMAGE

logger = logging.getLogger(__name__)

SPLIT_PREFIX = "split:"
_SEPARATOR = "|"
# What a family's repo contributes: configs, the tokenizer and the scheduler -- no weights.
CONFIG_PATTERNS = [
    "model_index.json", "*/config.json", "*.json", "tokenizer/*", "tokenizer_2/*", "scheduler/*", "*/*.txt", "*/*.model"
]


@dataclass(frozen=True)
class SplitFamily:
    arch: str
    pipeline: str  # diffusers pipeline class
    transformer: str  # diffusers model class, loaded with from_single_file
    repo: str  # Hugging Face repo with the configs
    text_encoder: str  # transformers model class
    text_encoder_prefix: str  # key prefix ComfyUI's text encoder file adds
    vae: str  # diffusers VAE class, loaded with from_single_file
    text_encoder_2: Optional[str] = None
    text_encoder_2_prefix: str = ""


# infer_diffusers_model_type() of the diffusion model file -> its family.
SPLIT_FAMILIES: Dict[str, SplitFamily] = {
    "z-image-turbo": SplitFamily(
        arch=ARCH_ZIMAGE, pipeline="ZImagePipeline", transformer="ZImageTransformer2DModel",
        repo="Tongyi-MAI/Z-Image-Turbo", text_encoder="Qwen3Model", text_encoder_prefix="model.", vae="AutoencoderKL",
    ),
    "flux-dev": SplitFamily(
        arch=ARCH_FLUX, pipeline="FluxPipeline", transformer="FluxTransformer2DModel",
        repo="black-forest-labs/FLUX.1-schnell", text_encoder="CLIPTextModel", text_encoder_prefix="text_model.",
        text_encoder_2="T5EncoderModel", vae="AutoencoderKL",
    ),
    "flux-schnell": SplitFamily(
        arch=ARCH_FLUX, pipeline="FluxPipeline", transformer="FluxTransformer2DModel",
        repo="black-forest-labs/FLUX.1-schnell", text_encoder="CLIPTextModel", text_encoder_prefix="text_model.",
        text_encoder_2="T5EncoderModel", vae="AutoencoderKL",
    ),
}


def split_model_name(
    diffusion_model: str, text_encoder: str, vae: str, text_encoder_2: Optional[str] = None
) -> str:
    """The pipeline cache's name for one combination of component files."""
    parts = (diffusion_model, text_encoder, text_encoder_2, vae) if text_encoder_2 is not None else (diffusion_model, text_encoder, vae)
    return SPLIT_PREFIX + _SEPARATOR.join(parts)


def parse_split_name(name: str) -> Optional[Tuple[str, ...]]:
    if not str(name).startswith(SPLIT_PREFIX):
        return None
    parts = str(name)[len(SPLIT_PREFIX):].split(_SEPARATOR)
    if len(parts) not in (3, 4) or not all(parts):
        raise ValueError(f"'{name}' is not a split model (diffusion model|text encoder[|text encoder 2]|vae)")
    return tuple(parts)


def detect_family(path: Path) -> Tuple[str, SplitFamily]:
    from diffusers.loaders.single_file_utils import infer_diffusers_model_type

    from .models import _LazySafetensors

    model_type = infer_diffusers_model_type(_LazySafetensors(path))
    family = SPLIT_FAMILIES.get(model_type)
    if family is None:
        supported = ", ".join(sorted(SPLIT_FAMILIES))
        raise ValueError(f"'{path.name}' is a '{model_type}' model; split-file loading supports {supported}")
    return model_type, family


def ensure_configs(repo: str, on_progress: Optional[Callable[..., None]] = None) -> Path:
    """The repo's config, tokenizer and scheduler files (cached after the first time)."""
    from huggingface_hub import snapshot_download

    from .models import _hub_errors

    try:
        return Path(snapshot_download(repo, allow_patterns=CONFIG_PATTERNS, local_files_only=True))
    except _hub_errors():
        pass
    if on_progress is not None:
        on_progress(0, 1, message=f"Fetching {repo}'s configuration files")
    try:
        return Path(snapshot_download(repo, allow_patterns=CONFIG_PATTERNS))
    except _hub_errors() as error:
        raise RuntimeError(
            f"Loading a split model needs {repo}'s configuration files (a few hundred KB, fetched once) "
            f"and the Hugging Face Hub can't be reached: {error}"
        ) from error


def load_text_encoder(path: Path, family: SplitFamily, config_dir: Path, dtype, encoder_2: bool = False):
    """The transformers text encoder from a ComfyUI single file, without a second copy in memory:
    built on the meta device, then given the file's tensors (assign=True). Its rotary embedding
    holds computed buffers, not weights, so it is rebuilt on the CPU."""
    import torch
    import transformers
    from safetensors.torch import load_file

    model_class = family.text_encoder_2 if encoder_2 else family.text_encoder
    if model_class is None:
        raise ValueError(f"{family.pipeline} has no second text encoder")
    config_dir = config_dir / ("text_encoder_2" if encoder_2 else "text_encoder")
    config = transformers.AutoConfig.from_pretrained(config_dir)
    with torch.device("meta"):
        model = getattr(transformers, model_class)(config)
    prefix = family.text_encoder_2_prefix if encoder_2 else family.text_encoder_prefix
    state = {key[len(prefix):] if key.startswith(prefix) else key: value.to(dtype)
             for key, value in load_file(str(path)).items() if not key.startswith("lm_head.")}
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    rotary = {name for name, module in model.named_modules() if type(module).__name__.endswith("RotaryEmbedding")}
    missing = [key for key in missing if not any(key.startswith(name + ".") for name in rotary)]
    if missing or unexpected:
        raise ValueError(f"'{path.name}' doesn't fit {model_class}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    for name in rotary:
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, attribute)
        setattr(parent, attribute, type(old)(config=config, device="cpu"))
    return model.eval()


def component_files(name: str) -> Dict[str, Path]:
    """Resolve split_model_name(...)'s component files from their model-library folders."""
    from .models import local_model_path

    parts = parse_split_name(name)
    if parts is None:
        raise ValueError(f"'{name}' is not a split model")
    if len(parts) == 3:
        diffusion_model, text_encoder, vae = parts
        requested = (("diffusion_models", "diffusion_models", diffusion_model),
                     ("text_encoder", "text_encoders", text_encoder),
                     ("vae", "vae", vae))
    else:
        diffusion_model, text_encoder, text_encoder_2, vae = parts
        requested = (("diffusion_models", "diffusion_models", diffusion_model),
                     ("text_encoder", "text_encoders", text_encoder),
                     ("text_encoder_2", "text_encoders", text_encoder_2),
                     ("vae", "vae", vae))
    files = {}
    for key, kind, file_name in requested:
        path = local_model_path(kind, file_name)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"'{file_name}' is not a file in models/{kind}/")
        files[key] = path
    return files


def split_arch(name: str) -> str:
    """The architecture of split model `name`, from its diffusion model's keys (no weights read)."""
    return detect_family(component_files(name)["diffusion_models"])[1].arch


def load_split_pipeline(name: str, dtype, on_progress: Optional[Callable[..., None]] = None):
    """The pipeline for split_model_name(...) `name`, on the CPU; returns (pipeline, class name)."""
    import diffusers
    import transformers

    from .models import dtype_kwarg_name

    files = component_files(name)
    model_type, family = detect_family(files["diffusion_models"])
    if family.text_encoder_2 and "text_encoder_2" not in files:
        raise ValueError(f"{model_type} split models require both CLIP-L and T5 text encoder files")
    if not family.text_encoder_2 and "text_encoder_2" in files:
        raise ValueError(f"{model_type} split models accept only one text encoder")
    config_dir = ensure_configs(family.repo, on_progress)
    dtype_kwarg = {dtype_kwarg_name(): dtype}
    logger.info("loading split model %s: %s", model_type, " + ".join(p.name for p in files.values()))
    transformer = getattr(diffusers, family.transformer).from_single_file(
        str(files["diffusion_models"]), config=str(config_dir), subfolder="transformer", **dtype_kwarg)
    text_encoder = load_text_encoder(files["text_encoder"], family, config_dir, dtype)
    encoder_kwargs = {}
    if family.text_encoder_2:
        text_encoder_2 = load_text_encoder(files["text_encoder_2"], family, config_dir, dtype, encoder_2=True)
        tokenizer_2 = transformers.AutoTokenizer.from_pretrained(config_dir / "tokenizer_2")
        encoder_kwargs = {"text_encoder_2": text_encoder_2, "tokenizer_2": tokenizer_2}
    vae = getattr(diffusers, family.vae).from_single_file(str(files["vae"]), config=str(config_dir), subfolder="vae", **dtype_kwarg)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config_dir / "tokenizer")
    model_index = json.loads((config_dir / "model_index.json").read_text(encoding="utf-8"))
    scheduler = getattr(diffusers, model_index["scheduler"][1]).from_pretrained(str(config_dir), subfolder="scheduler")
    pipe = getattr(diffusers, family.pipeline)(
        transformer=transformer, text_encoder=text_encoder, tokenizer=tokenizer, vae=vae, scheduler=scheduler,
        **encoder_kwargs)
    return pipe, family.pipeline
