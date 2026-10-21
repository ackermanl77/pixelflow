"""Textual-inversion embeddings in prompts (report item B22).

`embedding:name` in a prompt (ComfyUI's syntax; the extension is optional)
loads models/embeddings/name.safetensors (or .pt/.bin, subfolders as
`embedding:styles/name`) into the checkpoint's tokenizer and text encoder
as the token `<name>`, the first time it's used, and puts that token in the
text. SD 1.x/2.x files are the usual single-tensor (diffusers / A1111
`emb_params`) or `string_to_param` (A1111 .pt) formats; SDXL files carry
`clip_l` and `clip_g` tensors, loaded into both text encoders. SD3 and Flux
have no textual-inversion support in diffusers.

The tokenizer and text encoder are shared by the base pipeline and every
pipeline derived from it (img2img, inpaint, ControlNet), so an embedding is
loaded once per checkpoint and stays available, like a LoRA file. A
multi-vector embedding becomes `<name> <name>_1 ...`: diffusers' own
encode_prompt() expands that itself, so expand_vectors() is only for
PixelFlow's encoder (prompt_weights.py)."""

import re
import threading
from typing import Optional

from . import models, prompt_weights, runtime

EMBEDDING_EXTENSIONS = (".safetensors", ".pt", ".bin")
_REFERENCE = re.compile(r"embedding:([\w\-]+(?:/[\w\-]+)*)(?:\.(?:safetensors|pt|bin))?", re.IGNORECASE)
_load_lock = threading.Lock()


def referenced(text: Optional[str]):
    return [match.group(1) for match in _REFERENCE.finditer(text or "")]


def token_for(name: str) -> str:
    return f"<{name}>"


def embedding_path(name: str):
    for extension in EMBEDDING_EXTENSIONS:
        path = models.local_model_path("embeddings", name + extension)
        if path is not None and path.is_file():
            return path
    raise FileNotFoundError(f"embedding '{name}' is not in models/embeddings/ (.safetensors, .pt or .bin)")


def _read_state_dict(path):
    torch = runtime.torch()
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _load(pipe, family: str, name: str) -> None:
    token = token_for(name)
    if token in pipe.tokenizer.get_vocab():
        return
    path = embedding_path(name)
    state = _read_state_dict(path)
    try:
        if family == prompt_weights.FAMILY_SDXL:
            if "clip_l" not in state or "clip_g" not in state:
                raise ValueError("an SDXL embedding needs both 'clip_l' and 'clip_g' tensors (this looks like "
                                 "an SD 1.x/2.x embedding)")
            pipe.load_textual_inversion(state["clip_l"], token=token, text_encoder=pipe.text_encoder,
                                        tokenizer=pipe.tokenizer)
            pipe.load_textual_inversion(state["clip_g"], token=token, text_encoder=pipe.text_encoder_2,
                                        tokenizer=pipe.tokenizer_2)
        else:
            if "clip_l" in state and "clip_g" in state:
                raise ValueError("this is an SDXL embedding; the checkpoint is SD 1.x/2.x")
            pipe.load_textual_inversion(state, token=token)
    except ValueError as exc:
        raise ValueError(f"embedding '{name}' can't be used with this checkpoint: {exc}") from None


def prepare(pipe, family: Optional[str], text: Optional[str]) -> Optional[str]:
    """`text` with each `embedding:name` loaded (once) and replaced by its
    token. Raises a clear error for a missing file or an unsupported model."""
    names = referenced(text)
    if not names:
        return text
    if family not in (prompt_weights.FAMILY_SD, prompt_weights.FAMILY_SDXL) or not hasattr(
        pipe, "load_textual_inversion"
    ):
        raise ValueError("textual-inversion embeddings work with SD 1.x/2.x and SDXL checkpoints only")
    with _load_lock:
        for name in dict.fromkeys(names):
            _load(pipe, family, name)
    return _REFERENCE.sub(lambda match: token_for(match.group(1)), text)


def expand_vectors(pipe, text: Optional[str]) -> Optional[str]:
    """`<name>` -> `<name> <name>_1 ...` for multi-vector embeddings, as
    diffusers' encode_prompt() does, for PixelFlow's own encoder."""
    if not text or "<" not in text or not hasattr(pipe, "maybe_convert_prompt"):
        return text
    return pipe.maybe_convert_prompt(text, pipe.tokenizer)
