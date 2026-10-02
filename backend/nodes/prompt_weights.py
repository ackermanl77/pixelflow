"""Prompt weighting, long prompts and clip skip: PixelFlow's own text
encoding for every supported model family (report item B9).

Syntax (the A1111/ComfyUI one users already know):

    (word)        weight x 1.1         ((word)) x 1.21, and so on
    [word]        weight / 1.1
    (word:1.4)    weight x 1.4 (also for a whole phrase)
    \\( \\) \\[ \\]   literal brackets

Weights scale each token's embedding, after which the chunk's original mean
is restored (A1111's method), separately for every text encoder.

How each family's encoders are used (the way that family's own pipeline
does it, minus the truncation):

- SD 1.x/2.x: one CLIP. Prompts longer than 75 tokens are split into
  75-token chunks (each wrapped as BOS + 75 + EOS = 77) that are encoded
  separately and concatenated. Clip skip takes an earlier hidden layer and
  re-applies the final layer norm.
- SDXL: CLIP-L + OpenCLIP-G, penultimate hidden layer of each, concatenated
  per token (768 + 1280). Long prompts are chunked the same way for both;
  the pooled embedding comes from CLIP-G's projection of the first chunk.
  Clip skip goes N layers further back than SDXL's default penultimate one.
  An unconnected negative prompt gives all-zero embeddings when the
  checkpoint asks for it (force_zeros_for_empty_prompt), as in diffusers.
- SD3: CLIP-L + CLIP-G as for SDXL (chunked), zero-padded to T5's width and
  followed by the T5-XXL tokens (256 by default, up to 512 for a longer
  prompt); pooled = both CLIP projections of the first chunk. Clip skip
  applies to the CLIP part.
- Flux: T5-XXL carries the prompt (up to 512 tokens, weighted); CLIP-L only
  supplies the pooled vector, from the first 75 tokens. Flux has no clip
  skip and no negative prompt (it is guidance-distilled).

Prompts that need none of this (no weights, short enough, no clip skip) are
still passed to the pipeline as plain text, so diffusers' own encoding stays
in charge of the common case.
"""

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import models

logger = logging.getLogger(__name__)

ROUND_MULTIPLIER = 1.1
SQUARE_MULTIPLIER = 1 / 1.1

# T5 sequence lengths diffusers uses (SD3 defaults to 256 and accepts up to
# 512; Flux uses 512).
SD3_T5_DEFAULT_TOKENS = 256
T5_MAX_TOKENS = 512
FLUX_T5_TOKENS = 512

FAMILY_SD = "sd"
FAMILY_SDXL = "sdxl"
FAMILY_SD3 = "sd3"
FAMILY_FLUX = "flux"

_TOKENS = re.compile(
    r"""
    \\\(|\\\)|\\\[|\\\]|\\\\|\\|   # escapes
    \(|\[|                         # openers
    :\s*([+-]?[.\d]+)\s*\)|        # (text:1.2) closer with a weight
    \)|\]|                         # plain closers
    [^\\()\[\]:]+|                 # text
    :                              # a stray colon
    """,
    re.VERBOSE,
)


def parse_prompt_weights(text: str) -> List[Tuple[str, float]]:
    """Split a prompt into (text, weight) runs; neighbouring runs with the
    same weight are merged. Unbalanced brackets apply to the end of the
    prompt, like A1111."""
    res: List[List] = []
    round_open: List[int] = []
    square_open: List[int] = []

    def multiply(start: int, factor: float) -> None:
        for item in res[start:]:
            item[1] *= factor

    for match in _TOKENS.finditer(text or ""):
        token, weight = match.group(0), match.group(1)
        if token.startswith("\\"):
            res.append([token[1:] if len(token) > 1 else token, 1.0])
        elif token == "(":
            round_open.append(len(res))
        elif token == "[":
            square_open.append(len(res))
        elif weight is not None and round_open:
            try:
                multiply(round_open.pop(), float(weight))
            except ValueError:
                res.append([token, 1.0])
        elif token == ")" and round_open:
            multiply(round_open.pop(), ROUND_MULTIPLIER)
        elif token == "]" and square_open:
            multiply(square_open.pop(), SQUARE_MULTIPLIER)
        else:
            res.append([token, 1.0])
    for start in round_open:
        multiply(start, ROUND_MULTIPLIER)
    for start in square_open:
        multiply(start, SQUARE_MULTIPLIER)

    merged: List[List] = []
    for item in res:
        if merged and abs(merged[-1][1] - item[1]) < 1e-9:
            merged[-1][0] += item[0]
        else:
            merged.append(list(item))
    return [(t, round(w, 6)) for t, w in merged] or [("", 1.0)]


def plain_text(text: str) -> str:
    """The prompt with the weighting syntax removed."""
    return "".join(t for t, _ in parse_prompt_weights(text))


def has_weights(parsed: Sequence[Tuple[str, float]]) -> bool:
    return any(abs(w - 1.0) > 1e-6 for t, w in parsed if t.strip())


def tokenize_weighted(
    tokenizer, parsed: Sequence[Tuple[str, float]], strip: bool = False
) -> Tuple[List[int], List[float]]:
    """Token ids (no special tokens) with each token's weight. `strip` trims
    each run first (for SentencePiece tokenizers like T5's, where a run's
    edge spaces would otherwise turn into tokens of their own)."""
    ids: List[int] = []
    weights: List[float] = []
    for text, weight in parsed:
        if strip:
            text = text.strip()
        if not text:
            continue
        piece = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False).input_ids
        ids += piece
        weights += [weight] * len(piece)
    return ids, weights


def chunk_tokens(ids: Sequence[int], weights: Sequence[float], chunk_size: int) -> List[Tuple[List[int], List[float]]]:
    """Split into windows of at most `chunk_size` tokens (always >= 1 chunk)."""
    if not ids:
        return [([], [])]
    return [
        (list(ids[i : i + chunk_size]), list(weights[i : i + chunk_size])) for i in range(0, len(ids), chunk_size)
    ]


def _clip_window(tokenizer) -> int:
    """Prompt tokens per CLIP chunk (model_max_length minus BOS and EOS)."""
    return tokenizer.model_max_length - 2


def needs_embeddings(tokenizer, text: str) -> bool:
    """True when plain-text prompting would lose something: weights, or more
    tokens than one CLIP window holds."""
    parsed = parse_prompt_weights(text)
    if has_weights(parsed):
        return True
    ids, _ = tokenize_weighted(tokenizer, parsed)
    return len(ids) > _clip_window(tokenizer)


def _t5_token_count(tokenizer, text: str) -> int:
    ids, _ = tokenize_weighted(tokenizer, parse_prompt_weights(text), strip=True)
    return len(ids) + 1  # + EOS


# ---------------------------------------------------------------- families


def _has(pipe, *names) -> bool:
    return all(getattr(pipe, name, None) is not None for name in names)


def encoding_family(pipe, arch: Optional[str]) -> Optional[str]:
    """Which of our encoders fits this pipeline, or None (plain text only)."""
    if arch in (None, models.ARCH_SD):
        if (
            _has(pipe, "tokenizer", "text_encoder")
            and getattr(pipe, "tokenizer_2", None) is None
            and hasattr(pipe.tokenizer, "model_max_length")
        ):
            return FAMILY_SD
        return None
    if arch == models.ARCH_SDXL and _has(pipe, "tokenizer_2", "text_encoder_2"):
        return FAMILY_SDXL
    if arch == models.ARCH_SD3 and _has(pipe, "tokenizer", "text_encoder", "tokenizer_2", "text_encoder_2"):
        return FAMILY_SD3
    if arch == models.ARCH_FLUX and _has(pipe, "tokenizer", "text_encoder", "tokenizer_2", "text_encoder_2"):
        return FAMILY_FLUX
    return None


def _clip_tokenizers(pipe, family: str) -> List:
    if family == FAMILY_SD:
        return [pipe.tokenizer]
    if family == FAMILY_SDXL:
        return [tok for tok in (getattr(pipe, "tokenizer", None), pipe.tokenizer_2) if tok is not None]
    if family == FAMILY_SD3:
        return [pipe.tokenizer, pipe.tokenizer_2]
    return []  # Flux: CLIP only gives the pooled vector; its 77-token cut is by design


def _t5_tokenizer(pipe, family: str):
    if family == FAMILY_SD3 and _has(pipe, "tokenizer_3", "text_encoder_3"):
        return pipe.tokenizer_3
    if family == FAMILY_FLUX:
        return pipe.tokenizer_2
    return None


def needs_custom_encoding(pipe, family: Optional[str], texts: Sequence[Optional[str]], clip_skip: int = 0) -> bool:
    """True when this family's plain-text path would lose something for any
    of `texts`: weighting, CLIP's 77-token window (SD/SDXL/SD3), SD3's
    default T5 length, or clip skip (not for Flux, which has none)."""
    if family is None:
        return False
    if clip_skip > 0 and family != FAMILY_FLUX:
        return True
    clip = _clip_tokenizers(pipe, family)
    t5 = _t5_tokenizer(pipe, family)
    if family == FAMILY_FLUX:
        texts = list(texts)[:1]  # Flux has no negative prompt to encode
    for text in texts:
        if not text:
            continue
        if has_weights(parse_prompt_weights(text)):
            return True
        if any(needs_embeddings(tokenizer, text) for tokenizer in clip):
            return True
        if family == FAMILY_SD3 and t5 is not None and _t5_token_count(t5, text) > SD3_T5_DEFAULT_TOKENS:
            return True
    return False


# ---------------------------------------------------------------- encoders


def _device(pipe):
    return getattr(pipe, "_execution_device", None) or pipe.device


def _module_dtype(module):
    return getattr(module, "dtype", None)


def _apply_weights(hidden, row_weights):
    """A1111's approach: scale each token's embedding, then restore the
    original mean so the overall magnitude stays the same."""
    import torch

    if not any(abs(w - 1.0) > 1e-6 for w in row_weights):
        return hidden
    weight_tensor = torch.tensor(row_weights, device=hidden.device, dtype=hidden.dtype)
    original_mean = hidden.float().mean()
    hidden = hidden * weight_tensor.reshape(1, -1, 1)
    new_mean = hidden.float().mean()
    if new_mean.abs() > 1e-12:
        hidden = hidden * (original_mean / new_mean).to(hidden.dtype)
    return hidden


def _clip_row(tokenizer, ids, weights):
    """One CLIP window: BOS + tokens + EOS, padded to model_max_length."""
    window = tokenizer.model_max_length
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    padding = window - 2 - len(ids)
    return [bos] + list(ids) + [eos] + [pad] * padding, [1.0] + list(weights) + [1.0] + [1.0] * padding


def _pooled_of(out):
    """The pooled vector: a CLIPTextModelWithProjection's text_embeds
    (SDXL's CLIP-G, SD3's encoders) or a CLIPTextModel's pooler_output (Flux)."""
    pooled = getattr(out, "text_embeds", None)
    if pooled is None:
        pooled = getattr(out, "pooler_output", None)
    return pooled


def _encode_clip(encoder, tokenizer, chunks, device, select):
    """Encode `chunks` with one CLIP encoder: (hidden [1, n*77, dim], the
    pooled vector of the first chunk). `select(encoder, output)` picks the
    hidden layer."""
    import torch

    pieces, pooled = [], None
    for ids, weights in chunks:
        row, row_weights = _clip_row(tokenizer, ids, weights)
        input_ids = torch.tensor([row], device=device)
        with torch.no_grad():
            out = encoder(input_ids, output_hidden_states=True)
        if pooled is None:
            pooled = _pooled_of(out)
        pieces.append(_apply_weights(select(encoder, out), row_weights))
    return torch.cat(pieces, dim=1), pooled


def _select_sd(clip_skip: int):
    def select(encoder, out):
        if clip_skip and clip_skip > 0:
            hidden = out.hidden_states[-(clip_skip + 1)]
            # transformers 5 flattened CLIPTextModel (no .text_model).
            return getattr(encoder, "text_model", encoder).final_layer_norm(hidden)
        return out[0]

    return select


def _select_penultimate(clip_skip: int):
    # SDXL and SD3 read the penultimate layer; clip skip goes further back.
    def select(encoder, out):
        return out.hidden_states[-(max(int(clip_skip or 0), 0) + 2)]

    return select


def _paired_chunks(tokenizer, positive: str, negative: Optional[str], count: int = 1):
    """Positive/negative chunks of one tokenizer, padded with empty chunks to
    the same count (at least `count`) so CFG can batch them together."""
    size = _clip_window(tokenizer)
    pos = chunk_tokens(*tokenize_weighted(tokenizer, parse_prompt_weights(positive)), size)
    neg = chunk_tokens(*tokenize_weighted(tokenizer, parse_prompt_weights(negative or "")), size)
    count = max(count, len(pos), len(neg))
    return pos + [([], [])] * (count - len(pos)), neg + [([], [])] * (count - len(neg))


def _chunk_count(tokenizers, texts) -> int:
    """The most chunks any of `texts` needs with any of `tokenizers` (so all
    encoders' outputs line up token by token)."""
    count = 1
    for tokenizer in tokenizers:
        size = _clip_window(tokenizer)
        for text in texts:
            ids, weights = tokenize_weighted(tokenizer, parse_prompt_weights(text or ""))
            count = max(count, len(chunk_tokens(ids, weights, size)))
    return count


def _encode_t5(encoder, tokenizer, text: str, length: int, device):
    """T5 hidden states [1, length, dim] for a weighted prompt: tokens + EOS,
    padded (or truncated) to `length`."""
    import torch

    ids, weights = tokenize_weighted(tokenizer, parse_prompt_weights(text or ""), strip=True)
    if len(ids) > length - 1:
        logger.warning("prompt is longer than %d T5 tokens; the rest is ignored", length - 1)
        ids, weights = ids[: length - 1], weights[: length - 1]
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    padding = length - 1 - len(ids)
    row = list(ids) + [eos] + [pad] * padding
    row_weights = list(weights) + [1.0] + [1.0] * padding
    with torch.no_grad():
        hidden = encoder(torch.tensor([row], device=device))[0]
    return _apply_weights(hidden, row_weights)


def encode_prompt_pair(pipe, positive: str, negative: str, clip_skip: int = 0):
    """SD 1.x/2.x: (prompt_embeds, negative_prompt_embeds), both
    [1, n*77, dim] with the same n so they can be used together for
    classifier-free guidance."""
    pos_chunks, neg_chunks = _paired_chunks(pipe.tokenizer, positive, negative)
    device = _device(pipe)
    select = _select_sd(clip_skip)
    positive_embeds, _ = _encode_clip(pipe.text_encoder, pipe.tokenizer, pos_chunks, device, select)
    negative_embeds, _ = _encode_clip(pipe.text_encoder, pipe.tokenizer, neg_chunks, device, select)
    return positive_embeds, negative_embeds


def encode_sdxl(pipe, positive: str, negative: Optional[str], clip_skip: int = 0) -> Dict:
    """SDXL call kwargs: prompt/negative embeds [1, n*77, 2048] and the
    pooled [1, 1280] embeds. `negative` None means "not connected"."""
    import torch

    pairs = [
        (tok, enc)
        for tok, enc in (
            (getattr(pipe, "tokenizer", None), getattr(pipe, "text_encoder", None)),
            (pipe.tokenizer_2, pipe.text_encoder_2),
        )
        if tok is not None and enc is not None
    ]
    count = _chunk_count([tok for tok, _ in pairs], [positive, negative])
    device = _device(pipe)
    select = _select_penultimate(clip_skip)
    pos_hidden, neg_hidden, pooled, neg_pooled = [], [], None, None
    for tokenizer, encoder in pairs:
        pos_chunks, neg_chunks = _paired_chunks(tokenizer, positive, negative, count)
        # The pooled vectors that survive are the last encoder's (CLIP-G).
        hidden, pooled = _encode_clip(encoder, tokenizer, pos_chunks, device, select)
        pos_hidden.append(hidden)
        hidden, neg_pooled = _encode_clip(encoder, tokenizer, neg_chunks, device, select)
        neg_hidden.append(hidden)
    dtype = _module_dtype(pipe.text_encoder_2) or _module_dtype(getattr(pipe, "unet", None))
    prompt_embeds = torch.cat(pos_hidden, dim=-1).to(dtype=dtype, device=device)
    pooled = pooled.to(dtype=dtype, device=device)
    config = getattr(pipe, "config", None) or {}
    if negative is None and config.get("force_zeros_for_empty_prompt", False):
        negative_embeds, neg_pooled = torch.zeros_like(prompt_embeds), torch.zeros_like(pooled)
    else:
        negative_embeds = torch.cat(neg_hidden, dim=-1).to(dtype=dtype, device=device)
        neg_pooled = neg_pooled.to(dtype=dtype, device=device)
    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_embeds,
        "pooled_prompt_embeds": pooled,
        "negative_pooled_prompt_embeds": neg_pooled,
    }


def encode_sd3(pipe, positive: str, negative: Optional[str], clip_skip: int = 0) -> Dict:
    """SD3 call kwargs: [CLIP-L|CLIP-G chunks zero-padded to T5's width ;
    T5 tokens] and the pooled [1, 2048] embeds."""
    import torch

    pairs = [(pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)]
    count = _chunk_count([tok for tok, _ in pairs], [positive, negative])
    device = _device(pipe)
    select = _select_penultimate(clip_skip)
    sides = {"pos": positive, "neg": negative or ""}
    hidden = {"pos": [], "neg": []}
    pooled = {"pos": [], "neg": []}
    for tokenizer, encoder in pairs:
        chunks = dict(zip(("pos", "neg"), _paired_chunks(tokenizer, positive, negative, count)))
        for side in sides:
            h, p = _encode_clip(encoder, tokenizer, chunks[side], device, select)
            hidden[side].append(h)
            pooled[side].append(p)

    t5_tokenizer = _t5_tokenizer(pipe, FAMILY_SD3)
    dtype = _module_dtype(getattr(pipe, "text_encoder_3", None)) or _module_dtype(pipe.text_encoder)
    if t5_tokenizer is not None:
        needed = max(_t5_token_count(t5_tokenizer, text) for text in sides.values())
        length = min(max(SD3_T5_DEFAULT_TOKENS, needed), T5_MAX_TOKENS)
        t5 = {side: _encode_t5(pipe.text_encoder_3, t5_tokenizer, text, length, device) for side, text in sides.items()}
    else:  # loaded without T5 (text_encoder_3=None): zeros, as diffusers does
        transformer = getattr(pipe, "transformer", None)
        width = getattr(getattr(transformer, "config", None), "joint_attention_dim", 4096)
        zeros = torch.zeros((1, SD3_T5_DEFAULT_TOKENS, width), device=device, dtype=dtype)
        t5 = {"pos": zeros, "neg": zeros}

    out = {}
    for side, prefix in (("pos", ""), ("neg", "negative_")):
        clip = torch.cat(hidden[side], dim=-1).to(dtype=dtype, device=device)
        clip = torch.nn.functional.pad(clip, (0, t5[side].shape[-1] - clip.shape[-1]))
        out[prefix + "prompt_embeds"] = torch.cat([clip, t5[side].to(dtype=dtype, device=device)], dim=-2)
        out[prefix + "pooled_prompt_embeds"] = torch.cat(pooled[side], dim=-1).to(dtype=dtype, device=device)
    return out


def encode_flux(pipe, positive: str, negative: Optional[str] = None, clip_skip: int = 0) -> Dict:
    """Flux call kwargs: T5 prompt_embeds [1, 512, 4096] (weighted) and
    CLIP-L's pooled [1, 768] from the first 75 tokens. Negative prompts and
    clip skip don't apply to Flux."""
    if clip_skip:
        logger.info("clip_skip has no effect on Flux checkpoints")
    tokenizer = pipe.tokenizer
    ids, weights = tokenize_weighted(tokenizer, parse_prompt_weights(positive))
    first = chunk_tokens(ids, weights, _clip_window(tokenizer))[:1]
    device = _device(pipe)
    _, pooled = _encode_clip(pipe.text_encoder, tokenizer, first, device, _select_sd(0))
    prompt_embeds = _encode_t5(pipe.text_encoder_2, pipe.tokenizer_2, positive, FLUX_T5_TOKENS, device)
    return {
        "prompt_embeds": prompt_embeds.to(dtype=_module_dtype(pipe.text_encoder_2), device=device),
        "pooled_prompt_embeds": pooled.to(dtype=_module_dtype(pipe.text_encoder), device=device),
    }


def encode_conditioning(pipe, family: str, positive: str, negative: Optional[str], clip_skip: int = 0) -> Dict:
    """Pipeline call kwargs (embeds instead of text) for `family`.
    `negative` None means no negative prompt is connected."""
    if family == FAMILY_SD:
        prompt_embeds, negative_embeds = encode_prompt_pair(pipe, positive, negative or "", clip_skip)
        return {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_embeds}
    if family == FAMILY_SDXL:
        return encode_sdxl(pipe, positive, negative, clip_skip)
    if family == FAMILY_SD3:
        return encode_sd3(pipe, positive, negative, clip_skip)
    if family == FAMILY_FLUX:
        return encode_flux(pipe, positive, negative, clip_skip)
    raise ValueError(f"no prompt encoder for {family!r}")
