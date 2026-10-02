"""Prompt weighting and long-prompt chunking for CLIP text encoders
(report item B9).

Syntax (the A1111/ComfyUI one users already know):

    (word)        weight x 1.1         ((word)) x 1.21, and so on
    [word]        weight / 1.1
    (word:1.4)    weight x 1.4 (also for a whole phrase)
    \\( \\) \\[ \\]   literal brackets

Prompts longer than CLIP's 77-token window are split into 75-token chunks
that are encoded separately and concatenated, instead of being truncated.
Used for SD 1.x/2.x checkpoints (one CLIP text encoder); other
architectures get the plain text with the weighting syntax stripped.
"""

import re
from typing import List, Sequence, Tuple

ROUND_MULTIPLIER = 1.1
SQUARE_MULTIPLIER = 1 / 1.1

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


def tokenize_weighted(tokenizer, parsed: Sequence[Tuple[str, float]]) -> Tuple[List[int], List[float]]:
    ids: List[int] = []
    weights: List[float] = []
    for text, weight in parsed:
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


def needs_embeddings(tokenizer, text: str) -> bool:
    """True when plain-text prompting would lose something: weights, or more
    tokens than one CLIP window holds."""
    parsed = parse_prompt_weights(text)
    if has_weights(parsed):
        return True
    ids, _ = tokenize_weighted(tokenizer, parsed)
    return len(ids) > tokenizer.model_max_length - 2


def _encode_chunks(pipe, chunks, clip_skip: int):
    import torch

    tokenizer, encoder = pipe.tokenizer, pipe.text_encoder
    window = tokenizer.model_max_length
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    device = getattr(pipe, "_execution_device", None) or pipe.device
    pieces = []
    for ids, weights in chunks:
        padding = window - 2 - len(ids)
        row = [bos] + list(ids) + [eos] + [pad] * padding
        row_weights = [1.0] + list(weights) + [1.0] + [1.0] * padding
        input_ids = torch.tensor([row], device=device)
        with torch.no_grad():
            if clip_skip and clip_skip > 0:
                out = encoder(input_ids, output_hidden_states=True)
                hidden = out.hidden_states[-(clip_skip + 1)]
                # transformers 5 flattened CLIPTextModel (no .text_model).
                hidden = getattr(encoder, "text_model", encoder).final_layer_norm(hidden)
            else:
                hidden = encoder(input_ids)[0]
        weight_tensor = torch.tensor(row_weights, device=hidden.device, dtype=hidden.dtype)
        if any(abs(w - 1.0) > 1e-6 for w in row_weights):
            # A1111's approach: scale each token's embedding, then restore the
            # chunk's original mean so the overall magnitude stays the same.
            original_mean = hidden.float().mean()
            hidden = hidden * weight_tensor.reshape(1, -1, 1)
            new_mean = hidden.float().mean()
            if new_mean.abs() > 1e-12:
                hidden = hidden * (original_mean / new_mean).to(hidden.dtype)
        pieces.append(hidden)
    return torch.cat(pieces, dim=1)


def encode_prompt_pair(pipe, positive: str, negative: str, clip_skip: int = 0):
    """(prompt_embeds, negative_prompt_embeds), both [1, n*77, dim] with the
    same n so they can be used together for classifier-free guidance."""
    tokenizer = pipe.tokenizer
    size = tokenizer.model_max_length - 2
    pos_chunks = chunk_tokens(*tokenize_weighted(tokenizer, parse_prompt_weights(positive)), size)
    neg_chunks = chunk_tokens(*tokenize_weighted(tokenizer, parse_prompt_weights(negative or "")), size)
    count = max(len(pos_chunks), len(neg_chunks))
    pos_chunks += [([], [])] * (count - len(pos_chunks))
    neg_chunks += [([], [])] * (count - len(neg_chunks))
    return _encode_chunks(pipe, pos_chunks, clip_skip), _encode_chunks(pipe, neg_chunks, clip_skip)
