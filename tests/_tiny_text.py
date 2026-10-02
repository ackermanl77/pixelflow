"""Tiny, randomly initialised text encoders and a word-level tokenizer for
testing prompt encoding (report item B9) without downloading anything:
real transformers CLIP/T5 classes built from minimal configs, wired into
real diffusers SDXL/SD3/Flux pipeline objects (with no UNet/transformer/VAE),
so our encoding can be compared with each pipeline's own encode_prompt()."""

import re
import zlib
from types import SimpleNamespace

import torch

VOCAB = 1000
CLIP_BOS, CLIP_EOS = 3, 4  # not 2: transformers treats eos 2 as "legacy, use argmax"
T5_PAD, T5_EOS = 0, 1


class WordTokenizer:
    """Enough of the Hugging Face tokenizer API for diffusers and
    prompt_weights: one id per word or punctuation mark (a stable hash)."""

    def __init__(self, model_max_length=77, bos=CLIP_BOS, eos=CLIP_EOS, pad=None):
        self.model_max_length = model_max_length
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.pad_token_id = pad

    def _ids(self, text):
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        return [10 + zlib.crc32(word.encode()) % (VOCAB - 10) for word in words]

    def _encode(self, text, add_special_tokens):
        ids = self._ids(text)
        if add_special_tokens:
            ids = ([self.bos_token_id] if self.bos_token_id is not None else []) + ids + [self.eos_token_id]
        return ids

    def __call__(
        self,
        text,
        padding=False,
        max_length=None,
        truncation=False,
        return_tensors=None,
        add_special_tokens=True,
        **_,
    ):
        texts = [text] if isinstance(text, str) else list(text)
        rows = [self._encode(t, add_special_tokens) for t in texts]
        pad = self.pad_token_id if self.pad_token_id is not None else self.eos_token_id
        if truncation and max_length:
            rows = [r if len(r) <= max_length else r[: max_length - 1] + [self.eos_token_id] for r in rows]
        if padding == "max_length" and max_length:
            rows = [r + [pad] * (max_length - len(r)) for r in rows]
        elif padding == "longest":
            longest = max(len(r) for r in rows)
            rows = [r + [pad] * (longest - len(r)) for r in rows]
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor(rows))
        return SimpleNamespace(input_ids=rows[0] if isinstance(text, str) else rows)

    # Textual-inversion hooks diffusers' SD/SDXL pipelines call.
    added_tokens_encoder: dict = {}

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def batch_decode(self, ids, **_):
        return ["<tokens>" for _ in ids]


def clip_tokenizer(pad=None):
    return WordTokenizer(pad=pad)


def t5_tokenizer():
    return WordTokenizer(model_max_length=512, bos=None, eos=T5_EOS, pad=T5_PAD)


def _clip_config(hidden=32, projection=16):
    from transformers import CLIPTextConfig

    return CLIPTextConfig(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=37,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=77,
        projection_dim=projection,
        bos_token_id=CLIP_BOS,
        eos_token_id=CLIP_EOS,
        pad_token_id=CLIP_EOS,
    )


def clip_model(hidden=32, projection=None, seed=0):
    """CLIPTextModel, or CLIPTextModelWithProjection when `projection`."""
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    torch.manual_seed(seed)
    if projection:
        model = CLIPTextModelWithProjection(_clip_config(hidden, projection))
    else:
        model = CLIPTextModel(_clip_config(hidden))
    return model.eval()


def t5_model(width=48, seed=0):
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(seed)
    config = T5Config(vocab_size=VOCAB, d_model=width, d_kv=8, d_ff=37, num_layers=2, num_heads=4)
    return T5EncoderModel(config).eval()


def sdxl_pipe(force_zeros=True):
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline(
        vae=None,
        text_encoder=clip_model(32, seed=1),
        text_encoder_2=clip_model(48, projection=24, seed=2),
        tokenizer=clip_tokenizer(),
        tokenizer_2=clip_tokenizer(pad=0),  # SDXL's second tokenizer pads with "!" (id 0)
        unet=None,
        scheduler=None,
        force_zeros_for_empty_prompt=force_zeros,
    )
    return pipe


class _Transformer:
    """Stands in for SD3's transformer: encode_prompt only reads its config."""

    config = SimpleNamespace(joint_attention_dim=64, sample_size=8, patch_size=2)


def sd3_pipe(with_t5=True):
    from diffusers import StableDiffusion3Pipeline

    return StableDiffusion3Pipeline(
        transformer=_Transformer(),
        scheduler=None,
        vae=None,
        text_encoder=clip_model(16, projection=8, seed=3),
        tokenizer=clip_tokenizer(),
        text_encoder_2=clip_model(24, projection=12, seed=4),
        tokenizer_2=clip_tokenizer(pad=0),
        text_encoder_3=t5_model(64, seed=5) if with_t5 else None,
        tokenizer_3=t5_tokenizer() if with_t5 else None,
    )


def flux_pipe():
    from diffusers import FluxPipeline

    return FluxPipeline(
        scheduler=None,
        vae=None,
        text_encoder=clip_model(32, seed=6),
        tokenizer=clip_tokenizer(),
        text_encoder_2=t5_model(48, seed=7),
        tokenizer_2=t5_tokenizer(),
        transformer=None,
    )
