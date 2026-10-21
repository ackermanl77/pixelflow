"""Textual-inversion embeddings in prompts: embedding:name (report item B22)."""

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from backend.engine.graph import execute_prompt, validate_prompt
from backend.nodes import embeddings, models, prompt_weights

from _hub import skip_if_hub_unreachable

TINY = "hf-internal-testing/tiny-stable-diffusion-torch"
TINY_HIDDEN = 32  # the tiny checkpoint's CLIP width


@pytest.fixture
def embedding_dir(isolated_data_dirs):
    folder = isolated_data_dirs["embeddings"]
    folder.mkdir(parents=True)
    return folder


def test_references_use_comfyuis_syntax_with_an_optional_extension():
    text = "a embedding:style, (embedding:people/face.pt:1.2) and embedding:x.safetensors"
    assert embeddings.referenced(text) == ["style", "people/face", "x"]
    assert embeddings.referenced("no embeddings here") == []


def test_embeddings_are_listed_like_other_models(embedding_dir):
    (embedding_dir / "style.safetensors").write_bytes(b"x")
    (embedding_dir / "old.pt").write_bytes(b"x")
    (embedding_dir / "notes.txt").write_text("x")
    assert models.list_models("embeddings") == ["old.pt", "style.safetensors"]


class _Tokenizer:
    def __init__(self):
        self.vocab = {"a": 0}

    def get_vocab(self):
        return dict(self.vocab)


class _FakePipe:
    """Records load_textual_inversion calls the way diffusers' mixin is used."""

    def __init__(self, sdxl=False):
        self.tokenizer, self.text_encoder = _Tokenizer(), object()
        if sdxl:
            self.tokenizer_2, self.text_encoder_2 = _Tokenizer(), object()
        self.calls = []

    def load_textual_inversion(self, state, token=None, text_encoder=None, tokenizer=None):
        self.calls.append((sorted(state) if isinstance(state, dict) else tuple(state.shape), token,
                           text_encoder, tokenizer))
        (tokenizer or self.tokenizer).vocab[token] = len((tokenizer or self.tokenizer).vocab)


def test_an_sdxl_embedding_loads_into_both_text_encoders_once(embedding_dir):
    save_file({"clip_l": torch.zeros(2, 768), "clip_g": torch.zeros(2, 1280)}, str(embedding_dir / "s.safetensors"))
    pipe = _FakePipe(sdxl=True)
    text = embeddings.prepare(pipe, prompt_weights.FAMILY_SDXL, "a embedding:s photo, embedding:s")
    assert text == "a <s> photo, <s>"
    assert [(shape, token) for shape, token, _e, _t in pipe.calls] == [((2, 768), "<s>"), ((2, 1280), "<s>")]
    assert pipe.calls[0][2] is pipe.text_encoder and pipe.calls[1][2] is pipe.text_encoder_2
    embeddings.prepare(pipe, prompt_weights.FAMILY_SDXL, "embedding:s again")
    assert len(pipe.calls) == 2, "loaded once per checkpoint"


def test_mismatched_missing_and_unsupported_embeddings_are_clear_errors(embedding_dir):
    save_file({"emb_params": torch.zeros(1, 768)}, str(embedding_dir / "sd15.safetensors"))
    with pytest.raises(ValueError, match="needs both 'clip_l' and 'clip_g'"):
        embeddings.prepare(_FakePipe(sdxl=True), prompt_weights.FAMILY_SDXL, "embedding:sd15")
    save_file({"clip_l": torch.zeros(1, 768), "clip_g": torch.zeros(1, 1280)}, str(embedding_dir / "xl.safetensors"))
    with pytest.raises(ValueError, match="SDXL embedding; the checkpoint is SD 1.x/2.x"):
        embeddings.prepare(_FakePipe(), prompt_weights.FAMILY_SD, "embedding:xl")
    with pytest.raises(FileNotFoundError, match="'nope' is not in models/embeddings/"):
        embeddings.prepare(_FakePipe(), prompt_weights.FAMILY_SD, "embedding:nope")
    with pytest.raises(ValueError, match="SD 1.x/2.x and SDXL checkpoints only"):
        embeddings.prepare(_FakePipe(), prompt_weights.FAMILY_FLUX, "embedding:sd15")
    assert embeddings.prepare(_FakePipe(), prompt_weights.FAMILY_FLUX, "no reference") == "no reference"


# ---- with the tiny real pipeline ----


def _graph(text, seed=3):
    return {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": text}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#000000"}},
        "sampler": {
            "class_type": "KSampler",
            "inputs": {"model": ["ckpt", 0], "positive": ["clip", 0], "latent_image": ["latent", 0],
                       "sampler": "euler", "seed": seed, "steps": 2, "cfg": 7.0, "batch_size": 1},
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "emb"}},
    }


def _image(prompt, output_dir):
    try:
        outputs = execute_prompt(validate_prompt(_graph(prompt)))
    except Exception as exc:
        skip_if_hub_unreachable(exc)
        raise
    with Image.open(output_dir / outputs["save"]["images"][0]) as img:
        return list(img.convert("RGB").getdata())


def test_an_embedding_changes_the_image_on_both_encoder_paths(output_dir, embedding_dir, monkeypatch):
    generator = torch.Generator().manual_seed(0)
    save_file({"emb_params": torch.randn(2, TINY_HIDDEN, generator=generator) * 5},
              str(embedding_dir / "loud.safetensors"))
    # Plain short prompt: diffusers' own encode_prompt() expands the two vectors.
    assert _image("a red circle", output_dir) != _image("a red circle embedding:loud", output_dir)
    # A real weight goes through PixelFlow's encoder, which must get the expanded tokens.
    seen = []
    original = prompt_weights.encode_conditioning

    def recording(pipe, family, positive, negative, clip_skip=0):
        seen.append(positive)
        return original(pipe, family, positive, negative, clip_skip)

    monkeypatch.setattr(prompt_weights, "encode_conditioning", recording)
    weighted = _image("(a red circle:1.3) embedding:loud", output_dir)
    assert seen == ["(a red circle:1.3) <loud> <loud>_1"]
    assert weighted != _image("(a red circle:1.3)", output_dir)
