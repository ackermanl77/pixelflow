import json

import pytest
import torch
from diffusers import EulerDiscreteScheduler
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from requests.exceptions import ConnectionError as RequestsConnectionError

from backend.engine.graph import execute_prompt
from backend.nodes import generation, io_nodes

TINY_SD_CHECKPOINT = "hf-internal-testing/tiny-stable-diffusion-torch"


# ---- output filename sanitization ----


@pytest.mark.parametrize(
    "prefix,expected_start",
    [
        ("out.png:evil", "out.png_evil_"),  # NTFS alternate data stream
        ("a?b*c|d", "a_b_c_d_"),
        ("../../escape", "escape_"),
        ("C:\\Windows\\x", "x_"),
        ("..", "image_"),
        ("", "image_"),
    ],
)
def test_save_image_sanitizes_the_filename_prefix(output_dir, prefix, expected_start):
    result = io_nodes.SaveImage().save(Image.new("RGB", (4, 4)), prefix)
    (name,) = result["images"]
    assert name.startswith(expected_start)
    assert name.endswith(".png")
    assert (output_dir / name).is_file()
    assert [p.name for p in output_dir.iterdir()] == [name]


# ---- pipeline LRU cache ----


class _FakePipeline:
    instances = []

    def __init__(self, name):
        self.name = name
        _FakePipeline.instances.append(self)

    @classmethod
    def from_pretrained(cls, name, torch_dtype=None):
        return cls(name)

    def to(self, device):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass


@pytest.fixture
def fake_pipelines(monkeypatch):
    monkeypatch.setattr(io_nodes, "StableDiffusionPipeline", _FakePipeline)
    monkeypatch.setattr(io_nodes, "_pipeline_cache", io_nodes.OrderedDict())
    freed = []
    monkeypatch.setattr(io_nodes, "free_memory", lambda: freed.append(True))
    return freed


def test_pipeline_cache_evicts_the_least_recently_used(fake_pipelines, monkeypatch):
    monkeypatch.setattr(io_nodes, "MAX_CACHED_PIPELINES", 2)
    a = io_nodes._load_pipeline("repo/a")
    b = io_nodes._load_pipeline("repo/b")
    assert io_nodes._load_pipeline("repo/a") is a  # cache hit, and now most recent
    io_nodes._load_pipeline("repo/c")  # evicts b, not a
    names = [key[0] for key in io_nodes._pipeline_cache]
    assert names == ["repo/a", "repo/c"]
    assert fake_pipelines, "evicting must free (V)RAM"
    assert io_nodes._load_pipeline("repo/b") is not b  # reloaded


def test_default_cache_keeps_only_one_pipeline(fake_pipelines):
    io_nodes._load_pipeline("repo/a")
    io_nodes._load_pipeline("repo/b")
    assert [key[0] for key in io_nodes._pipeline_cache] == ["repo/b"]


# ---- out-of-memory handling ----


class _OomPipeline:
    device = torch.device("cpu")

    def __init__(self):
        self.scheduler = EulerDiscreteScheduler()

    def __call__(self, **kwargs):
        raise io_nodes.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB")


def test_ksampler_turns_oom_into_a_clear_error_and_frees_memory(monkeypatch):
    freed = []
    monkeypatch.setattr(generation, "free_memory", lambda: freed.append(True))
    with pytest.raises(RuntimeError, match="Out of GPU memory while sampling 512x512 \\(batch_size=4\\)"):
        generation.KSamplerStub().sample(
            model={"pipeline": _OomPipeline()},
            positive={"text": "x"},
            latent_image=Image.new("RGB", (512, 512)),
            seed=1,
            steps=1,
            cfg=0.0,
            batch_size=4,
        )
    assert freed


# ---- seed reporting + PNG metadata ----


def test_random_seed_is_reported_and_embedded_in_png_metadata(output_dir):
    prompt = {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY_SD_CHECKPOINT}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red circle"}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "color": "#000000"}},
        "sampler": {
            "class_type": "KSamplerStub",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["clip", 0],
                "latent_image": ["latent", 0],
                "seed": -1,
                "steps": 2,
                "cfg": 7.0,
            },
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "meta"}},
    }
    workflow = {"nodes": [{"id": 1}], "links": []}
    try:
        outputs = execute_prompt(prompt, extra_pnginfo={"workflow": workflow})
    except (HfHubHTTPError, RequestsConnectionError) as exc:
        pytest.skip(f"Hugging Face Hub unreachable: {exc}")

    seed = outputs["sampler"]["seed"]
    assert isinstance(seed, int) and seed >= 0
    assert prompt["sampler"]["inputs"]["seed"] == -1, "the caller's prompt must not be mutated"

    with Image.open(output_dir / outputs["save"]["images"][0]) as img:
        embedded_prompt = json.loads(img.text["prompt"])
        embedded_workflow = json.loads(img.text["workflow"])
    assert embedded_prompt["sampler"]["inputs"]["seed"] == seed
    assert embedded_workflow == workflow

    # Re-running the embedded prompt reproduces the same image.
    rerun = execute_prompt(embedded_prompt)
    assert rerun["sampler"]["seed"] == seed
    with Image.open(output_dir / outputs["save"]["images"][0]) as a, Image.open(
        output_dir / rerun["save"]["images"][0]
    ) as b:
        assert list(a.convert("RGB").getdata()) == list(b.convert("RGB").getdata())


def test_save_without_a_sampler_still_embeds_the_prompt(output_dir):
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "color": "#000000"}},
        "2": {"class_type": "PreviewImage", "inputs": {"image": ["1", 0]}},
    }
    outputs = execute_prompt(prompt)
    with Image.open(output_dir / outputs["2"]["images"][0]) as img:
        assert json.loads(img.text["prompt"]) == prompt
        assert "workflow" not in img.text
