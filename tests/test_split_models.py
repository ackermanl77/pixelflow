"""Split-file models (B26): a diffusion model, text encoder and VAE as separate files, ComfyUI's
layout. The real Z-Image Turbo files are only used by the opt-in test at the end
(PIXELFLOW_SPLIT_MODEL_TESTS=1, files under D:/models or PIXELFLOW_SPLIT_MODEL_DIR)."""

import json
import os
from pathlib import Path

import pytest

from backend.nodes import generation, io_nodes, models, split_models

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _write(path: Path, tensors: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file({key: value.contiguous() for key, value in tensors.items()}, str(path))
    return path


def test_split_names_round_trip():
    name = split_models.split_model_name("sub/z.safetensors", "qwen.safetensors", "ae.safetensors")
    assert split_models.parse_split_name(name) == ("sub/z.safetensors", "qwen.safetensors", "ae.safetensors")
    assert split_models.parse_split_name("stabilityai/sd-turbo") is None
    with pytest.raises(ValueError):
        split_models.parse_split_name("split:only|two")


def test_the_family_comes_from_the_diffusion_models_keys(tmp_path):
    z_image = _write(tmp_path / "z.safetensors", {"model.diffusion_model.layers.0.adaLN_modulation.0.weight": torch.zeros(2)})
    model_type, family = split_models.detect_family(z_image)
    assert model_type == "z-image-turbo" and family.arch == models.ARCH_ZIMAGE and family.pipeline == "ZImagePipeline"
    other = _write(tmp_path / "other.safetensors", {"something.else.weight": torch.zeros(2)})
    with pytest.raises(ValueError, match="supports z-image-turbo"):
        split_models.detect_family(other)


def test_component_files_resolve_in_their_own_folders(isolated_data_dirs):
    _write(isolated_data_dirs["diffusion_models"] / "z.safetensors", {"model.diffusion_model.layers.0.adaLN_modulation.0.weight": torch.zeros(2)})
    _write(isolated_data_dirs["text_encoders"] / "te.safetensors", {"x": torch.zeros(1)})
    _write(isolated_data_dirs["vae"] / "ae.safetensors", {"x": torch.zeros(1)})
    name = split_models.split_model_name("z.safetensors", "te.safetensors", "ae.safetensors")
    assert split_models.split_arch(name) == models.ARCH_ZIMAGE
    with pytest.raises(FileNotFoundError, match="models/vae"):
        split_models.component_files(split_models.split_model_name("z.safetensors", "te.safetensors", "missing.safetensors"))
    with pytest.raises(FileNotFoundError):  # one kind's file name never reads another kind's folder
        split_models.component_files(split_models.split_model_name("te.safetensors", "te.safetensors", "ae.safetensors"))


def test_a_comfyui_text_encoder_file_loads_into_the_transformers_model(tmp_path):
    """ComfyUI's text encoder files keep the causal LM's "model." prefix; the loaded encoder
    must compute exactly what the original model does (rotary buffers rebuilt off meta)."""
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    (tmp_path / "configs" / "text_encoder").mkdir(parents=True)
    config.save_pretrained(tmp_path / "configs" / "text_encoder")
    torch.manual_seed(0)
    reference = transformers.Qwen3Model(config).eval()
    path = _write(tmp_path / "te.safetensors", {f"model.{key}": value for key, value in reference.state_dict().items()})
    family = split_models.SPLIT_FAMILIES["z-image-turbo"]
    loaded = split_models.load_text_encoder(path, family, tmp_path / "configs", torch.float32)
    ids = torch.tensor([[1, 5, 9, 3]])
    with torch.no_grad():
        assert torch.allclose(loaded(ids).last_hidden_state, reference(ids).last_hidden_state, atol=1e-6)
    wrong = _write(tmp_path / "wrong.safetensors", {"model.embed_tokens.weight": torch.zeros(64, 32)})
    with pytest.raises(ValueError, match="doesn't fit"):
        split_models.load_text_encoder(wrong, family, tmp_path / "configs", torch.float32)


def test_z_image_loads_in_bf16_by_default():
    assert models.resolve_dtype_name("auto", models.ARCH_ZIMAGE, "cuda", True) == "bf16"
    assert models.arch_from_pipeline_class("ZImagePipeline") == models.ARCH_ZIMAGE


def test_a_flow_matching_model_keeps_its_scheduler():
    diffusers = pytest.importorskip("diffusers")
    base = diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0)

    class _Pipe:
        scheduler = base

    scheduler = generation.make_scheduler(_Pipe, "dpm_multistep", "normal", dict(base.config), "FlowMatchEulerDiscreteScheduler")
    assert type(scheduler).__name__ == "FlowMatchEulerDiscreteScheduler" and scheduler.config.shift == 3.0
    assert type(generation.make_scheduler(_Pipe, "dpm_multistep", "normal", dict(base.config))).__name__ == "DPMSolverMultistepScheduler"


def test_the_node_loads_through_the_pipeline_cache(monkeypatch):
    asked = []

    def fake_get(name, dtype, on_progress=None):
        asked.append((name, dtype))
        return models.CacheEntry(pipe="PIPE", nbytes=1, placement="gpu", arch=models.ARCH_ZIMAGE, dtype="bf16")

    monkeypatch.setattr(models.PIPELINES, "get", fake_get)
    (model,) = io_nodes.LoadSplitModel().load("z.safetensors", "qwen.safetensors", "ae.safetensors")
    assert asked == [("split:z.safetensors|qwen.safetensors|ae.safetensors", "auto")]
    assert model["pipeline"] == "PIPE" and model["arch"] == models.ARCH_ZIMAGE and model["name"] == "z.safetensors"
    assert io_nodes.LoadSplitModel.VALIDATE_INPUTS(diffusion_model=" ") != True  # noqa: E712


def test_the_cache_dispatches_split_names(monkeypatch):
    seen = {}
    monkeypatch.setattr(split_models, "split_arch", lambda name: models.ARCH_ZIMAGE)

    class _Pipe:
        scheduler = None

        def set_progress_bar_config(self, **_):
            pass

    def fake_load(name, dtype, on_progress=None):
        seen["dtype"] = dtype
        return _Pipe(), "ZImagePipeline"

    monkeypatch.setattr(split_models, "load_split_pipeline", fake_load)
    monkeypatch.setattr(models, "estimate_pipeline_bytes", lambda pipe: 5)
    # "auto" picks bf16 only where the GPU supports it; CI's CPU torch says no.
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *args, **kwargs: True)
    entry = models._load_entry("split:a|b|c", "cuda", "auto")
    assert entry.arch == models.ARCH_ZIMAGE and entry.dtype == "bf16" and seen["dtype"] == torch.bfloat16


_REAL_DIR = Path(os.environ.get("PIXELFLOW_SPLIT_MODEL_DIR", "D:/models"))


@pytest.mark.skipif(os.environ.get("PIXELFLOW_SPLIT_MODEL_TESTS") != "1", reason="set PIXELFLOW_SPLIT_MODEL_TESTS=1 (loads ~20 GB)")
def test_z_image_turbo_generates_from_the_real_split_files(monkeypatch, tmp_path):
    files = {"diffusion_models": "z_image_turbo_bf16.safetensors", "text_encoders": "qwen_3_4b.safetensors", "vae": "ae.safetensors"}
    for kind, file_name in files.items():
        if not (_REAL_DIR / kind / file_name).is_file():
            pytest.skip(f"{_REAL_DIR / kind / file_name} not found")
    monkeypatch.setattr(models, "local_model_path", lambda kind, name: _REAL_DIR / kind / name)
    monkeypatch.delenv("PIXELFLOW_TEST_MODE", raising=False)
    cache = models.PipelineCache()
    entry = cache.get(split_models.split_model_name(*files.values()))
    assert entry.arch == models.ARCH_ZIMAGE
    model = {"name": files["diffusion_models"], "pipeline": entry.pipe, "arch": entry.arch, "dtype": entry.dtype, "loras": []}
    from PIL import Image

    out = generation.KSampler().sample(model, {"text": "a red apple on a wooden table, photograph"}, Image.new("RGB", (512, 512)),
                                       seed=1, steps=9, cfg=0)
    image = out["result"][0][0]
    assert image.size == (512, 512)
    image.save(tmp_path / "z_image.png")
    import numpy as np

    pixels = np.asarray(image, dtype=np.float32)
    assert pixels.std() > 20  # a picture, not a flat or noise-free blank
    red = pixels[..., 0] - pixels[..., 1:].mean(axis=-1)
    assert (red > 40).mean() > 0.02  # the prompt's red apple is there
    json.dumps(models.pipeline_meta(entry.pipe), default=str)
