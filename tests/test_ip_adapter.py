"""IP-Adapter (B20): the node, the CLIP vision encoder read from a ComfyUI clip_vision file, and
how a run makes exactly its MODEL's adapter active on the shared pipeline. The real SDXL check is
opt-in (PIXELFLOW_IP_ADAPTER_TESTS=1, files under D:/models or PIXELFLOW_IP_ADAPTER_DIR)."""

import os
from pathlib import Path

import pytest
from PIL import Image

from backend.nodes import generation, ip_adapter, models

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _tiny_vision(hidden=1280, layers=2, patch=14, image=28):
    config = transformers.CLIPVisionConfig(hidden_size=hidden, intermediate_size=64, num_hidden_layers=layers,
                                           num_attention_heads=16, patch_size=patch, image_size=image, projection_dim=32,
                                           hidden_act="gelu", layer_norm_eps=1e-5)
    torch.manual_seed(0)
    return transformers.CLIPVisionModelWithProjection(config).eval()


def test_the_vision_config_comes_from_the_tensor_shapes():
    model = _tiny_vision()
    config = ip_adapter.vision_config_from_state(model.state_dict())
    assert config["hidden_size"] == 1280 and config["num_hidden_layers"] == 2 and config["intermediate_size"] == 64
    assert config["patch_size"] == 14 and config["image_size"] == 28 and config["projection_dim"] == 32
    assert config["hidden_act"] == "gelu" and config["num_attention_heads"] == 16
    with pytest.raises(ValueError, match="transformers' layout"):
        ip_adapter.vision_config_from_state({"visual.proj": torch.zeros(1)})


def test_a_full_clip_file_loads_its_vision_half(tmp_path):
    """ComfyUI's clip_vision files can hold the whole CLIPModel; the text half is ignored and
    the loaded encoder computes what the original does."""
    model = _tiny_vision()
    state = {**model.state_dict(), "text_model.embeddings.token_embedding.weight": torch.zeros(4, 4), "logit_scale": torch.ones(()),
             "vision_model.embeddings.position_ids": torch.arange(5).unsqueeze(0)}  # stored by older checkpoints
    path = tmp_path / "clip.safetensors"
    safetensors_torch.save_file({key: value.contiguous() for key, value in state.items()}, str(path))
    loaded = ip_adapter.load_image_encoder(str(path), "stamp", "fp32")
    pixels = torch.rand(1, 3, 28, 28)
    with torch.no_grad():
        assert torch.allclose(loaded(pixels).image_embeds, model(pixels).image_embeds, atol=1e-5)


class _FakeUnet:
    dtype = torch.float16
    device = torch.device("cpu")


class _FakePipe:
    def __init__(self):
        self.unet = _FakeUnet()
        self.calls = []
        self.image_encoder = None

    def load_ip_adapter(self, folder, subfolder, weight_name, image_encoder_folder):
        self.calls.append(("load", weight_name, image_encoder_folder))

    def unload_ip_adapter(self):
        self.calls.append(("unload",))

    def set_ip_adapter_scale(self, scale):
        self.calls.append(("scale", scale))

    def register_modules(self, **modules):
        for name, module in modules.items():
            setattr(self, name, module)


@pytest.fixture
def adapter_files(isolated_data_dirs, monkeypatch):
    for kind, name in (("ipadapter", "plus.safetensors"), ("ipadapter", "other.safetensors"), ("clip_vision", "vit-h.safetensors")):
        isolated_data_dirs[kind].mkdir(parents=True, exist_ok=True)
        (isolated_data_dirs[kind] / name).write_bytes(b"x")

    class _Encoder:
        def to(self, device):
            return self

    monkeypatch.setattr(ip_adapter, "load_image_encoder", lambda *args: _Encoder())


def test_a_run_loads_its_adapter_once_switches_and_unloads(adapter_files):
    pipe = _FakePipe()
    image = Image.new("RGB", (8, 8))
    adapter = {"ipadapter": "plus.safetensors", "clip_vision": "vit-h.safetensors", "image": image, "weight": 0.6}
    assert ip_adapter.activate(pipe, adapter) == {"ip_adapter_image": image}
    assert pipe.calls == [("load", "plus.safetensors", None), ("scale", 0.6)]
    assert pipe.image_encoder is not None and pipe.feature_extractor is not None
    ip_adapter.activate(pipe, {**adapter, "weight": 0.9})  # same adapter: only the scale changes
    assert pipe.calls[-1] == ("scale", 0.9) and sum(call[0] == "load" for call in pipe.calls) == 1
    ip_adapter.activate(pipe, {**adapter, "ipadapter": "other.safetensors"})  # another one: swap
    assert pipe.calls[-3:] == [("unload",), ("load", "other.safetensors", None), ("scale", 0.6)]
    assert ip_adapter.activate(pipe, None) == {}  # a run without one: unloaded
    assert pipe.calls[-1] == ("unload",)
    assert ip_adapter.activate(pipe, None) == {} and pipe.calls[-1] == ("unload",) and len(pipe.calls) == 7


def test_the_node_adds_the_adapter_to_the_model(adapter_files):
    model = {"name": "sdxl", "pipeline": "PIPE", "loras": []}
    (out,) = ip_adapter.ApplyIPAdapter().apply(model, [Image.new("RGBA", (4, 4))], "plus.safetensors", "vit-h.safetensors", 0.5)
    assert out["ip_adapter"]["weight"] == 0.5 and out["ip_adapter"]["image"].mode == "RGB"
    assert "ip_adapter" not in model  # the input MODEL is unchanged
    with pytest.raises(FileNotFoundError, match="models/ipadapter"):
        ip_adapter.ApplyIPAdapter().apply(model, Image.new("RGB", (4, 4)), "missing.safetensors", "vit-h.safetensors")


def test_an_adapter_needs_a_unet():
    class _Transformer:
        load_ip_adapter = None

    with pytest.raises(RuntimeError, match="no UNet"):
        ip_adapter._unet_owner(_Transformer())


_REAL = Path(os.environ.get("PIXELFLOW_IP_ADAPTER_DIR", "D:/models"))


@pytest.mark.skipif(os.environ.get("PIXELFLOW_IP_ADAPTER_TESTS") != "1", reason="set PIXELFLOW_IP_ADAPTER_TESTS=1 (SDXL, ~8 GB)")
def test_the_reference_image_steers_a_real_sdxl_run(monkeypatch, tmp_path):
    import numpy as np

    files = {"checkpoints": "SDXL/sd_xl_base_1.0.safetensors", "ipadapter": "ip-adapter-plus_sdxl_vit-h.safetensors",
             "clip_vision": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"}
    for kind, name in files.items():
        if not (_REAL / kind / name).is_file():
            pytest.skip(f"{_REAL / kind / name} not found")
    monkeypatch.setattr(models, "local_model_path", lambda kind, name: _REAL / kind / name)
    monkeypatch.delenv("PIXELFLOW_TEST_MODE", raising=False)
    entry = models.PipelineCache().get(files["checkpoints"])
    model = {"name": files["checkpoints"], "pipeline": entry.pipe, "arch": entry.arch, "dtype": entry.dtype, "loras": []}
    # The reference: saturated blue and orange stripes. The prompt says nothing about colour.
    stripes = np.zeros((512, 512, 3), np.uint8)
    stripes[:, :] = (20, 70, 230)
    for x in range(0, 512, 128):
        stripes[:, x:x + 64] = (250, 140, 20)
    reference = Image.fromarray(stripes)
    (with_image,) = ip_adapter.ApplyIPAdapter().apply(model, reference, files["ipadapter"], files["clip_vision"], 0.9)
    prompt = {"text": "an abstract painting"}

    def run(m):
        image = generation.KSampler().sample(m, prompt, Image.new("RGB", (768, 768)), seed=3, steps=20, cfg=6.0)["result"][0][0]
        return image, np.asarray(image.convert("HSV"), np.float32)

    steered, steered_hsv = run(with_image)
    plain, plain_hsv = run(model)  # also checks a run without the adapter works after one with it
    steered.save(tmp_path / "steered.png")
    plain.save(tmp_path / "plain.png")

    def share_like_reference(hsv):
        hue, saturation = hsv[..., 0] * 360 / 255, hsv[..., 1]
        blue = (hue > 200) & (hue < 250)
        orange = (hue > 15) & (hue < 45)
        return ((blue | orange) & (saturation > 90)).mean()

    assert share_like_reference(steered_hsv) > share_like_reference(plain_hsv) + 0.15
