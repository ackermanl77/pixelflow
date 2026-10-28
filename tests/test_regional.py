"""Regional prompting (F19): the Regional Prompt node, the region weights, and the UNet blending --
with a stand-in UNet; the real SDXL run is opt-in (PIXELFLOW_REGIONAL_TESTS=1, a checkpoint under
D:/models/checkpoints or PIXELFLOW_REGIONAL_CHECKPOINT)."""

import os
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.nodes import generation, models, regional

torch = pytest.importorskip("torch")


def test_the_node_adds_regions_from_a_box_or_a_mask():
    base = {"text": "a landscape", "clip_skip": 0}
    (one,) = regional.RegionalPrompt().apply(base, "a red barn", x=0.0, y=0.5, width=0.4, height=0.5)
    (two,) = regional.RegionalPrompt().apply(one, "a lake", mask=Image.new("L", (8, 8), 255))
    assert [r["text"] for r in two["regions"]] == ["a red barn", "a lake"]
    assert two["regions"][0]["box"] == (0.0, 0.5, 0.4, 0.5) and two["regions"][1]["mask"].mode == "L"
    assert "regions" not in base and two["text"] == "a landscape"  # the main prompt is unchanged
    with pytest.raises(ValueError, match="empty"):
        regional.RegionalPrompt().apply(base, "x", width=0.0)


def test_region_masks_and_overlaps():
    box = regional.region_mask({"box": (0.0, 0.0, 0.5, 1.0), "feather": 0.0, "strength": 1.0}, 100, 40)
    assert box.shape == (40, 100) and box[:, :50].min() == 1.0 and box[:, 50:].max() == 0.0
    soft = regional.region_mask({"box": (0.0, 0.0, 0.5, 1.0), "feather": 0.2, "strength": 0.5}, 100, 40)
    assert 0.0 < soft[20, 50] < 0.5 and soft[20, 10] == pytest.approx(0.5, abs=0.01)
    weights = regional.blend_weights([np.full((2, 2), 0.8), np.full((2, 2), 0.6), np.zeros((2, 2))])
    assert np.allclose(sum(weights), 1.0)  # overlapping regions share, never above 1
    assert np.allclose(weights[0] / weights[1], 0.8 / 0.6)


class _FakeUNet:
    """A UNet stand-in: its 'noise prediction' is the mean of the prompt embeddings, per row, so
    which prompt reached which row (and where) can be read off the output."""

    config = types.SimpleNamespace(time_cond_proj_dim=None)

    def __init__(self):
        self.calls = []

    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, return_dict=True, **kwargs):
        self.calls.append((sample.shape[0], encoder_hidden_states.shape[1], kwargs.get("extra")))
        value = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)
        out = value.expand(-1, *sample.shape[1:]).clone()
        return (out,) if not return_dict else types.SimpleNamespace(sample=out)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def test_the_unet_blends_regions_into_the_positive_half_only():
    unet = _FakeUNet()
    pipe = types.SimpleNamespace(unet=unet)
    region = {"embeds": torch.full((1, 5, 4), 10.0), "pooled": None,
              "mask": regional.region_mask({"box": (0.0, 0.0, 0.5, 1.0), "feather": 0.0}, 64, 64)}
    sample = torch.zeros(4, 4, 8, 8)  # CFG: [negative x2; positive x2]
    prompts = torch.cat([torch.zeros(2, 7, 4), torch.ones(2, 7, 4)])
    extra = torch.arange(4.0)  # a batched extra input (e.g. ControlNet residuals) is split too
    with regional.regional_unet(pipe, [region], cfg=True):
        out = unet(sample, 1, prompts, return_dict=False, extra=extra)[0]
    assert "forward" not in vars(unet)  # restored
    assert unet.calls[1][0] == 2 and unet.calls[1][1] == 5  # the region pass: positive rows, its own prompt length
    assert torch.equal(unet.calls[1][2], extra[2:])
    assert torch.all(out[:2] == 0.0)  # the negative half is the plain prediction
    assert torch.allclose(out[2:, :, :, :4], torch.tensor(10.0))  # left half: the region's prompt
    assert torch.allclose(out[2:, :, :, 4:], torch.tensor(1.0))  # right half: the main prompt
    with regional.regional_unet(pipe, [region], cfg=False):
        no_cfg = unet(torch.zeros(1, 4, 8, 8), 1, torch.ones(1, 7, 4), return_dict=False)[0]
    assert torch.allclose(no_cfg[:, :, :, :4], torch.tensor(10.0))


def test_transformer_models_are_refused_clearly():
    pipe = types.SimpleNamespace(scheduler=None)
    model = {"pipeline": pipe, "arch": models.ARCH_FLUX}
    with pytest.raises(RuntimeError, match="SD 1.x/2.x and SDXL"):
        generation._prepared_regions(model, pipe, ({"text": "a", "regions": [{"text": "b", "box": (0, 0, 1, 1)}]}, None), 64, 64)
    assert generation._prepared_regions(model, pipe, ({"text": "a"}, None), 64, 64) == []


_CHECKPOINT = Path(os.environ.get("PIXELFLOW_REGIONAL_CHECKPOINT", "D:/models/checkpoints/SDXL/sd_xl_base_1.0.safetensors"))


@pytest.mark.skipif(os.environ.get("PIXELFLOW_REGIONAL_TESTS") != "1", reason="set PIXELFLOW_REGIONAL_TESTS=1 (SDXL, ~7 GB)")
def test_left_and_right_prompts_in_a_real_sdxl_run(monkeypatch, tmp_path):
    if not _CHECKPOINT.is_file():
        pytest.skip(f"{_CHECKPOINT} not found")
    monkeypatch.setattr(models, "local_model_path", lambda kind, name: _CHECKPOINT)
    monkeypatch.delenv("PIXELFLOW_TEST_MODE", raising=False)
    entry = models.PipelineCache().get(_CHECKPOINT.name)
    model = {"name": _CHECKPOINT.name, "pipeline": entry.pipe, "arch": entry.arch, "dtype": entry.dtype, "loras": []}
    positive = {"text": "a painted wall, flat colour, photograph", "clip_skip": 0}
    (positive,) = regional.RegionalPrompt().apply(positive, "a bright red painted wall", x=0.0, y=0.0, width=0.5, height=1.0)
    (positive,) = regional.RegionalPrompt().apply(positive, "a bright blue painted wall", x=0.5, y=0.0, width=0.5, height=1.0)
    image = generation.KSampler().sample(model, positive, Image.new("RGB", (1024, 768)), seed=7, steps=25, cfg=7.0)["result"][0][0]
    image.save(tmp_path / "regional.png")
    pixels = np.asarray(image, dtype=np.float32)
    left, right = pixels[:, :512].reshape(-1, 3).mean(axis=0), pixels[:, 512:].reshape(-1, 3).mean(axis=0)
    redness_left, redness_right = left[0] - left[2], right[0] - right[2]
    assert redness_left > 40  # the left half follows "a bright red painted wall"
    assert redness_right < 0  # the right half leans blue ("a bright blue painted wall")
    assert redness_left - redness_right > 50  # and the two halves clearly differ
