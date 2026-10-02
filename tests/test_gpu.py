"""CUDA-only checks (report item T4), marked `gpu`: skipped without a CUDA
device (CI), run locally on the RTX 5080 before a release:

    .venv\\Scripts\\python.exe -m pytest -q -m gpu

The tiny test checkpoint covers the CUDA code paths (fp16/bf16 loading,
VAE slicing, sampling, previews). The real-model smoke test downloads
stabilityai/sd-turbo (~2.5 GB) and only runs with PIXELFLOW_GPU_SMOKE=1."""

import os

import pytest
import torch

from backend.engine.graph import execute_prompt, validate_prompt
from backend.nodes import models

from _hub import skip_if_hub_unreachable

pytestmark = pytest.mark.gpu

TINY = "hf-internal-testing/tiny-stable-diffusion-torch"


def _entry(name, dtype="auto"):
    try:
        return models.PIPELINES.get(name, dtype)
    except Exception as exc:
        skip_if_hub_unreachable(exc)
        raise


def test_cuda_build_and_device():
    assert torch.version.cuda, "the GPU suite needs a CUDA build of torch"
    assert torch.cuda.is_available()
    major, minor = torch.cuda.get_device_capability(0)
    assert (major, minor) >= (7, 0)


def test_auto_precision_is_fp16_on_the_gpu():
    entry = _entry(TINY)
    assert entry.placement == "gpu"
    assert entry.pipe.unet.dtype == torch.float16 and entry.pipe.unet.device.type == "cuda"
    assert entry.pipe.vae.use_slicing


def test_bf16_option_loads_in_bf16():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("GPU without bf16")
    entry = _entry(TINY, "bf16")
    assert entry.pipe.unet.dtype == torch.bfloat16
    assert ("hf-internal-testing/tiny-stable-diffusion-torch", "cuda", "bf16") in models.PIPELINES.entries


def _graph(ckpt, steps=2, size=64, cfg=7.0):
    return {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": ckpt}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a (red:1.2) lighthouse at dusk"}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": size, "height": size, "color": "#000"}},
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["clip", 0],
                "latent_image": ["latent", 0],
                "sampler": "euler",
                "seed": 5,
                "steps": steps,
                "cfg": cfg,
                "batch_size": 2,
            },
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "gpu"}},
    }


def test_sampling_on_cuda_with_previews(output_dir):
    events = []
    try:
        outputs = execute_prompt(validate_prompt(_graph(TINY)), on_progress_detail=lambda *a: events.append(a))
    except Exception as exc:
        skip_if_hub_unreachable(exc)
        raise
    assert len(outputs["save"]["images"]) == 2
    assert any("preview" in info for *_, info in events)


@pytest.mark.skipif(os.environ.get("PIXELFLOW_GPU_SMOKE") != "1", reason="set PIXELFLOW_GPU_SMOKE=1 (downloads sd-turbo)")
def test_real_sd_turbo_512(output_dir):
    outputs = execute_prompt(validate_prompt(_graph(models.DEFAULT_CHECKPOINT, steps=4, size=512, cfg=0.0)))
    from PIL import Image

    with Image.open(output_dir / outputs["save"]["images"][0]) as img:
        assert img.size == (512, 512)
        assert max(img.convert("L").getextrema()) > 30, "not a black (NaN) image"


def test_unet_and_vae_use_channels_last_on_the_gpu():
    # Report item P3: channels_last memory format for the convolutions.
    import torch as _torch

    entry = _entry(TINY)
    weight = next(p for p in entry.pipe.unet.parameters() if p.ndim == 4)
    assert weight.is_contiguous(memory_format=_torch.channels_last)
