"""Uploads feeding Load Image / Load Mask (F2 backend side), and img2img and
inpainting via from_pipe (B5), plus VAE slicing/tiling (GPU item P1)."""

import io
import os

import pytest
import torch
from diffusers import EulerDiscreteScheduler
from fastapi.testclient import TestClient
from PIL import Image

from backend import main
from backend.engine.graph import PromptValidationError, execute_prompt, validate_prompt
from backend.nodes import generation, io_nodes

from _hub import skip_if_hub_unreachable

TINY_SD_CHECKPOINT = "hf-internal-testing/tiny-stable-diffusion-torch"
BASE = "http://127.0.0.1:8288"


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---- uploads -> LoadImage / LoadMask ----


def test_uploaded_file_shows_up_in_the_load_image_combo(upload_dir):
    client = TestClient(main.app, base_url=BASE)
    res = client.post("/api/upload", files={"file": ("pic.png", _png_bytes(Image.new("RGB", (8, 8))), "image/png")})
    assert res.status_code == 200
    assert client.get("/api/uploads").json() == ["pic.png"]
    spec = client.get("/api/nodes").json()["LoadImage"]["input_types"]["required"]["filename"]
    assert spec == [["pic.png"], {"image_upload": True}]
    assert client.get("/api/nodes").json()["LoadImage"]["return_types"] == ["IMAGE", "MASK"]
    served = client.get("/uploads/pic.png")
    assert served.status_code == 200 and served.content.startswith(b"\x89PNG")
    assert client.get("/uploads/missing.png").status_code == 404
    assert client.get("/uploads/..%2Fsecret").status_code == 404


def test_load_image_rejects_a_filename_that_is_not_uploaded(upload_dir):
    (upload_dir / "real.png").write_bytes(_png_bytes(Image.new("RGB", (4, 4))))
    graph = {
        "load": {"class_type": "LoadImage", "inputs": {"filename": "fake.png"}},
        "out": {"class_type": "PreviewImage", "inputs": {"image": ["load", 0]}},
    }
    with pytest.raises(PromptValidationError, match="must be one of") as exc_info:
        validate_prompt(graph)
    assert exc_info.value.input_name == "filename"


def test_load_image_mask_comes_from_alpha_and_exif_is_applied(upload_dir):
    rgba = Image.new("RGBA", (6, 4), (255, 0, 0, 255))
    rgba.putpixel((0, 0), (0, 0, 0, 0))  # transparent -> repaint
    (upload_dir / "alpha.png").write_bytes(_png_bytes(rgba))
    image, mask = io_nodes.LoadImage().load("alpha.png")
    assert image.mode == "RGB" and mask.mode == "L"
    assert mask.getpixel((0, 0)) == 255 and mask.getpixel((1, 1)) == 0

    rotated = Image.new("RGB", (6, 4))
    exif = rotated.getexif()
    exif[0x0112] = 6  # "rotate 90 CW to display"
    buf = io.BytesIO()
    rotated.save(buf, format="JPEG", exif=exif)
    (upload_dir / "phone.jpg").write_bytes(buf.getvalue())
    image, mask = io_nodes.LoadImage().load("phone.jpg")
    assert image.size == (4, 6)
    assert mask.getextrema() == (0, 0), "no alpha -> nothing masked"


def test_load_mask_luminance_and_invert(upload_dir):
    src = Image.new("L", (4, 4), 0)
    src.putpixel((1, 1), 255)
    (upload_dir / "m.png").write_bytes(_png_bytes(src))
    (mask,) = io_nodes.LoadMask().load("m.png")
    assert mask.getpixel((1, 1)) == 255 and mask.getpixel((0, 0)) == 0
    (inverted,) = io_nodes.LoadMask().load("m.png", invert=True)
    assert inverted.getpixel((1, 1)) == 0


def test_replacing_an_upload_invalidates_the_cache(upload_dir):
    path = upload_dir / "a.png"
    path.write_bytes(_png_bytes(Image.new("RGB", (4, 4))))
    first = io_nodes.LoadImage.IS_CHANGED(filename="a.png")
    path.write_bytes(_png_bytes(Image.new("RGB", (64, 64), (1, 2, 3))))
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 5_000_000_000))
    assert io_nodes.LoadImage.IS_CHANGED(filename="a.png") != first
    missing = io_nodes.LoadImage.IS_CHANGED(filename="gone.png")
    assert missing != missing  # NaN -> always re-run (and report the real error)


# ---- img2img / inpainting with the tiny real checkpoint ----


def _sd_graph(upload_name, sampler_class, extra_inputs):
    return {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY_SD_CHECKPOINT}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red circle"}},
        "load": {"class_type": "LoadImage", "inputs": {"filename": upload_name}},
        "sampler": {
            "class_type": sampler_class,
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["clip", 0],
                "image": ["load", 0],
                "sampler": "euler",
                "seed": 3,
                "steps": 4,
                "cfg": 7.0,
                "batch_size": 2,
                **extra_inputs,
            },
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "i2i"}},
    }


def _run(graph, **kwargs):
    try:
        return execute_prompt(validate_prompt(graph), **kwargs)
    except Exception as exc:
        skip_if_hub_unreachable(exc)
        raise


def test_img2img_samples_from_an_uploaded_image(upload_dir, output_dir):
    (upload_dir / "src.png").write_bytes(_png_bytes(Image.new("RGB", (40, 36), (200, 30, 30))))
    progress = []
    outputs = _run(
        _sd_graph("src.png", "KSamplerImage", {"denoise": 0.5}),
        on_progress=lambda node_id, step, total: progress.append((step, total)),
    )
    names = outputs["save"]["images"]
    assert len(names) == 2
    with Image.open(output_dir / names[0]) as img:
        assert img.size == (40, 32), "snapped to multiples of 8"
    assert outputs["sampler"]["seed"] == 3
    assert progress and all(total == 2 for _, total in progress), "only steps x denoise steps run"


def test_inpaint_uses_the_mask_from_load_image(upload_dir, output_dir):
    rgba = Image.new("RGBA", (32, 32), (0, 0, 255, 255))
    for x in range(8, 24):
        for y in range(8, 24):
            rgba.putpixel((x, y), (0, 0, 0, 0))
    (upload_dir / "hole.png").write_bytes(_png_bytes(rgba))
    graph = _sd_graph("hole.png", "KSamplerInpaint", {"mask": ["load", 1], "denoise": 1.0})
    outputs = _run(graph)
    assert len(outputs["save"]["images"]) == 2


def test_derived_pipelines_share_the_base_weights_and_are_reused():
    try:
        pipe = io_nodes._load_pipeline(TINY_SD_CHECKPOINT)
    except Exception as exc:
        skip_if_hub_unreachable(exc)
        raise
    img2img = generation.derived_pipeline(pipe, "img2img")
    assert img2img.unet is pipe.unet and img2img.vae is pipe.vae
    assert generation.derived_pipeline(pipe, "img2img") is img2img
    assert generation.derived_pipeline(pipe, "inpaint") is not img2img
    assert pipe.vae.use_slicing, "VAE slicing is always enabled on load"


def test_denoise_too_small_for_the_step_count_is_a_clear_error():
    with pytest.raises(ValueError, match="leaves no sampling steps"):
        generation._denoise_steps(4, 0.1)
    assert generation._denoise_steps(20, 0.5) == 10


# ---- VAE tiling and the out-of-memory retry ----


class _FakeVae:
    def __init__(self):
        self.tiling = False
        self.history = []

    def enable_tiling(self):
        self.tiling = True

    def disable_tiling(self):
        self.tiling = False


class _FakePipe:
    device = torch.device("cpu")

    def __init__(self, oom_without_tiling=False, always_oom=False):
        self.scheduler = EulerDiscreteScheduler()
        self.vae = _FakeVae()
        self.oom_without_tiling = oom_without_tiling
        self.always_oom = always_oom
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(self.vae.tiling)
        if self.always_oom or (self.oom_without_tiling and not self.vae.tiling):
            raise io_nodes.OutOfMemoryError("CUDA out of memory")

        class _Result:
            images = [Image.new("RGB", (kwargs["width"], kwargs["height"]))]

        return _Result()


def _sample(pipe, size):
    return generation.KSamplerStub().sample(
        model={"pipeline": pipe},
        positive={"text": "x"},
        latent_image=Image.new("RGB", size),
        seed=1,
        steps=1,
        cfg=0.0,
    )


def test_small_images_decode_untiled_and_large_ones_tiled(monkeypatch):
    monkeypatch.setattr(generation, "free_memory", lambda: None)
    pipe = _FakePipe()
    _sample(pipe, (512, 512))
    _sample(pipe, (2048, 2048))
    _sample(pipe, (512, 512))
    assert pipe.calls == [False, True, False]


def test_oom_is_retried_once_with_tiling(monkeypatch):
    freed = []
    monkeypatch.setattr(generation, "free_memory", lambda: freed.append(True))
    pipe = _FakePipe(oom_without_tiling=True)
    result = _sample(pipe, (512, 512))
    assert pipe.calls == [False, True]
    assert result["seed"] == 1 and freed


def test_oom_even_with_tiling_gives_the_clear_error(monkeypatch):
    monkeypatch.setattr(generation, "free_memory", lambda: None)
    pipe = _FakePipe(always_oom=True)
    with pytest.raises(RuntimeError, match="Out of GPU memory while sampling 512x512"):
        _sample(pipe, (512, 512))
    assert pipe.calls == [False, True], "exactly one retry"
