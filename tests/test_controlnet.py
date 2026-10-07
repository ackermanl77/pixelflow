"""ControlNet (B7): Load / Apply ControlNet on every sampler with the tiny real
checkpoint and a ControlNet built from its own UNet (no extra download), plus
the Canny Edges preprocessor."""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from backend.engine.graph import execute_prompt, validate_prompt
from backend.nodes import NODE_REGISTRY, controlnet
from backend.nodes.image_ops import canny

from _hub import skip_if_hub_unreachable

TINY_SD_CHECKPOINT = "hf-internal-testing/tiny-stable-diffusion-torch"


def _png(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def tiny_controlnet(isolated_data_dirs):
    """models/controlnet/tiny-cn: a ControlNet shaped for the tiny checkpoint's UNet."""
    from diffusers import ControlNetModel

    from diffusers import StableDiffusionPipeline

    controlnet.clear_cache()
    try:
        tiny = StableDiffusionPipeline.from_pretrained(TINY_SD_CHECKPOINT)
    except Exception as exc:  # noqa: BLE001
        skip_if_hub_unreachable(exc)
        raise
    # The tiny VAE downsamples by 2 (real SD: 8), so the control embedding needs one stride-2 stage.
    net = ControlNetModel.from_unet(tiny.unet, conditioning_embedding_out_channels=(16, 32))
    # A fresh ControlNet's "zero convolutions" output exactly zero (no effect until trained); give
    # them weights so the strength test below can see the control.
    import torch

    torch.manual_seed(0)
    with torch.no_grad():
        for block in [*net.controlnet_down_blocks, net.controlnet_mid_block]:
            for param in block.parameters():
                param.normal_(0.0, 0.5)
    target = isolated_data_dirs["controlnet"] / "tiny-cn"
    net.save_pretrained(target)
    yield "tiny-cn"
    controlnet.clear_cache()


def _graph(sampler_class, sampler_inputs, upload=None, strength=1.0):
    graph = {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY_SD_CHECKPOINT}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a square"}},
        "control_src": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#ffffff"}},
        "edges": {"class_type": "CannyEdges", "inputs": {"image": ["control_src", 0], "low": 100, "high": 200}},
        "cn": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "tiny-cn"}},
        "apply": {"class_type": "ControlNetApply", "inputs": {"conditioning": ["clip", 0], "control_net": ["cn", 0],
                                                             "image": ["edges", 0], "strength": strength, "start": 0.0, "end": 1.0}},
        "sampler": {"class_type": sampler_class, "inputs": {"model": ["ckpt", 0], "positive": ["apply", 0], "sampler": "euler",
                                                             "seed": 5, "steps": 2, "cfg": 7.0, "batch_size": 1, **sampler_inputs}},
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "cn"}},
    }
    if upload:
        graph["load"] = {"class_type": "LoadImage", "inputs": {"filename": upload}}
    return graph


def _run(graph):
    try:
        return execute_prompt(validate_prompt(graph))
    except Exception as exc:  # noqa: BLE001
        skip_if_hub_unreachable(exc)
        raise


def test_controlnet_guides_txt2img_and_its_strength_matters(tiny_controlnet, output_dir):
    empty = {"latent_image": ["control_src", 0]}
    full = _run(_graph("KSampler", empty))
    none = _run(_graph("KSampler", empty, strength=0.0))
    with Image.open(output_dir / full["save"]["images"][0]) as a, Image.open(output_dir / none["save"]["images"][0]) as b:
        assert a.size == (64, 64)
        assert np.abs(np.asarray(a, dtype=int) - np.asarray(b, dtype=int)).max() > 0  # the control changes the result


def test_controlnet_works_with_img2img_and_inpaint(tiny_controlnet, upload_dir, output_dir):
    (upload_dir / "src.png").write_bytes(_png(Image.new("RGB", (32, 32), (200, 30, 30))))
    (upload_dir / "mask.png").write_bytes(_png(Image.new("L", (32, 32), 255)))
    img2img = _run(_graph("KSamplerImage", {"image": ["load", 0], "denoise": 1.0}, upload="src.png"))
    assert len(img2img["save"]["images"]) == 1
    graph = _graph("KSamplerInpaint", {"image": ["load", 0], "mask": ["mask", 0], "denoise": 1.0}, upload="src.png")
    graph["mask"] = {"class_type": "LoadMask", "inputs": {"filename": "mask.png", "channel": "luminance", "invert": False}}
    assert len(_run(graph)["save"]["images"]) == 1


def test_the_loader_lists_local_models_and_reports_missing_ones(isolated_data_dirs):
    folder = isolated_data_dirs["controlnet"]
    (folder / "nested" / "diffusers-cn").mkdir(parents=True)
    (folder / "nested" / "diffusers-cn" / "config.json").write_text("{}")
    (folder / "canny.safetensors").write_bytes(b"x")
    (folder / "notes.txt").write_text("ignored")
    assert controlnet.list_controlnets() == ["canny.safetensors", "nested/diffusers-cn"]
    names = NODE_REGISTRY["ControlNetLoader"].INPUT_TYPES()["required"]["control_net_name"][0]
    assert names == ["canny.safetensors", "nested/diffusers-cn"]
    with pytest.raises(ValueError):
        NODE_REGISTRY["ControlNetLoader"]().load("")


def test_apply_controlnet_chains_and_checks_its_window():
    net = {"name": "x", "model": object()}
    image = Image.new("RGB", (8, 8))
    (first,) = NODE_REGISTRY["ControlNetApply"]().apply({"text": "t"}, net, [image], 0.8)
    (second,) = NODE_REGISTRY["ControlNetApply"]().apply(first, net, image, 0.5, 0.1, 0.9)
    assert [c["strength"] for c in second["controls"]] == [0.8, 0.5] and second["text"] == "t"
    with pytest.raises(ValueError):
        NODE_REGISTRY["ControlNetApply"]().apply({"text": "t"}, net, image, 1.0, 0.6, 0.4)
    kwargs = controlnet.control_kwargs("img2img", second["controls"], (16, 16))
    assert kwargs["control_image"][0].size == (16, 16) and kwargs["controlnet_conditioning_scale"] == [0.8, 0.5]
    assert "image" in controlnet.control_kwargs("text2img", first["controls"], (16, 16))


def test_canny_finds_one_pixel_edges_and_rejects_inverted_thresholds():
    img = Image.new("RGB", (64, 64), "black")
    ImageDraw.Draw(img).rectangle((16, 16, 47, 47), fill="white")
    edges = np.asarray(canny(img, 100, 200))[..., 0]
    assert set(np.unique(edges)) == {0, 255}
    assert list(np.nonzero(edges[:, 32])[0]) == [15, 47]  # one pixel per side of the square
    assert edges[32, 32] == 0 and edges[2, 2] == 0
    assert canny(Image.new("RGB", (16, 16), "gray"), 100, 200).getextrema() == ((0, 0), (0, 0), (0, 0))
    with pytest.raises(ValueError):
        NODE_REGISTRY["CannyEdges"]().run(img, 300, 200)
