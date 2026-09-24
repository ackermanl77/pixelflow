"""Every IMAGE consumer must accept KSampler's output shape (a list of images)
as well as a single PIL image."""

import pytest
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from requests.exceptions import ConnectionError as RequestsConnectionError

from backend.engine.graph import execute_prompt
from backend.nodes import NODE_REGISTRY
from backend.nodes.base import NodeBase, map_image_pairs, register_node

TINY_SD_CHECKPOINT = "hf-internal-testing/tiny-stable-diffusion-torch"

# (class_type, extra widget inputs) for every single-IMAGE-input image op.
SINGLE_INPUT_OPS = [
    ("ImageResize", {"width": 20, "height": 10}),
    ("ImageBlur", {"radius": 1.0}),
    ("ImageBrightnessContrast", {"brightness": 1.2, "contrast": 0.8}),
    ("ImageHueSaturation", {"hue_shift_degrees": 30.0, "saturation": 1.5}),
    ("ImageTextOverlay", {"text": "hi", "x": 1, "y": 1, "size": 8, "color": "#ffffff"}),
]


def _batch(n=3, size=(16, 16)):
    return [Image.new("RGB", size, (i * 40, 0, 0)) for i in range(n)]


@pytest.mark.parametrize("class_type,kwargs", SINGLE_INPUT_OPS)
def test_image_ops_map_over_a_batch(class_type, kwargs):
    node = NODE_REGISTRY[class_type]()
    (out,) = getattr(node, node.FUNCTION)(image=_batch(), **kwargs)
    assert isinstance(out, list) and len(out) == 3
    assert all(isinstance(img, Image.Image) for img in out)


@pytest.mark.parametrize("class_type,kwargs", SINGLE_INPUT_OPS)
def test_image_ops_still_return_a_single_image_for_a_single_input(class_type, kwargs):
    node = NODE_REGISTRY[class_type]()
    (out,) = getattr(node, node.FUNCTION)(image=Image.new("RGB", (16, 16)), **kwargs)
    assert isinstance(out, Image.Image)


def test_resize_applies_to_every_image_in_a_batch():
    (out,) = NODE_REGISTRY["ImageResize"]().run(image=_batch(), width=20, height=10)
    assert [img.size for img in out] == [(20, 10)] * 3


def test_blend_broadcasts_a_single_overlay_across_a_batch():
    node = NODE_REGISTRY["ImageBlend"]()
    (out,) = node.run(base=_batch(), overlay=Image.new("RGB", (4, 4)), mode="multiply", opacity=0.5)
    assert len(out) == 3
    (out,) = node.run(base=Image.new("RGB", (16, 16)), overlay=_batch(2), mode="screen", opacity=1.0)
    assert len(out) == 2
    (out,) = node.run(base=_batch(2), overlay=_batch(2), mode="normal", opacity=1.0)
    assert len(out) == 2


def test_blend_rejects_mismatched_batch_sizes():
    with pytest.raises(ValueError, match="different sizes"):
        map_image_pairs(_batch(2), _batch(3), lambda a, b: a)


def test_ksampler_style_list_output_flows_through_image_ops_to_save(output_dir):
    NODE_REGISTRY.pop("_FakeBatchSource", None)

    @register_node()
    class _FakeBatchSource(NodeBase):
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "run"

        def run(self):
            return (_batch(2),)

    try:
        prompt = {
            "src": {"class_type": "_FakeBatchSource", "inputs": {}},
            "blur": {"class_type": "ImageBlur", "inputs": {"image": ["src", 0], "radius": 1.0}},
            "resize": {"class_type": "ImageResize", "inputs": {"image": ["blur", 0], "width": 8, "height": 8}},
            "save": {"class_type": "SaveImage", "inputs": {"image": ["resize", 0], "filename_prefix": "b"}},
        }
        outputs = execute_prompt(prompt)
        assert len(outputs["save"]["images"]) == 2
        for name in outputs["save"]["images"]:
            with Image.open(output_dir / name) as img:
                assert img.size == (8, 8)
    finally:
        NODE_REGISTRY.pop("_FakeBatchSource", None)


def test_real_ksampler_output_into_blur_and_as_latent_image(output_dir):
    """The originally reported crash: KSampler -> Blur. Also feeds a
    KSampler's list output into a second KSampler's latent_image."""
    sampler_inputs = {
        "model": ["ckpt", 0],
        "positive": ["clip", 0],
        "seed": 3,
        "steps": 2,
        "cfg": 7.0,
        "batch_size": 2,
    }
    prompt = {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY_SD_CHECKPOINT}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red circle"}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "color": "#000000"}},
        "s1": {"class_type": "KSamplerStub", "inputs": dict(sampler_inputs, latent_image=["latent", 0])},
        "s2": {"class_type": "KSamplerStub", "inputs": dict(sampler_inputs, latent_image=["s1", 0])},
        "blur": {"class_type": "ImageBlur", "inputs": {"image": ["s2", 0], "radius": 2.0}},
        "save": {"class_type": "SaveImage", "inputs": {"image": ["blur", 0], "filename_prefix": "blurred"}},
    }
    try:
        outputs = execute_prompt(prompt)
    except (HfHubHTTPError, RequestsConnectionError) as exc:
        pytest.skip(f"Hugging Face Hub unreachable: {exc}")
    assert len(outputs["save"]["images"]) == 2
