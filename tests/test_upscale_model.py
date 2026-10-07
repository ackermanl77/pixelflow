"""B7: model upscaling with spandrel (Load Upscale Model / Upscale Image (model))."""
import numpy as np
import pytest
from PIL import Image

from backend.engine.graph import execute_prompt, validate_prompt
from backend.nodes import NODE_REGISTRY, upscale

spandrel = pytest.importorskip("spandrel")
torch = pytest.importorskip("torch")


@pytest.fixture
def tiny_esrgan(isolated_data_dirs):
    """A tiny 2x ESRGAN saved as models/upscale_models/tiny-2x.pth."""
    from spandrel.architectures.ESRGAN import ESRGAN

    torch.manual_seed(0)
    model = ESRGAN(in_nc=3, out_nc=3, num_filters=8, num_blocks=1, scale=2)
    folder = isolated_data_dirs["upscale_models"]
    folder.mkdir(parents=True)
    torch.save(model.state_dict(), folder / "tiny-2x.pth")
    yield "tiny-2x.pth"
    upscale._cache.clear()


def test_the_nodes_upscale_a_real_graph_2x(tiny_esrgan, output_dir):
    assert NODE_REGISTRY["UpscaleModelLoader"].INPUT_TYPES()["required"]["model_name"][0] == ["tiny-2x.pth"]
    outputs = execute_prompt(validate_prompt({
        "load": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": tiny_esrgan}},
        "img": {"class_type": "EmptyImage", "inputs": {"width": 96, "height": 64, "color": "#336699"}},
        "up": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["load", 0], "image": ["img", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"image": ["up", 0], "filename_prefix": "up"}},
    }))
    with Image.open(output_dir / outputs["save"]["images"][0]) as result:
        assert result.size == (192, 128)


def test_tiling_matches_one_pass(tiny_esrgan):
    descriptor = upscale.load_upscale_model(tiny_esrgan)
    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 255, (70, 90, 3), dtype=np.uint8))
    whole = np.asarray(upscale.upscale_image(descriptor, image, tile=512), dtype=int)
    tiled = np.asarray(upscale.upscale_image(descriptor, image, tile=48, overlap=12), dtype=int)
    assert whole.shape == (140, 180, 3)
    # Inside the overlaps the receptive field is covered, so tiles agree with one pass
    # (fp16 on a GPU may differ by a level).
    assert np.abs(whole - tiled).max() <= 2


def test_missing_or_unsafe_names_are_refused(isolated_data_dirs):
    with pytest.raises(FileNotFoundError):
        upscale.load_upscale_model("nope.pth")
    with pytest.raises(FileNotFoundError):
        upscale.load_upscale_model("../../secrets.pth")
    with pytest.raises(ValueError):
        NODE_REGISTRY["UpscaleModelLoader"]().load("")
