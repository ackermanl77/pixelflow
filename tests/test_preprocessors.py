"""ControlNet preprocessors (backend/nodes/preprocessors.py): Depth Map with a
tiny random Depth Anything model (no download), and the classical Soft Edges,
Lineart and Scribble."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from backend.nodes import NODE_REGISTRY
from backend.nodes import preprocessors as pp


def _scene(noise=6.0):
    image = Image.new("RGB", (160, 120), (40, 60, 90))
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 20, 100, 90), fill=(220, 180, 120))
    draw.rectangle((105, 60, 150, 110), fill=(90, 200, 90))
    rng = np.random.default_rng(0)
    noisy = np.asarray(image, dtype=np.float32) + rng.normal(0, noise, (120, 160, 3))
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def _gray(image):
    return np.asarray(image.convert("L"))


def test_nodes_are_registered_without_importing_torch():
    for name in ("DepthMap", "SoftEdges", "Lineart", "Scribble"):
        assert name in NODE_REGISTRY
    assert "depth-anything/Depth-Anything-V2-Small-hf" in NODE_REGISTRY["DepthMap"].INPUT_TYPES()["required"]["model"][0]


@pytest.mark.parametrize("fn", [pp.soft_edges, pp.lineart, pp.scribble])
def test_edges_are_white_on_black_at_the_shapes_outline(fn):
    out = fn(_scene())
    gray = _gray(out)
    assert out.size == (160, 120) and out.mode == "RGB"
    assert gray[50:60, 28:35].max() > 200  # the circle's left edge
    assert gray[55, 65] < 60  # inside the circle: flat
    assert gray[5:15, 5:15].mean() < 60  # flat background


def test_lineart_ignores_sensor_noise_but_scribble_is_binary_and_thick():
    lines = _gray(pp.lineart(_scene(noise=8.0)))
    assert (lines[:15, :] > 64).mean() < 0.01  # noisy flat area: no speckle
    strokes = _gray(pp.scribble(_scene(), thickness=5))
    assert set(np.unique(strokes)) <= {0, 255}
    thin = _gray(pp.scribble(_scene(), thickness=1))
    assert (strokes == 255).sum() > 1.2 * (thin == 255).sum()


def test_more_soft_edge_detail_keeps_more_texture_and_thinner_contours():
    textured = _scene(noise=14.0)
    fine, coarse = (_gray(pp.soft_edges(textured, detail=d)).astype(float) for d in (4.0, 0.5))
    assert fine[5:15, 5:60].mean() > coarse[5:15, 5:60].mean() + 10  # background texture
    assert (fine[55, 20:45] > 128).sum() < (coarse[55, 20:45] > 128).sum()  # the circle's contour band


@pytest.fixture(scope="module")
def tiny_depth_model(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    backbone = transformers.Dinov2Config(
        hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64, image_size=56, patch_size=14,
        out_features=["stage1", "stage2", "stage3", "stage4"], reshape_hidden_states=False,
    )
    config = transformers.DepthAnythingConfig(
        backbone_config=backbone, reassemble_hidden_size=32, neck_hidden_sizes=[8, 16, 32, 32],
        fusion_hidden_size=16, head_hidden_size=8, patch_size=14,
    )
    folder = tmp_path_factory.mktemp("tiny-depth")
    transformers.DepthAnythingForDepthEstimation(config).save_pretrained(folder)
    transformers.DPTImageProcessor(size={"height": 56, "width": 56}, keep_aspect_ratio=False, ensure_multiple_of=14).save_pretrained(folder)
    return str(folder)


def test_depth_map_node_runs_a_depth_anything_model(tiny_depth_model):
    node = NODE_REGISTRY["DepthMap"]()
    batch = [_scene(), _scene().resize((90, 70))]
    (out,) = node.run(batch, model=tiny_depth_model)
    assert [image.size for image in out] == [(160, 120), (90, 70)]  # each at its source size
    gray = _gray(out[0])
    # Normalized per image: 0..255, or all black when the (random) model's output is flat.
    assert gray.min() == 0 and gray.max() in (0, 255)
    assert pp.load_depth_model(tiny_depth_model, "cpu") is pp.load_depth_model(tiny_depth_model, "cpu")  # cached


def test_a_missing_depth_model_says_how_to_get_it(tmp_path):
    pytest.importorskip("transformers")
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        pp.load_depth_model(str(tmp_path / "not-a-model"), "cpu")
