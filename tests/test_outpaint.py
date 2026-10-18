"""Pad Image for Outpainting (B21)."""
import numpy as np
from PIL import Image

from backend.nodes.image_ops import ImagePadForOutpaint


def _image(width=64, height=48):
    array = np.zeros((height, width, 3), np.uint8)
    array[:, : width // 2] = (200, 30, 30)  # red left half
    array[:, width // 2:] = (30, 30, 200)  # blue right half
    return Image.fromarray(array)


def test_the_canvas_grows_by_multiples_of_8_and_the_original_is_kept():
    padded, mask = ImagePadForOutpaint().run(_image(), left=5, top=0, right=100, bottom=16, feathering=0)
    assert padded.size == (8 + 64 + 104, 48 + 16) and mask.size == padded.size
    original = np.asarray(padded)[0:48, 8:72]
    assert np.array_equal(original, np.asarray(_image()))


def test_the_mask_is_white_outside_feathered_at_the_seam_and_black_inside():
    _padded, mask = ImagePadForOutpaint().run(_image(), right=64, feathering=16)
    m = np.asarray(mask).astype(int)
    assert m[:, 64:].min() == 255  # the new area
    assert m[:, :40].max() == 0  # deep inside the original: kept
    seam = m[24, 48:64]
    assert seam[0] < seam[-1] and 0 < seam[8] < 255  # a soft ramp up to the edge


def test_the_new_area_starts_from_the_edge_colours_not_grey():
    padded, _mask = ImagePadForOutpaint().run(_image(), left=32, right=32, feathering=0)
    a = np.asarray(padded).astype(int)
    assert a[24, 2, 0] > 150 and a[24, 2, 2] < 90  # left extension: red, like the left edge
    assert a[24, -3, 2] > 150 and a[24, -3, 0] < 90  # right extension: blue


def test_no_padding_is_a_plain_copy_and_batches_stay_batches():
    padded, mask = ImagePadForOutpaint().run(_image(), right=0, feathering=10)
    assert padded.size == (64, 48) and np.asarray(mask).max() == 0
    images, masks = ImagePadForOutpaint().run([_image(), _image(32, 32)], top=8, right=0)
    assert [i.size for i in images] == [(64, 56), (32, 40)] and len(masks) == 2


def test_the_outpaint_graph_runs_end_to_end_with_a_real_pipeline(upload_dir, output_dir):
    import io

    from _hub import skip_if_hub_unreachable

    from backend.engine.graph import execute_prompt, validate_prompt

    buffer = io.BytesIO()
    _image(32, 32).save(buffer, format="PNG")
    (upload_dir / "src.png").write_bytes(buffer.getvalue())
    graph = {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": "hf-internal-testing/tiny-stable-diffusion-torch"}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "more of the scene"}},
        "load": {"class_type": "LoadImage", "inputs": {"filename": "src.png"}},
        "pad": {"class_type": "ImagePadForOutpaint", "inputs": {"image": ["load", 0], "left": 0, "top": 0, "right": 32,
                                                                "bottom": 0, "feathering": 0}},
        "sampler": {"class_type": "KSamplerInpaint", "inputs": {"model": ["ckpt", 0], "positive": ["clip", 0],
                                                                "image": ["pad", 0], "mask": ["pad", 1], "denoise": 1.0,
                                                                "sampler": "euler", "seed": 3, "steps": 2, "cfg": 7.0,
                                                                "batch_size": 1}},
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "outpaint"}},
    }
    try:
        result = execute_prompt(validate_prompt(graph))
    except Exception as exc:  # noqa: BLE001
        skip_if_hub_unreachable(exc)
        raise
    with Image.open(output_dir / result["save"]["images"][0]) as out:
        assert out.size == (64, 32)  # the canvas grew to the right
