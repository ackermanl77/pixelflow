"""Face and hand detail pass (F20): where the boxes go, how crops are placed and blended back --
with a stand-in redraw; the real SDXL + ViTPose run is opt-in (PIXELFLOW_DETAILER_TESTS=1)."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.nodes import detailer, models


def _person(score=0.9, **overrides):
    """17 COCO keypoints (x, y, score) of an upright person facing the camera."""
    points = {0: (100, 50), 1: (106, 44), 2: (94, 44), 3: (114, 48), 4: (86, 48),
              5: (125, 90), 6: (75, 90), 7: (135, 140), 8: (65, 140), 9: (140, 185), 10: (60, 185)}
    out = [(0.0, 0.0, 0.0)] * 17
    for index, (x, y) in points.items():
        out[index] = (float(x), float(y), score)
    for index, value in overrides.items():
        out[int(index[1:])] = value
    return out


def test_a_face_box_is_centred_on_the_face_points_and_sized_by_their_spread():
    cx, cy, side = detailer.face_box(_person(), 0.3)
    assert 95 < cx < 105 and 40 < cy < 50
    assert side == pytest.approx(28 * 2.2)  # ears 28 px apart
    hidden = _person(p3=(0, 0, 0.1), p4=(0, 0, 0.1), p1=(0, 0, 0.1))
    assert detailer.face_box(hidden, 0.3) is None  # fewer than three visible points


def test_hand_boxes_sit_past_the_wrists_along_the_forearm():
    boxes = detailer.hand_boxes(_person(), 0.3)
    assert len(boxes) == 2
    (lx, ly, lside), (rx, ry, rside) = boxes
    forearm = np.hypot(5, 45)
    assert lside == pytest.approx(forearm) and ly > 185 and lx > 140  # beyond the left wrist
    assert rx < 60 and ry > 185
    assert len(detailer.hand_boxes(_person(p9=(140, 185, 0.1)), 0.3)) == 1  # a hidden wrist: no box


def test_crops_stay_inside_the_image_and_tiny_ones_are_skipped():
    assert detailer.crop_rect((5, 5, 40), (200, 100), 0.0) == (0, 0, 40, 40)  # moved in, not shrunk
    assert detailer.crop_rect((190, 95, 40), (200, 100), 0.5) == (140, 40, 200, 100)
    assert detailer.crop_rect((50, 50, 10), (200, 100), 0.0) is None
    mask = np.asarray(detailer.blend_mask(100, 0.2))
    assert mask[50, 50] == 255 and mask[0, 0] < 30 and 0 < mask[50, 12] < 255


def test_detail_redraws_only_the_crops():
    image = Image.new("RGB", (300, 200), (40, 40, 40))
    seen = []

    def redraw(crop):
        seen.append(crop.size)
        return Image.new("RGB", crop.size, (250, 0, 0))

    out = np.asarray(detailer.detail(image, [(100, 100, 60), (5, 5, 4)], redraw, 512, padding=0.0, feather=0.0))
    assert seen == [(512, 512)]  # the 4-px box was skipped; the crop went in at the model's size
    assert tuple(out[100, 100]) == (250, 0, 0) and tuple(out[10, 250]) == (40, 40, 40)


def test_the_native_size_follows_the_model():
    assert detailer.native_size({"arch": models.ARCH_SDXL}) == 1024
    assert detailer.native_size({"arch": models.ARCH_SD}) == 512


_CHECKPOINT = Path(os.environ.get("PIXELFLOW_DETAILER_CHECKPOINT", "D:/models/checkpoints/SDXL/sd_xl_base_1.0.safetensors"))


@pytest.mark.skipif(os.environ.get("PIXELFLOW_DETAILER_TESTS") != "1", reason="set PIXELFLOW_DETAILER_TESTS=1 (SDXL + pose models)")
def test_a_small_face_in_a_real_sdxl_image_is_found_and_redrawn(monkeypatch, tmp_path):
    from backend.nodes import generation, pose, runtime

    if not _CHECKPOINT.is_file():
        pytest.skip(f"{_CHECKPOINT} not found")
    monkeypatch.setattr(models, "local_model_path", lambda kind, name: _CHECKPOINT)
    monkeypatch.delenv("PIXELFLOW_TEST_MODE", raising=False)
    entry = models.PipelineCache().get(_CHECKPOINT.name)
    model = {"name": _CHECKPOINT.name, "pipeline": entry.pipe, "arch": entry.arch, "dtype": entry.dtype, "loras": []}
    scene = {"text": "full body photograph of a woman standing in a city park, facing the camera, daylight", "clip_skip": 0}
    image = generation.KSampler().sample(model, scene, Image.new("RGB", (832, 1216)), seed=11, steps=25, cfg=6.0)["result"][0][0]
    image.save(tmp_path / "before.png")
    device = "cuda" if runtime.torch().cuda.is_available() else "cpu"
    people = pose.detect_poses(image, pose.load_pose_models(pose.POSE_MODELS[0], device), device)
    faces = [box for box in (detailer.face_box(p, 0.3) for p in people) if box is not None]
    assert faces, "no face found in the generated scene"
    detail_prompt = {"text": "detailed face of a woman, sharp eyes, natural skin", "clip_skip": 0}
    (after,) = detailer.DetailFacesHands().run(image, model, detail_prompt, targets="faces", denoise=0.45, steps=20,
                                               cfg=6.0, seed=3)
    after = after[0]
    after.save(tmp_path / "after.png")
    before_px, after_px = np.asarray(image, dtype=np.float32), np.asarray(after, dtype=np.float32)
    left, top, right, bottom = detailer.crop_rect(faces[0], image.size, 0.3)
    inner = (slice(top + (bottom - top) // 4, bottom - (bottom - top) // 4), slice(left + (right - left) // 4, right - (right - left) // 4))
    assert np.abs(after_px[inner] - before_px[inner]).mean() > 3  # the face was redrawn
    untouched = np.ones(before_px.shape[:2], bool)
    for box in faces:
        l, t, r, b = detailer.crop_rect(box, image.size, 0.3)
        untouched[t:b, l:r] = False
    assert np.abs(after_px[untouched] - before_px[untouched]).max() == 0  # nothing else changed
