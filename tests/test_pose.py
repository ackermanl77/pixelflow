"""Pose (OpenPose) preprocessor (backend/nodes/pose.py): the crop/heatmap
geometry, the COCO -> OpenPose mapping and drawing, the model plumbing with a
tiny random ViTPose (no download), and -- opt-in, with PIXELFLOW_POSE_SMOKE set
to a photo of a person -- the real RT-DETR + ViTPose models."""
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from backend.nodes import NODE_REGISTRY
from backend.nodes import pose


def test_the_node_is_registered_without_importing_torch():
    assert "OpenPose" in NODE_REGISTRY
    inputs = NODE_REGISTRY["OpenPose"].INPUT_TYPES()["required"]
    assert inputs["model"][0][0] == "usyd-community/vitpose-base-simple"
    assert inputs["keypoint_threshold"][1]["default"] == 0.3


def test_the_crop_and_the_heatmap_decoding_agree():
    torch = pytest.importorskip("torch")
    image = Image.new("RGB", (400, 300))
    ImageDraw.Draw(image).ellipse((247, 97, 253, 103), fill=(255, 255, 255))  # a dot at (250, 100)
    region = pose.person_crop_box((200, 40, 90, 160), (192, 256))
    left, top, width, height = region
    assert abs(width / height - 192 / 256) < 1e-9 and height == pytest.approx(160 * 1.25)
    crop = pose.crop_person(image, region, (192, 256))
    assert crop.size == (192, 256)
    # The crop at heatmap resolution, as a one-joint "heatmap": decoding must land on the dot.
    small = crop.convert("L").resize((48, 64), Image.Resampling.BOX)
    heatmap = torch.tensor(np.asarray(small, dtype=np.float32))[None]
    (x, y, score), = pose.decode_heatmaps(heatmap, region)
    assert abs(x - 250) < width / 48 and abs(y - 100) < height / 64 and score > 0


def test_coco_points_become_openpose_with_a_neck():
    points = [(float(i), float(i * 2), 0.9) for i in range(17)]
    points[3] = (3.0, 6.0, 0.1)  # left ear: not sure
    out = pose.coco_to_openpose(points, threshold=0.3)
    assert len(out) == 18
    assert out[1] == ((5 + 6) / 2, (10 + 12) / 2, 0.9)  # neck between the shoulders
    assert out[2] == points[6] and out[5] == points[5]  # right / left shoulder
    assert out[17] is None  # OpenPose's left ear = COCO 3
    no_shoulder = list(points)
    no_shoulder[5] = (0.0, 0.0, 0.0)
    assert pose.coco_to_openpose(no_shoulder, 0.3)[1] is None


def test_skeletons_are_drawn_in_openpose_colours():
    person = [None] * 18
    person[1] = (50.0, 40.0, 1.0)  # neck
    person[2] = (20.0, 40.0, 1.0)  # right shoulder
    canvas = np.asarray(pose.draw_openpose((100, 80), [person], stick_width=3))
    assert canvas[0, 0].tolist() == [0, 0, 0]
    assert canvas[40, 35].tolist() == [153, 0, 0]  # limb 1-2: red at 60 %
    assert canvas[40, 50].tolist() == list(pose.COLORS[1])  # the neck's joint dot, full colour
    assert canvas[40, 20].tolist() == list(pose.COLORS[2])


@pytest.fixture(scope="module")
def tiny_vitpose(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    backbone = transformers.VitPoseBackboneConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        image_size=[64, 48], patch_size=[16, 16], out_indices=[2],
    )
    config = transformers.VitPoseConfig(backbone_config=backbone, num_labels=17)
    folder = tmp_path_factory.mktemp("tiny-vitpose")
    transformers.VitPoseForPoseEstimation(config).save_pretrained(folder)
    transformers.models.vitpose.image_processing_vitpose.VitPoseImageProcessor(
        size={"height": 64, "width": 48}).save_pretrained(folder)
    return str(folder)


class _FakeDetections(dict):
    def to(self, _device):
        return self


def test_detect_poses_runs_the_pose_model_on_each_person_box(tiny_vitpose):
    import torch
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation

    class Processor:
        def __call__(self, images, return_tensors):
            return _FakeDetections(pixel_values=torch.zeros(1))

        def post_process_object_detection(self, outputs, target_sizes, threshold):
            return [{"boxes": torch.tensor([[40.0, 20.0, 100.0, 140.0], [120.0, 30.0, 160.0, 110.0], [0.0, 0.0, 5.0, 5.0]]),
                     "labels": torch.tensor([0, 0, 2]), "scores": torch.tensor([0.9, 0.8, 0.9])}]

    class Detector:
        config = type("Config", (), {"id2label": {0: "person", 2: "car"}})()

        def __call__(self, **kwargs):
            return None

    models = (Processor(), Detector(), AutoImageProcessor.from_pretrained(tiny_vitpose),
              VitPoseForPoseEstimation.from_pretrained(tiny_vitpose).eval())
    people = pose.detect_poses(Image.new("RGB", (200, 160), (90, 90, 90)), models, "cpu")
    assert len(people) == 2  # the car is not a person
    first = people[0]
    assert len(first) == 17 and all(isinstance(v, float) for point in first for v in point)
    left, top, width, height = pose.person_crop_box((40, 20, 60, 120), (48, 64))
    assert all(left - 1 <= x <= left + width + 1 and top - 1 <= y <= top + height + 1 for x, y, _s in first)
    drawn = pose.openpose_image(Image.new("RGB", (200, 160)), models, "cpu", keypoint_threshold=0.0)
    assert drawn.size == (200, 160) and np.asarray(drawn).max() > 0


def test_missing_pose_models_say_how_to_get_them(tmp_path):
    pytest.importorskip("transformers")
    pose._cache.clear()
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        pose.load_pose_models(str(tmp_path / "not-a-model"), "cpu")


@pytest.mark.skipif(not os.environ.get("PIXELFLOW_POSE_SMOKE"), reason="set PIXELFLOW_POSE_SMOKE to a photo of a person")
def test_real_models_find_the_person_in_a_photo():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(os.environ["PIXELFLOW_POSE_SMOKE"]).convert("RGB")
    models = pose.load_pose_models(pose.POSE_MODELS[0], device)
    people = pose.detect_poses(image, models, device)
    assert people
    openpose = pose.coco_to_openpose(people[0], 0.3)
    assert openpose[0] is not None and openpose[1] is not None  # nose and neck
    (out,) = NODE_REGISTRY["OpenPose"]().run([image])
    assert out[0].size == image.size and np.asarray(out[0]).max() > 0
