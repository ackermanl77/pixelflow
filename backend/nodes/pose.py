"""Pose preprocessor for an OpenPose ControlNet (report item B7's missing piece).

Top-down pose estimation with two transformers models, downloaded from the
Hugging Face Hub on first use like the Depth Map model: RT-DETR finds the
people (COCO "person" boxes), then ViTPose estimates 17 COCO keypoints in each
box. The keypoints become OpenPose's 18-point body layout (the neck is the
middle of the shoulders) and are drawn the way ControlNet's OpenPose models
were trained on (controlnet_aux's draw_bodypose): coloured limb ellipses at
60 % on black, then a dot per joint.

transformers' own ViTPose image processor needs scipy (an affine crop and a
Gaussian blur), which PixelFlow doesn't ship; the crop and the heatmap
decoding here do the same with PIL and torch (`crop_person`,
`decode_heatmaps`) -- checked against transformers' processor in the tests
when scipy is available.
"""

from __future__ import annotations

import math
import threading
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw

from .base import NodeBase, map_images, register_node

PERSON_DETECTOR = "PekingU/rtdetr_r50vd_coco_o365"
POSE_MODELS = ["usyd-community/vitpose-base-simple", "usyd-community/vitpose-plus-small"]
BOX_PADDING = 1.25  # ViTPose's own top-down padding around each person box

# OpenPose body (18 points) from COCO's 17: nose, neck, R shoulder/elbow/wrist,
# L shoulder/elbow/wrist, R hip/knee/ankle, L hip/knee/ankle, R/L eye, R/L ear.
# None = the neck, the middle of the two shoulders.
OPENPOSE_FROM_COCO = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
# Limbs as OpenPose point pairs (controlnet_aux's limbSeq, 0-based) and colours.
LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
    (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

Keypoints = List[Tuple[float, float, float]]  # (x, y, score) per point, image pixels

_cache: dict = {}
_lock = threading.Lock()


# -- geometry (pure) ------------------------------------------------------------------
def person_crop_box(box: Sequence[float], input_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """(left, top, width, height) of the region ViTPose looks at for a person box
    (x, y, w, h): the box grown to the model's input aspect, then padded."""
    x, y, w, h = (float(v) for v in box)
    in_w, in_h = input_size
    aspect = in_w / in_h
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    w, h = w * BOX_PADDING, h * BOX_PADDING
    cx, cy = float(box[0]) + float(box[2]) / 2, float(box[1]) + float(box[3]) / 2
    return cx - w / 2, cy - h / 2, w, h


def crop_person(image: Image.Image, region, input_size: Tuple[int, int]) -> Image.Image:
    """The region resampled to the model's input size (pixels outside the image are black)."""
    left, top, width, height = region
    in_w, in_h = input_size
    sx, sy = width / in_w, height / in_h
    # PIL's affine data maps output pixel centres to input coordinates.
    data = (sx, 0.0, left + 0.5 * sx - 0.5, 0.0, sy, top + 0.5 * sy - 0.5)
    return image.convert("RGB").transform((in_w, in_h), Image.Transform.AFFINE, data, resample=Image.Resampling.BILINEAR)


def decode_heatmaps(heatmaps, region) -> Keypoints:
    """Keypoints in image pixels from one person's (joints, h, w) heatmaps: the
    peak of each, moved a quarter pixel towards its higher neighbour (the
    usual sub-pixel refinement), scored by the peak value."""
    import torch

    joints, hm_h, hm_w = heatmaps.shape
    flat = heatmaps.reshape(joints, -1)
    scores, index = flat.max(dim=1)
    ys = torch.div(index, hm_w, rounding_mode="floor").float()
    xs = (index % hm_w).float()
    points: Keypoints = []
    left, top, width, height = region
    for j in range(joints):
        px, py = float(xs[j]), float(ys[j])
        ix, iy = int(px), int(py)
        hm = heatmaps[j]
        if 0 < ix < hm_w - 1:
            px += 0.25 * math.copysign(1.0, float(hm[iy, ix + 1] - hm[iy, ix - 1])) if hm[iy, ix + 1] != hm[iy, ix - 1] else 0.0
        if 0 < iy < hm_h - 1:
            py += 0.25 * math.copysign(1.0, float(hm[iy + 1, ix] - hm[iy - 1, ix])) if hm[iy + 1, ix] != hm[iy - 1, ix] else 0.0
        x = left + (px + 0.5) * width / hm_w - 0.5
        y = top + (py + 0.5) * height / hm_h - 0.5
        points.append((x, y, float(scores[j])))
    return points


def coco_to_openpose(points: Keypoints, threshold: float) -> List[Tuple[float, float, float] | None]:
    """OpenPose's 18 body points from COCO's 17 (None where not confidently seen)."""
    out: List[Tuple[float, float, float] | None] = []
    for source in OPENPOSE_FROM_COCO:
        if source is None:
            left, right = points[5], points[6]
            if left[2] >= threshold and right[2] >= threshold:
                out.append(((left[0] + right[0]) / 2, (left[1] + right[1]) / 2, min(left[2], right[2])))
            else:
                out.append(None)
        else:
            point = points[source]
            out.append(point if point[2] >= threshold else None)
    return out


def _ellipse(center, length: float, width: float, angle_deg: float, steps: int = 36):
    cx, cy = center
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    pts = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        ex, ey = length * math.cos(t), width * math.sin(t)
        pts.append((cx + ex * cos_a - ey * sin_a, cy + ex * sin_a + ey * cos_a))
    return pts


def draw_openpose(size: Tuple[int, int], people: Sequence[Sequence[Tuple[float, float, float] | None]],
                  stick_width: float | None = None) -> Image.Image:
    """OpenPose-style skeletons (18-point lists from coco_to_openpose) on black."""
    width, height = size
    stick = stick_width if stick_width is not None else max(2.0, 4.0 * max(width, height) / 512)
    limbs = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(limbs)
    for person in people:
        for (a, b), color in zip(LIMBS, COLORS):
            pa, pb = person[a], person[b]
            if pa is None or pb is None:
                continue
            length = math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / 2
            angle = math.degrees(math.atan2(pb[1] - pa[1], pb[0] - pa[0]))
            draw.polygon(_ellipse(((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2), length, stick, angle), fill=color)
    canvas = Image.eval(limbs, lambda v: int(v * 0.6))  # limbs at 60 %, as the training data
    draw = ImageDraw.Draw(canvas)
    radius = stick
    for person in people:
        for point, color in zip(person, COLORS):
            if point is not None:
                draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=color)
    return canvas


# -- models -------------------------------------------------------------------------------
def load_pose_models(pose_repo: str, device: str):
    """(detector processor, detector, pose processor config, pose model), cached."""
    key = (pose_repo, device)
    with _lock:
        if key in _cache:
            return _cache[key]
    from transformers import AutoImageProcessor, AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

    try:
        detector_processor = AutoProcessor.from_pretrained(PERSON_DETECTOR)
        detector = RTDetrForObjectDetection.from_pretrained(PERSON_DETECTOR).to(device).eval()
        pose_processor = AutoImageProcessor.from_pretrained(pose_repo)  # only its size/mean/std are used
        pose_model = VitPoseForPoseEstimation.from_pretrained(pose_repo).to(device).eval()
    except OSError as error:
        raise RuntimeError(
            f"Pose models couldn't be loaded ({PERSON_DETECTOR}, {pose_repo}): they're downloaded from the "
            f"Hugging Face Hub on first use, so connect to the internet once ({error})"
        ) from error
    entry = (detector_processor, detector, pose_processor, pose_model)
    with _lock:
        _cache.clear()  # one set in memory at a time
        _cache[key] = entry
    return entry


def detect_poses(image: Image.Image, models, device: str, person_threshold: float = 0.4) -> List[Keypoints]:
    """COCO keypoints for every person RT-DETR finds in `image`."""
    from . import runtime

    torch = runtime.torch()
    detector_processor, detector, pose_processor, pose_model = models
    rgb = image.convert("RGB")
    with torch.inference_mode():
        inputs = detector_processor(images=rgb, return_tensors="pt").to(device)
        found = detector_processor.post_process_object_detection(
            detector(**inputs), target_sizes=torch.tensor([(rgb.height, rgb.width)]), threshold=person_threshold
        )[0]
        person = [i for i, label in detector.config.id2label.items() if label.lower() == "person"]
        keep = found["labels"] == (person[0] if person else 0)
        boxes = found["boxes"][keep].float().cpu().tolist()
        if not boxes:
            return []
        size = pose_processor.size
        input_size = (int(size["width"]), int(size["height"]))
        mean = torch.tensor(pose_processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(pose_processor.image_std).view(1, 3, 1, 1)
        regions, crops = [], []
        for x0, y0, x1, y1 in boxes:
            region = person_crop_box((x0, y0, x1 - x0, y1 - y0), input_size)
            regions.append(region)
            crop = torch.frombuffer(bytearray(crop_person(rgb, region, input_size).tobytes()), dtype=torch.uint8)
            crops.append(crop.view(input_size[1], input_size[0], 3).permute(2, 0, 1).float() / 255.0)
        pixel_values = ((torch.stack(crops) - mean) / std).to(device=device, dtype=next(pose_model.parameters()).dtype)
        kwargs = {}
        if getattr(pose_model.config.backbone_config, "num_experts", 1) > 1:
            kwargs["dataset_index"] = torch.zeros(len(crops), dtype=torch.long, device=device)  # COCO head
        heatmaps = pose_model(pixel_values=pixel_values, **kwargs).heatmaps.float().cpu()
    return [decode_heatmaps(heatmaps[i], regions[i]) for i in range(len(regions))]


def openpose_image(image: Image.Image, models, device: str, keypoint_threshold: float = 0.3,
                   person_threshold: float = 0.4) -> Image.Image:
    people = [coco_to_openpose(points, keypoint_threshold)
              for points in detect_poses(image, models, device, person_threshold)]
    return draw_openpose(image.size, people)


@register_node("Pose (OpenPose)")
class OpenPose(NodeBase):
    """Finds the people in the image and draws their body poses as OpenPose
    skeletons on black, for an OpenPose ControlNet (RT-DETR + ViTPose from the
    Hugging Face Hub, downloaded on first use, ~250 MB; GPU when there is one)."""

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    DESCRIPTION = "OpenPose body skeletons (for an OpenPose ControlNet)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (POSE_MODELS, {"allow_custom": True, "tooltip": "A transformers ViTPose model id"}),
                "keypoint_threshold": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05,
                                                 "tooltip": "Joints less certain than this are left out"}),
                "person_threshold": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 0.95, "step": 0.05,
                                               "tooltip": "How sure the detector must be that a box is a person"}),
            }
        }

    def run(self, image, model=POSE_MODELS[0], keypoint_threshold=0.3, person_threshold=0.4):
        from . import runtime

        torch = runtime.torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models = load_pose_models(model or POSE_MODELS[0], device)
        return (map_images(image, lambda img: openpose_image(img, models, device, float(keypoint_threshold),
                                                             float(person_threshold))),)
