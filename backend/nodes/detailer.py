"""Face and hand detail pass (F20), as ADetailer does in A1111: small faces and hands come out
mushy because the model draws them with few latent pixels, so each one is cropped, re-sampled at
the model's own resolution with a prompt of its own, and pasted back with a soft edge.

Where they are comes from the Pose (OpenPose) node's models (RT-DETR people + ViTPose COCO
keypoints, downloaded on first use): a face box around the nose, eyes and ears, a hand box ahead
of each wrist along the forearm. The re-sampling is KSampler (Image) on the given MODEL, so LoRAs,
ControlNets and regional prompts work as anywhere else.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter

from . import generation, pose
from .base import NodeBase, as_image_list, register_node

# COCO keypoints (ViTPose's order).
FACE_POINTS = (0, 1, 2, 3, 4)  # nose, eyes, ears
HANDS = ((7, 9), (8, 10))  # (elbow, wrist): left, right
TARGETS = ["faces", "hands", "faces and hands"]
MIN_BOX = 24  # px: smaller detections are noise, not a face or hand to redraw

Box = Tuple[float, float, float]  # centre x, centre y, side (a square, image pixels)


def face_box(points, threshold: float) -> Box | None:
    """A square around a face: centred on its visible nose/eye/ear points, 2.2x their spread
    (they span the face's middle, not forehead to chin). None when fewer than 3 are visible."""
    seen = [(x, y) for x, y, score in (points[i] for i in FACE_POINTS) if score >= threshold]
    if len(seen) < 3:
        return None
    xs, ys = [p[0] for p in seen], [p[1] for p in seen]
    spread = max(max(xs) - min(xs), max(ys) - min(ys))
    return (sum(xs) / len(xs), sum(ys) / len(ys) - 0.1 * spread, max(spread * 2.2, MIN_BOX))


def hand_boxes(points, threshold: float) -> List[Box]:
    """Squares around the hands: past each visible wrist, along the forearm (elbow -> wrist), as
    big as the forearm is long."""
    boxes = []
    for elbow, wrist in HANDS:
        (ex, ey, es), (wx, wy, ws) = points[elbow], points[wrist]
        if min(es, ws) < threshold:
            continue
        forearm = math.hypot(wx - ex, wy - ey)
        if forearm < MIN_BOX / 2:
            continue
        ux, uy = (wx - ex) / forearm, (wy - ey) / forearm
        boxes.append((wx + 0.4 * forearm * ux, wy + 0.4 * forearm * uy, max(forearm, MIN_BOX)))
    return boxes


def crop_rect(box: Box, size: Tuple[int, int], padding: float) -> Tuple[int, int, int, int] | None:
    """The padded square crop of `box`, moved (not shrunk) to stay inside the image; None when
    it's below MIN_BOX or bigger than the image."""
    cx, cy, side = box
    side = side * (1.0 + padding)
    width, height = size
    side = min(side, width, height)
    if side < MIN_BOX:
        return None
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return (round(left), round(top), round(left + side), round(top + side))


def blend_mask(side: int, feather: float) -> Image.Image:
    """A soft square mask for pasting a redrawn crop back: opaque in the middle, fading out over
    `feather` of the side at every edge."""
    mask = Image.new("L", (side, side), 0)
    inset = round(side * feather)
    ImageDraw.Draw(mask).rectangle([inset, inset, side - 1 - inset, side - 1 - inset], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(max(1.0, inset / 2))) if inset else mask


def native_size(model) -> int:
    """The square size the model draws best at: 1024 for SDXL, 512 otherwise."""
    from . import models

    return 1024 if (model or {}).get("arch") == models.ARCH_SDXL else 512


def detail(image: Image.Image, boxes: Sequence[Box], redraw, size: int, padding: float, feather: float) -> Image.Image:
    """`image` with every box's crop redrawn by `redraw(crop at size x size) -> image` and pasted back."""
    out = image.convert("RGB").copy()
    for box in boxes:
        rect = crop_rect(box, out.size, padding)
        if rect is None:
            continue
        side = rect[2] - rect[0]
        crop = out.crop(rect).resize((size, size), Image.LANCZOS)
        redrawn = redraw(crop).convert("RGB").resize((side, side), Image.LANCZOS)
        out.paste(redrawn, rect[:2], blend_mask(side, feather))
    return out


@register_node("Detail Faces & Hands")
class DetailFacesHands(NodeBase):
    """Redraws each face and/or hand in the image at the model's own resolution (ADetailer-style):
    found with the Pose (OpenPose) node's models, cropped with `padding` around it, re-sampled with
    `positive` at `denoise`, pasted back with a soft edge. Use after a sampler, with a prompt for
    the detail ("detailed face, sharp eyes" / "detailed hand, five fingers")."""

    CATEGORY = "sampling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    DESCRIPTION = "Redraws faces and hands at full resolution (detect, re-sample the crop, paste back)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING", {"tooltip": "The prompt for the redrawn faces or hands"}),
                "targets": (TARGETS, {"default": "faces"}),
                "denoise": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05,
                                      "tooltip": "How much each crop is redrawn: 0.3-0.5 keeps the likeness"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": -1, "max": generation.MAX_SEED}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "padding": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Extra context around each face/hand, as a fraction of its size"}),
                "feather": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.4, "step": 0.01,
                                      "tooltip": "How softly a redrawn crop blends in, as a fraction of its side"}),
                "keypoint_threshold": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05}),
                "pose_model": (pose.POSE_MODELS, {"allow_custom": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, image, model, positive, targets="faces", denoise=0.4, steps=20, cfg=6.0, seed=0, negative=None,
            padding=0.3, feather=0.12, keypoint_threshold=0.3, pose_model=pose.POSE_MODELS[0], unique_id=None,
            on_progress=None):
        from . import runtime

        torch = runtime.torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detectors = pose.load_pose_models(pose_model or pose.POSE_MODELS[0], device)
        size = native_size(model)
        sampler = generation.KSamplerImage()

        def redraw(crop):
            result = sampler.sample(model, positive, crop, denoise=float(denoise), seed=int(seed), steps=int(steps),
                                    cfg=float(cfg), negative=negative, unique_id=unique_id)
            return result["result"][0][0]

        outputs = []
        for source in as_image_list(image):
            boxes: List[Box] = []
            for points in pose.detect_poses(source, detectors, device):
                if targets in ("faces", "faces and hands"):
                    face = face_box(points, float(keypoint_threshold))
                    if face is not None:
                        boxes.append(face)
                if targets in ("hands", "faces and hands"):
                    boxes.extend(hand_boxes(points, float(keypoint_threshold)))
            outputs.append(detail(source, boxes, redraw, size, float(padding), float(feather)))
        return (outputs,)
