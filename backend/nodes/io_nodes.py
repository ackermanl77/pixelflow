import uuid

from PIL import Image, ImageColor

from .base import NodeBase, register_node
from ..paths import OUTPUT_DIR, UPLOAD_DIR


@register_node("Load Checkpoint (Stub)")
class LoadCheckpoint(NodeBase):
    """Stands in for a real checkpoint loader, e.g.
    diffusers.StableDiffusionPipeline.from_pretrained(ckpt_name).
    """

    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": ("STRING", {"default": "stub-diffusion-v1"})}}

    def load(self, ckpt_name):
        return ({"name": ckpt_name},)


@register_node("Empty Image")
class EmptyImage(NodeBase):
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "color": ("STRING", {"default": "#202020"}),
            }
        }

    def create(self, width, height, color):
        rgb = ImageColor.getrgb(color)
        return (Image.new("RGB", (int(width), int(height)), rgb),)


@register_node("Load Image")
class LoadImage(NodeBase):
    CATEGORY = "loaders"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filename": ("STRING", {"default": ""})}}

    def load(self, filename):
        path = UPLOAD_DIR / filename
        return (Image.open(path).convert("RGB"),)


def _save(image, prefix, folder):
    folder.mkdir(exist_ok=True)
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    image.save(folder / name)
    return name


@register_node("Preview Image")
class PreviewImage(NodeBase):
    CATEGORY = "output"
    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def preview(self, image):
        name = _save(image, "preview", OUTPUT_DIR)
        return {"images": [name]}


@register_node("Save Image")
class SaveImage(NodeBase):
    CATEGORY = "output"
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "PixelFlow"}),
            }
        }

    def save(self, image, filename_prefix):
        name = _save(image, filename_prefix, OUTPUT_DIR)
        return {"images": [name]}
