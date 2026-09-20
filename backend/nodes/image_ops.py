from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .base import NodeBase, register_node


@register_node("Resize Image")
class ImageResize(NodeBase):
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096}),
            }
        }

    def run(self, image, width, height):
        return (image.resize((int(width), int(height)), Image.LANCZOS),)


@register_node("Blur")
class ImageBlur(NodeBase):
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0}),
            }
        }

    def run(self, image, radius):
        return (image.filter(ImageFilter.GaussianBlur(radius=float(radius))),)


@register_node("Brightness / Contrast")
class ImageBrightnessContrast(NodeBase):
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
            }
        }

    def run(self, image, brightness, contrast):
        out = ImageEnhance.Brightness(image).enhance(float(brightness))
        out = ImageEnhance.Contrast(out).enhance(float(contrast))
        return (out,)


@register_node("Hue / Saturation")
class ImageHueSaturation(NodeBase):
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue_shift_degrees": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
            }
        }

    def run(self, image, hue_shift_degrees, saturation):
        hsv = image.convert("HSV")
        h, s, v = hsv.split()
        shift = int((float(hue_shift_degrees) / 360.0) * 255)
        h = h.point(lambda p: (p + shift) % 256)
        s = s.point(lambda p: max(0, min(255, int(p * float(saturation)))))
        out = Image.merge("HSV", (h, s, v)).convert("RGB")
        return (out,)


@register_node("Blend")
class ImageBlend(NodeBase):
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("IMAGE",),
                "overlay": ("IMAGE",),
                "mode": (["normal", "multiply", "screen"], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    def run(self, base, overlay, mode, opacity):
        base = base.convert("RGB")
        overlay = overlay.resize(base.size).convert("RGB")
        if mode == "multiply":
            blended = ImageChops.multiply(base, overlay)
        elif mode == "screen":
            blended = ImageChops.screen(base, overlay)
        else:
            blended = overlay
        out = Image.blend(base, blended, float(opacity))
        return (out,)


@register_node("Text Overlay")
class ImageTextOverlay(NodeBase):
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": "Hello", "multiline": True}),
                "x": ("INT", {"default": 10, "min": 0, "max": 4096}),
                "y": ("INT", {"default": 10, "min": 0, "max": 4096}),
                "size": ("INT", {"default": 32, "min": 6, "max": 300}),
                "color": ("STRING", {"default": "#ffffff"}),
            }
        }

    def run(self, image, text, x, y, size, color):
        out = image.copy()
        draw = ImageDraw.Draw(out)
        try:
            font = ImageFont.truetype("arial.ttf", int(size))
        except OSError:
            font = ImageFont.load_default()
        draw.text((int(x), int(y)), text, fill=color, font=font)
        return (out,)
