from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .base import NodeBase, map_image_pairs, map_images, register_node


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
        size = (int(width), int(height))
        return (map_images(image, lambda img: img.resize(size, Image.LANCZOS)),)


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
        blur = ImageFilter.GaussianBlur(radius=float(radius))
        return (map_images(image, lambda img: img.filter(blur)),)


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
        def _adjust(img):
            out = ImageEnhance.Brightness(img).enhance(float(brightness))
            return ImageEnhance.Contrast(out).enhance(float(contrast))

        return (map_images(image, _adjust),)


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
        shift = int((float(hue_shift_degrees) / 360.0) * 255)

        def _adjust(img):
            h, s, v = img.convert("HSV").split()
            h = h.point(lambda p: (p + shift) % 256)
            s = s.point(lambda p: max(0, min(255, int(p * float(saturation)))))
            return Image.merge("HSV", (h, s, v)).convert("RGB")

        return (map_images(image, _adjust),)


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
        def _blend(base_img, overlay_img):
            base_img = base_img.convert("RGB")
            overlay_img = overlay_img.resize(base_img.size).convert("RGB")
            if mode == "multiply":
                blended = ImageChops.multiply(base_img, overlay_img)
            elif mode == "screen":
                blended = ImageChops.screen(base_img, overlay_img)
            else:
                blended = overlay_img
            return Image.blend(base_img, blended, float(opacity))

        return (map_image_pairs(base, overlay, _blend),)


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
                "color": ("STRING", {"default": "#ffffff", "tooltip": "Any CSS-style colour"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, color=None, **_):
        if color is None:
            return True
        try:
            ImageColor.getrgb(color)
        except (ValueError, AttributeError):
            return f"'{color}' is not a colour (use e.g. #ffffff, rgb(255,255,255) or a name like 'white')"
        return True

    def run(self, image, text, x, y, size, color):
        try:
            font = ImageFont.truetype("arial.ttf", int(size))
        except OSError:
            font = ImageFont.load_default()

        def _draw(img):
            out = img.copy()
            ImageDraw.Draw(out).text((int(x), int(y)), text, fill=color, font=font)
            return out

        return (map_images(image, _draw),)



@register_node("Canny Edges")
class CannyEdges(NodeBase):
    """Canny edge detection: white edges on black, the control image a Canny
    ControlNet expects (B7). `low`/`high` are the hysteresis thresholds on
    the Sobel gradient strength, on OpenCV's Canny scale (the usual ControlNet
    preprocessor setting is 100/200): edges above `high` are kept, and so are
    weaker ones above `low` that connect to them. Lower values keep more
    detail."""

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    DESCRIPTION = "White-on-black edge map (for a Canny ControlNet)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "low": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "high": ("INT", {"default": 200, "min": 1, "max": 1000}),
            }
        }

    def run(self, image, low=100, high=200):
        if int(low) > int(high):
            raise ValueError(f"low ({int(low)}) must not be above high ({int(high)})")
        return (map_images(image, lambda img: canny(img, int(low), int(high))),)


def canny(image, low: int, high: int):
    """Canny on `image`'s luminance: Gaussian blur, Sobel gradients,
    non-maximum suppression along the gradient direction, double threshold
    and hysteresis (weak edges kept where they touch strong ones)."""
    import numpy as np

    gray = np.asarray(image.convert("L").filter(ImageFilter.GaussianBlur(1.4)), dtype=np.float32)
    padded = np.pad(gray, 1, mode="edge")
    gx = (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]) - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    gy = (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]) - (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
    magnitude = np.hypot(gx, gy)  # raw Sobel magnitude, the scale cv2.Canny thresholds use
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    m = np.pad(magnitude, 1)
    neighbours = {  # the two neighbours along each quantized gradient direction
        0: (m[1:-1, :-2], m[1:-1, 2:]),
        45: (m[:-2, 2:], m[2:, :-2]),
        90: (m[:-2, 1:-1], m[2:, 1:-1]),
        135: (m[:-2, :-2], m[2:, 2:]),
    }
    direction = (np.round(angle / 45.0) % 4 * 45).astype(int)
    keep = np.zeros_like(magnitude, dtype=bool)
    for value, (a, b) in neighbours.items():
        here = direction == value
        keep |= here & (magnitude > a) & (magnitude >= b)  # ">" on one side: a tie keeps one pixel
    thin = np.where(keep, magnitude, 0.0)
    strong = thin >= high
    weak = thin >= low
    edges = strong.copy()
    for _ in range(max(gray.shape)):  # grow strong edges through connected weak pixels
        grown = np.pad(edges, 1)
        reach = np.zeros_like(edges)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                reach |= grown[dy:dy + edges.shape[0], dx:dx + edges.shape[1]]
        updated = weak & reach
        if np.array_equal(updated, edges):
            break
        edges = updated
    return Image.fromarray((edges * 255).astype(np.uint8)).convert("RGB")
