import colorsys
import hashlib

from PIL import Image, ImageDraw, ImageFont

from .base import NodeBase, register_node


@register_node("CLIP Text Encode (Stub)")
class CLIPTextEncode(NodeBase):
    """Stands in for a real CLIP text encoder. A real implementation would
    tokenize `text` and run it through the checkpoint's text encoder to
    produce an embedding tensor instead of passing the raw string through.
    """

    CATEGORY = "conditioning"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": "", "multiline": True})}}

    def encode(self, text):
        return ({"text": text},)


@register_node("KSampler (Stub)")
class KSamplerStub(NodeBase):
    """Deterministic placeholder for real diffusion sampling, so the graph
    shape (checkpoint -> conditioning -> sampler -> image) matches a real
    text-to-image pipeline without needing a GPU.

    To wire up real generation once a GPU is available, replace the body of
    `sample()` with something like:

        import torch
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(model["name"]).to("cuda")
        image = pipe(
            positive["text"], num_inference_steps=steps, guidance_scale=cfg,
            generator=torch.manual_seed(seed),
        ).images[0]
        return (image,)
    """

    CATEGORY = "sampling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0}),
            }
        }

    def sample(self, model, positive, latent_image, seed, steps, cfg):
        text = positive.get("text", "")
        width, height = latent_image.size
        digest = hashlib.sha256(f"{text}|{seed}".encode()).digest()
        hue = digest[0] / 255.0
        r1, g1, b1 = (int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.55, 0.95))
        r2, g2, b2 = (int(c * 255) for c in colorsys.hsv_to_rgb((hue + 0.5) % 1.0, 0.6, 0.6))

        img = Image.new("RGB", (width, height))
        pixels = img.load()
        for y in range(height):
            t = y / max(height - 1, 1)
            row = (int(r1 + (r2 - r1) * t), int(g1 + (g2 - g1) * t), int(b1 + (b2 - b1) * t))
            for x in range(width):
                pixels[x, y] = row

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", max(12, width // 24))
        except OSError:
            font = ImageFont.load_default()
        label = text.strip() or "(empty prompt)"
        draw.text(
            (12, 12),
            f"[stub render]\n{label}\nsteps={steps} cfg={cfg} seed={seed}",
            fill=(255, 255, 255),
            font=font,
        )
        return (img,)
