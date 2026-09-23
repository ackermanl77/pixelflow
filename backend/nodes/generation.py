import torch

from .base import NodeBase, register_node


@register_node("CLIP Text Encode")
class CLIPTextEncode(NodeBase):
    """Wraps the prompt text as CONDITIONING. There's no separate embedding
    step here: KSampler passes this text straight to the loaded pipeline's
    own `prompt=` argument, which does its own tokenization/CLIP-encoding
    internally. This node exists so the graph keeps the familiar
    checkpoint -> conditioning -> sampler shape.
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


@register_node("KSampler")
class KSamplerStub(NodeBase):
    """Runs real Stable Diffusion sampling via the diffusers pipeline that
    LoadCheckpoint attaches to `model`. `latent_image` only supplies the
    output width/height (it's a plain RGB PIL.Image, not a real VAE latent),
    so this is txt2img-only for now - no image-to-image or inpainting.
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
        pipe = model["pipeline"]
        text = positive.get("text", "")
        width, height = latent_image.size
        # SD pipelines require dimensions that are multiples of 8.
        width = max(8, (width // 8) * 8)
        height = max(8, (height // 8) * 8)

        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
        result = pipe(
            prompt=text,
            width=width,
            height=height,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            generator=generator,
        )
        return (result.images[0],)
