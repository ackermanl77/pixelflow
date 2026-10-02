"""Cheap live previews of the latents during sampling (report item F10).

A fixed linear projection from the 4 latent channels to RGB (ComfyUI's
"latent2rgb" factors) - no VAE decode, so it costs next to nothing and runs
every few steps. Only 4-channel latents (SD 1.x/2.x, SDXL) are previewed."""

import base64
import io

# Rows: latent channels; columns: R, G, B.
_FACTORS = {
    "sd": ([[0.3512, 0.2297, 0.3227], [0.3250, 0.4974, 0.2350], [-0.2829, 0.1762, 0.2721], [-0.2120, -0.2616, -0.7177]], [0.0, 0.0, 0.0]),
    "sdxl": ([[0.3651, 0.4232, 0.4341], [-0.2533, -0.0042, 0.1068], [0.1076, 0.1111, -0.0362], [-0.3165, -0.2492, -0.2188]], [0.1084, -0.0175, -0.0011]),
}

MAX_PREVIEW_SIDE = 256


def latents_to_image(latents, arch: str = "sd"):
    """A PIL preview of the first latent of a batch, or None when the
    latent layout isn't one this projection knows."""
    import torch
    from PIL import Image

    factors = _FACTORS.get(arch) or _FACTORS["sd"]
    if latents is None or latents.ndim != 4 or latents.shape[1] != 4:
        return None
    with torch.no_grad():
        sample = latents[0].detach().float().cpu()
        matrix = torch.tensor(factors[0])
        bias = torch.tensor(factors[1])
        rgb = torch.einsum("chw,cr->hwr", sample, matrix) + bias
        rgb = ((rgb + 1.0) / 2.0).clamp(0, 1).mul(255).byte().numpy()
    image = Image.fromarray(rgb)
    scale = MAX_PREVIEW_SIDE / max(image.size)
    if scale > 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.NEAREST)
    return image


def to_data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
