"""Cheap live previews of the latents during sampling (report item F10).

A fixed linear projection from the latent channels to RGB (ComfyUI's
"latent2rgb" factors) - no VAE decode, so it costs next to nothing and runs
every few steps. SD 1.x/2.x and SDXL have 4 latent channels, SD3 and Flux 16;
Flux's latents arrive packed as 2x2 patches (B, tokens, 64) and are unpacked
with the image size first."""

import base64
import io

# Rows: latent channels; columns: R, G, B.
_FACTORS = {
    "sd": ([[0.3512, 0.2297, 0.3227], [0.3250, 0.4974, 0.2350], [-0.2829, 0.1762, 0.2721], [-0.2120, -0.2616, -0.7177]], [0.0, 0.0, 0.0]),
    "sdxl": ([[0.3651, 0.4232, 0.4341], [-0.2533, -0.0042, 0.1068], [0.1076, 0.1111, -0.0362], [-0.3165, -0.2492, -0.2188]], [0.1084, -0.0175, -0.0011]),
    "sd3": (
        [
            [-0.0922, -0.0175, 0.0749], [0.0311, 0.0633, 0.0954], [0.1994, 0.0927, 0.0458], [0.0856, 0.0339, 0.0902],
            [0.0587, 0.0272, -0.0496], [-0.0006, 0.1104, 0.0309], [0.0978, 0.0306, 0.0427], [-0.0042, 0.1038, 0.1358],
            [-0.0194, 0.0020, 0.0669], [-0.0488, 0.0130, -0.0268], [0.0922, 0.0988, 0.0951], [-0.0278, 0.0524, -0.0542],
            [0.0332, 0.0456, 0.0895], [-0.0069, -0.0030, -0.0810], [-0.0596, -0.0465, -0.0293], [-0.1448, -0.1463, -0.1189],
        ],
        [0.2394, 0.2135, 0.1925],
    ),
    "flux": (
        [
            [-0.0346, 0.0244, 0.0681], [0.0034, 0.0210, 0.0687], [0.0275, -0.0668, -0.0433], [-0.0174, 0.0160, 0.0617],
            [0.0859, 0.0721, 0.0329], [0.0004, 0.0383, 0.0115], [0.0405, 0.0861, 0.0915], [-0.0236, -0.0185, -0.0259],
            [-0.0245, 0.0250, 0.1180], [0.1008, 0.0755, -0.0421], [-0.0515, 0.0201, 0.0011], [0.0428, -0.0012, -0.0036],
            [0.0817, 0.0765, 0.0749], [-0.1264, -0.0522, -0.1103], [-0.0280, -0.0881, -0.0499], [-0.1262, -0.0982, -0.0778],
        ],
        [-0.0329, -0.0718, -0.0851],
    ),
}
# The 16-channel factors map straight to 0..1 (the 4-channel ones to -1..1).
_UNIT_RANGE = {"sd3", "flux"}
# Flux's VAE downscales 8x and the transformer packs 2x2 latent patches.
FLUX_PATCH_SCALE = 16


def unpack_flux_latents(latents, width, height):
    """(B, tokens, 64) packed Flux latents -> (B, 16, h, w), as diffusers'
    FluxPipeline._unpack_latents does; None when the sizes don't fit."""
    if latents is None or latents.ndim != 3 or latents.shape[-1] % 4 or not width or not height:
        return None
    rows, cols = int(height) // FLUX_PATCH_SCALE, int(width) // FLUX_PATCH_SCALE
    batch, tokens, channels = latents.shape
    if rows * cols != tokens:
        return None
    unpacked = latents.view(batch, rows, cols, channels // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return unpacked.reshape(batch, channels // 4, rows * 2, cols * 2)

MAX_PREVIEW_SIDE = 256


def latents_to_image(latents, arch: str = "sd", width: int | None = None, height: int | None = None):
    """A PIL preview of the first latent of a batch, or None when the
    latent layout isn't one this projection knows. `width`/`height` (the
    image size) are needed only to unpack Flux's latents."""
    import torch
    from PIL import Image

    if latents is None:
        return None
    if latents.ndim == 3:
        latents = unpack_flux_latents(latents, width, height)
        arch = "flux"
        if latents is None:
            return None
    if latents.ndim != 4:
        return None
    channels = latents.shape[1]
    if channels == 16 and arch not in _UNIT_RANGE:
        arch = "sd3"
    elif channels == 4 and arch in _UNIT_RANGE:
        arch = "sd"
    factors = _FACTORS.get(arch) or _FACTORS["sd"]
    if channels != len(factors[0]):
        return None
    with torch.no_grad():
        sample = latents[0].detach().float().cpu()
        matrix = torch.tensor(factors[0])
        bias = torch.tensor(factors[1])
        rgb = torch.einsum("chw,cr->hwr", sample, matrix) + bias
        if arch not in _UNIT_RANGE:
            rgb = (rgb + 1.0) / 2.0
        rgb = rgb.clamp(0, 1).mul(255).byte().numpy()
    image = Image.fromarray(rgb)
    scale = MAX_PREVIEW_SIDE / max(image.size)
    if scale > 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.NEAREST)
    return image


def to_data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
