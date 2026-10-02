"""Prompt encoding for SDXL, SD3 and Flux (report item B9), against tiny
random text encoders (tests/_tiny_text.py): nothing is downloaded. Where a
prompt needs nothing of ours, the result must equal the pipeline's own
encode_prompt(); weighting, long prompts and clip skip are checked for shape
and effect."""

import inspect

import pytest
import torch

import _tiny_text as tiny
from backend.nodes import generation, prompt_weights

LONG = ", ".join(f"detail{i}" for i in range(60))  # 119 tokens: two CLIP chunks
CPU = torch.device("cpu")


def _close(a, b):
    return a.shape == b.shape and torch.allclose(a.float(), b.float(), atol=1e-5)


# ---- SDXL ----


@pytest.fixture(scope="module")
def sdxl():
    return tiny.sdxl_pipe()


def test_sdxl_plain_prompt_matches_diffusers(sdxl):
    ours = prompt_weights.encode_sdxl(sdxl, "a red cat", "blurry")
    pe, npe, pooled, npooled = sdxl.encode_prompt(
        "a red cat", device=CPU, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt="blurry"
    )
    assert _close(ours["prompt_embeds"], pe) and _close(ours["negative_prompt_embeds"], npe)
    assert _close(ours["pooled_prompt_embeds"], pooled) and _close(ours["negative_pooled_prompt_embeds"], npooled)


def test_sdxl_unconnected_negative_is_zeros_like_diffusers(sdxl):
    ours = prompt_weights.encode_sdxl(sdxl, "a red cat", None)
    assert not ours["negative_prompt_embeds"].any() and not ours["negative_pooled_prompt_embeds"].any()
    # A connected (even empty) negative prompt is encoded, as in diffusers.
    assert prompt_weights.encode_sdxl(sdxl, "a red cat", "")["negative_prompt_embeds"].any()
    keep = prompt_weights.encode_sdxl(tiny.sdxl_pipe(force_zeros=False), "a red cat", None)
    assert keep["negative_prompt_embeds"].any()


def test_sdxl_long_prompt_is_chunked_not_truncated(sdxl):
    ours = prompt_weights.encode_sdxl(sdxl, LONG, "blurry")
    assert ours["prompt_embeds"].shape == (1, 154, 32 + 48)  # two 77-token chunks, CLIP-L | CLIP-G
    assert ours["negative_prompt_embeds"].shape == ours["prompt_embeds"].shape, "negative padded for CFG"
    assert ours["pooled_prompt_embeds"].shape == (1, 24)
    # The first chunk is exactly what diffusers makes of the (truncated) prompt.
    truncated, _, pooled, _ = sdxl.encode_prompt(LONG, device=CPU, do_classifier_free_guidance=False)
    assert _close(ours["prompt_embeds"][:, :77], truncated) and _close(ours["pooled_prompt_embeds"], pooled)


def test_sdxl_weighting_and_clip_skip(sdxl):
    plain = prompt_weights.encode_sdxl(sdxl, "a red cat", None)["prompt_embeds"]
    weighted = prompt_weights.encode_sdxl(sdxl, "a (red:1.5) cat", None)["prompt_embeds"]
    assert weighted.shape == plain.shape and not _close(weighted, plain)
    skipped = prompt_weights.encode_sdxl(sdxl, "a red cat", None, clip_skip=1)["prompt_embeds"]
    theirs, _, _, _ = sdxl.encode_prompt("a red cat", device=CPU, do_classifier_free_guidance=False, clip_skip=1)
    assert _close(skipped, theirs) and not _close(skipped, plain)


# ---- SD3 ----


@pytest.fixture(scope="module")
def sd3():
    return tiny.sd3_pipe()


def _sd3_reference(pipe, prompt, negative):
    return pipe.encode_prompt(
        prompt, None, None, device=CPU, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative
    )


def test_sd3_plain_prompt_matches_diffusers(sd3):
    ours = prompt_weights.encode_sd3(sd3, "a red cat", "blurry")
    pe, npe, pooled, npooled = _sd3_reference(sd3, "a red cat", "blurry")
    assert ours["prompt_embeds"].shape == (1, 77 + 256, 64)
    assert _close(ours["prompt_embeds"], pe) and _close(ours["negative_prompt_embeds"], npe)
    assert _close(ours["pooled_prompt_embeds"], pooled) and _close(ours["negative_pooled_prompt_embeds"], npooled)
    # No negative connected: SD3 encodes "" (no zeroing), like diffusers.
    assert _close(prompt_weights.encode_sd3(sd3, "a red cat", None)["negative_prompt_embeds"], _sd3_reference(sd3, "a red cat", None)[1])


def test_sd3_long_prompt_chunks_clip_and_extends_t5(sd3):
    ours = prompt_weights.encode_sd3(sd3, LONG, None)
    assert ours["prompt_embeds"].shape == (1, 2 * 77 + 256, 64)
    assert ours["negative_prompt_embeds"].shape == ours["prompt_embeds"].shape
    very_long = " ".join(f"w{i}" for i in range(300))  # > 256 T5 tokens
    assert prompt_weights.needs_custom_encoding(sd3, "sd3", [very_long])
    out = prompt_weights.encode_sd3(sd3, very_long, None)
    assert out["prompt_embeds"].shape[1] == 4 * 77 + 301
    capped = prompt_weights.encode_sd3(sd3, " ".join(f"w{i}" for i in range(700)), None)
    assert capped["prompt_embeds"].shape[1] == 10 * 77 + prompt_weights.T5_MAX_TOKENS


def test_sd3_weighting_clip_skip_and_missing_t5(sd3):
    plain = prompt_weights.encode_sd3(sd3, "a red cat", None)["prompt_embeds"]
    assert not _close(prompt_weights.encode_sd3(sd3, "a (red:1.4) cat", None)["prompt_embeds"], plain)
    skipped = prompt_weights.encode_sd3(sd3, "a red cat", None, clip_skip=1)["prompt_embeds"]
    theirs = sd3.encode_prompt("a red cat", None, None, device=CPU, do_classifier_free_guidance=False, clip_skip=1)[0]
    assert _close(skipped, theirs)
    no_t5 = prompt_weights.encode_sd3(tiny.sd3_pipe(with_t5=False), "a (red:1.2) cat", "ugly")
    assert no_t5["prompt_embeds"].shape == (1, 77 + 256, 64) and not no_t5["prompt_embeds"][:, 77:].any()


# ---- Flux ----


@pytest.fixture(scope="module")
def flux():
    return tiny.flux_pipe()


def test_flux_plain_and_long_prompts_match_diffusers(flux):
    for prompt in ("a red cat", LONG):
        ours = prompt_weights.encode_flux(flux, prompt)
        pe, pooled, _ = flux.encode_prompt(prompt, None, device=CPU)
        assert ours["prompt_embeds"].shape == (1, 512, 48)
        assert _close(ours["prompt_embeds"], pe) and _close(ours["pooled_prompt_embeds"], pooled)


def test_flux_weighting_reaches_t5(flux):
    plain = prompt_weights.encode_flux(flux, "a red cat")
    weighted = prompt_weights.encode_flux(flux, "a (red:1.5) cat")
    assert not _close(weighted["prompt_embeds"], plain["prompt_embeds"])
    assert _close(weighted["pooled_prompt_embeds"], plain["pooled_prompt_embeds"]), "pooled CLIP is unweighted"


# ---- routing in the samplers ----


def test_plain_short_prompts_stay_plain_text(sdxl, sd3, flux):
    for arch, pipe in (("sdxl", sdxl), ("sd3", sd3), ("flux", flux)):
        kwargs = generation._prompt_kwargs({"arch": arch, "pipeline": pipe}, pipe, {"text": "a cat"}, {"text": "ugly"}, 1)
        assert kwargs["prompt"] == "a cat" and kwargs["negative_prompt"] == "ugly" and "prompt_embeds" not in kwargs


@pytest.mark.parametrize(
    "arch, text, clip_skip, keys",
    [
        ("sdxl", "a (cat:1.3)", 0, {"prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"}),
        ("sdxl", LONG, 0, {"prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"}),
        ("sdxl", "a cat", 2, {"prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"}),
        ("sd3", LONG, 0, {"prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"}),
        ("flux", "a (cat:1.3)", 0, {"prompt_embeds", "pooled_prompt_embeds"}),
    ],
)
def test_weighted_long_or_skipped_prompts_become_embeddings(sdxl, sd3, flux, arch, text, clip_skip, keys):
    pipe = {"sdxl": sdxl, "sd3": sd3, "flux": flux}[arch]
    positive = {"text": text, "clip_skip": clip_skip}
    kwargs = generation._prompt_kwargs({"arch": arch, "pipeline": pipe}, pipe, positive, {"text": "[ugly]"}, 3)
    assert set(kwargs) == keys
    assert all(value.shape[0] == 3 for value in kwargs.values()), "repeated per source image"
    if "negative_prompt_embeds" in kwargs:
        assert kwargs["negative_prompt_embeds"].shape == kwargs["prompt_embeds"].shape


def test_flux_ignores_clip_skip_and_negative_weights(flux):
    kwargs = generation._prompt_kwargs({"arch": "flux"}, flux, {"text": "a cat", "clip_skip": 2}, {"text": "(ugly:1.5)"}, 1)
    assert kwargs == {"prompt": "a cat", "negative_prompt": "ugly"}


def test_unknown_architectures_get_plain_text_without_weight_syntax(sdxl):
    kwargs = generation._prompt_kwargs({"arch": "other"}, sdxl, {"text": "a (cat:1.3)", "clip_skip": 1}, {"text": "[ugly]"}, 1)
    assert kwargs == {"prompt": "a cat", "negative_prompt": "ugly", "clip_skip": 1}


@pytest.mark.parametrize(
    "class_name",
    [
        "StableDiffusionXLPipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "StableDiffusionXLInpaintPipeline",
        "StableDiffusion3Pipeline",
        "StableDiffusion3Img2ImgPipeline",
        "StableDiffusion3InpaintPipeline",
        "FluxPipeline",
        "FluxImg2ImgPipeline",
        "FluxInpaintPipeline",
    ],
)
def test_every_pipeline_class_takes_our_embedding_kwargs(class_name):
    import diffusers

    params = inspect.signature(getattr(diffusers, class_name).__call__).parameters
    needed = {"prompt_embeds", "pooled_prompt_embeds"}
    if not class_name.startswith("Flux"):
        needed |= {"negative_prompt_embeds", "negative_pooled_prompt_embeds"}
    assert needed <= set(params)


@pytest.mark.gpu
def test_encoders_run_on_cuda_like_diffusers():
    """Device handling: everything lands on the pipeline's device (and still
    matches diffusers there). Tiny encoders only - no checkpoint download."""
    sdxl = tiny.sdxl_pipe().to("cuda")
    ours = prompt_weights.encode_sdxl(sdxl, "a red cat", "blurry")
    pe = sdxl.encode_prompt("a red cat", device=torch.device("cuda"), do_classifier_free_guidance=False)[0]
    assert all(v.device.type == "cuda" for v in ours.values()) and _close(ours["prompt_embeds"], pe)
    for pipe, encode in ((tiny.sd3_pipe().to("cuda"), prompt_weights.encode_sd3), (tiny.flux_pipe().to("cuda"), prompt_weights.encode_flux)):
        out = encode(pipe, "a (red:1.2) cat, " + LONG, None)
        assert all(v.device.type == "cuda" for v in out.values())
