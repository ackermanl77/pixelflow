import pytest
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from requests.exceptions import ConnectionError as RequestsConnectionError

from backend.engine.graph import GraphError, execute_prompt

# A real, randomly-initialized-weights Stable Diffusion pipeline published by
# the diffusers team specifically for exercising the real pipeline code path
# in tests without downloading a multi-GB production checkpoint.
TINY_SD_CHECKPOINT = "hf-internal-testing/tiny-stable-diffusion-torch"


def test_execute_prompt_runs_save_image(output_dir):
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#ff0000"}},
        "2": {"class_type": "SaveImage", "inputs": {"image": ["1", 0], "filename_prefix": "test"}},
    }
    outputs = execute_prompt(prompt)
    assert "2" in outputs
    filename = outputs["2"]["images"][0]
    assert (output_dir / filename).exists()


def test_execute_prompt_runs_a_real_diffusion_pipeline(output_dir):
    prompt = {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY_SD_CHECKPOINT}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red circle"}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "color": "#000000"}},
        "sampler": {
            "class_type": "KSamplerStub",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["clip", 0],
                "latent_image": ["latent", 0],
                "seed": 42,
                "steps": 2,
                "cfg": 7.0,
            },
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "pipeline"}},
    }
    try:
        outputs = execute_prompt(prompt)
    except (HfHubHTTPError, RequestsConnectionError) as exc:
        pytest.skip(f"Hugging Face Hub unreachable: {exc}")
    filename = outputs["save"]["images"][0]
    assert (output_dir / filename).exists()


def _run_or_skip(prompt):
    """Run a prompt, skipping the test (rather than failing) if the tiny
    checkpoint can't be fetched from the Hugging Face Hub - the same
    fallback test_execute_prompt_runs_a_real_diffusion_pipeline uses."""
    try:
        return execute_prompt(prompt)
    except (HfHubHTTPError, RequestsConnectionError) as exc:
        pytest.skip(f"Hugging Face Hub unreachable: {exc}")


def test_ksampler_same_seed_is_reproducible(output_dir):
    """A given (non-random) seed must reproduce the same image, and the
    chosen `sampler` must actually be applied to the pipeline."""

    def _prompt(seed):
        return {
            "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY_SD_CHECKPOINT}},
            "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red circle"}},
            "latent": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "color": "#000000"}},
            "sampler": {
                "class_type": "KSamplerStub",
                "inputs": {
                    "model": ["ckpt", 0],
                    "positive": ["clip", 0],
                    "latent_image": ["latent", 0],
                    "sampler": "dpm_multistep",
                    "seed": seed,
                    "steps": 2,
                    "cfg": 7.0,
                },
            },
            "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "seeded"}},
        }

    outputs_a = _run_or_skip(_prompt(123))
    outputs_b = _run_or_skip(_prompt(123))
    outputs_c = _run_or_skip(_prompt(456))

    def _pixels(outputs):
        filename = outputs["save"]["images"][0]
        with Image.open(output_dir / filename) as img:
            return list(img.convert("RGB").getdata())

    pixels_a = _pixels(outputs_a)
    pixels_b = _pixels(outputs_b)
    pixels_c = _pixels(outputs_c)

    assert pixels_a == pixels_b, "same seed + sampler must reproduce the same image"
    assert pixels_a != pixels_c, "different seeds should (almost certainly) differ"


def test_ksampler_negative_prompt_changes_output(output_dir):
    """Wiring a second CLIPTextEncode into KSampler's `negative` input must
    actually reach the pipeline (different negative prompt -> different
    image for the same seed)."""

    def _prompt(negative_text):
        nodes = {
            "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY_SD_CHECKPOINT}},
            "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red circle"}},
            "latent": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "color": "#000000"}},
            "sampler": {
                "class_type": "KSamplerStub",
                "inputs": {
                    "model": ["ckpt", 0],
                    "positive": ["clip", 0],
                    "latent_image": ["latent", 0],
                    "seed": 7,
                    "steps": 2,
                    "cfg": 7.0,
                },
            },
            "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "neg"}},
        }
        if negative_text is not None:
            nodes["clip_neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_text}}
            nodes["sampler"]["inputs"]["negative"] = ["clip_neg", 0]
        return nodes

    outputs_plain = _run_or_skip(_prompt(None))
    outputs_negative = _run_or_skip(_prompt("blurry, low quality"))

    def _pixels(outputs):
        filename = outputs["save"]["images"][0]
        with Image.open(output_dir / filename) as img:
            return list(img.convert("RGB").getdata())

    assert _pixels(outputs_plain) != _pixels(outputs_negative), (
        "a wired negative prompt should change the pipeline's output"
    )


def test_ksampler_batch_size_produces_multiple_images(output_dir):
    prompt = {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY_SD_CHECKPOINT}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red circle"}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "color": "#000000"}},
        "sampler": {
            "class_type": "KSamplerStub",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["clip", 0],
                "latent_image": ["latent", 0],
                "seed": 99,
                "steps": 2,
                "cfg": 7.0,
                "batch_size": 3,
            },
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "batch"}},
    }
    outputs = _run_or_skip(prompt)
    filenames = outputs["save"]["images"]
    assert len(filenames) == 3
    for filename in filenames:
        assert (output_dir / filename).exists()
    assert len(set(filenames)) == 3, "each batch image must get its own saved filename"


def test_preview_image_and_save_image_still_accept_a_single_pil_image(output_dir):
    """Non-batch nodes (EmptyImage etc.) still hand SaveImage/PreviewImage a
    single PIL.Image directly rather than a list - this must keep working
    after adding batch (list-of-images) support for KSampler's output."""
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 16, "height": 16, "color": "#00ff00"}},
        "preview": {"class_type": "PreviewImage", "inputs": {"image": ["1", 0]}},
    }
    outputs = execute_prompt(prompt)
    filenames = outputs["preview"]["images"]
    assert len(filenames) == 1
    assert (output_dir / filenames[0]).exists()


def test_ksampler_reports_step_progress_via_on_progress_callback():
    """KSampler must call execute_prompt's on_progress with (node_id, step,
    total) once per diffusion step, via diffusers' callback_on_step_end."""
    prompt = {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY_SD_CHECKPOINT}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red circle"}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "color": "#000000"}},
        "sampler": {
            "class_type": "KSamplerStub",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["clip", 0],
                "latent_image": ["latent", 0],
                "seed": 1,
                "steps": 3,
                "cfg": 7.0,
            },
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "progress"}},
    }
    progress_calls = []

    def on_progress(node_id, step, total):
        progress_calls.append((node_id, step, total))

    try:
        execute_prompt(prompt, on_progress=on_progress)
    except (HfHubHTTPError, RequestsConnectionError) as exc:
        pytest.skip(f"Hugging Face Hub unreachable: {exc}")

    assert progress_calls, "expected at least one progress callback during sampling"
    assert all(node_id == "sampler" for node_id, _, _ in progress_calls)
    assert all(total == 3 for _, _, total in progress_calls)
    steps_seen = [step for _, step, _ in progress_calls]
    assert steps_seen == sorted(steps_seen), "steps should be reported in increasing order"
    assert max(steps_seen) <= 3
    assert min(steps_seen) >= 1, "steps should be reported 1-based"


def test_on_progress_is_not_passed_to_nodes_that_do_not_accept_it():
    """A node whose FUNCTION has no `on_progress` parameter (e.g. EmptyImage)
    must run fine even when execute_prompt is given an on_progress callback -
    the engine should only inject it into nodes that declare it."""
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "color": "#000000"}},
        "2": {"class_type": "SaveImage", "inputs": {"image": ["1", 0], "filename_prefix": "no_progress"}},
    }
    outputs = execute_prompt(prompt, on_progress=lambda node_id, step, total: None)
    assert "2" in outputs


def test_execute_prompt_detects_cycle():
    prompt = {
        "a": {"class_type": "EmptyImage", "inputs": {"width": ["b", 0], "height": 8, "color": "#000000"}},
        "b": {"class_type": "EmptyImage", "inputs": {"width": ["a", 0], "height": 8, "color": "#000000"}},
    }
    with pytest.raises(GraphError, match="Cycle detected"):
        execute_prompt(prompt)


def test_execute_prompt_rejects_unknown_class_type():
    prompt = {"1": {"class_type": "NotARealNode", "inputs": {}}}
    with pytest.raises(GraphError, match="Unknown node class"):
        execute_prompt(prompt)


def test_execute_prompt_rejects_missing_referenced_node():
    prompt = {
        "1": {"class_type": "SaveImage", "inputs": {"image": ["missing", 0], "filename_prefix": "x"}},
    }
    with pytest.raises(GraphError, match="not present in the prompt"):
        execute_prompt(prompt)
