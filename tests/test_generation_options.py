"""Sampler options: per-image seeds and schedules (B8), prompt weighting,
long prompts and clip skip (B9), LoRAs (B6), live previews (F10) and the
KSamplerStub -> KSampler rename (B14)."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from backend.engine.graph import execute_prompt, validate_prompt
from backend.nodes import generation, latent_preview, prompt_weights

from _hub import skip_if_hub_unreachable

TINY = "hf-internal-testing/tiny-stable-diffusion-torch"


def _graph(text="a red circle", seed=7, batch_size=1, **sampler):
    return {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": text}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#000000"}},
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["clip", 0],
                "latent_image": ["latent", 0],
                "sampler": "euler",
                "seed": seed,
                "steps": 2,
                "cfg": 7.0,
                "batch_size": batch_size,
                **sampler,
            },
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "opt"}},
    }


def _run(prompt, **kwargs):
    try:
        return execute_prompt(validate_prompt(prompt), **kwargs)
    except Exception as exc:
        skip_if_hub_unreachable(exc)
        raise


def _pixels(path):
    with Image.open(path) as img:
        return list(img.convert("RGB").getdata())


# ---- B8: per-image seeds ----


def test_batch_images_get_their_own_seeds_and_reproduce_alone(output_dir):
    outputs = _run(_graph(seed=100, batch_size=3))
    assert outputs["sampler"]["seeds"] == [100, 101, 102]
    names = outputs["save"]["images"]
    with Image.open(output_dir / names[2]) as img:
        embedded = json.loads(img.text["prompt"])
    assert embedded["sampler"]["inputs"]["seed"] == 102
    assert embedded["sampler"]["inputs"]["batch_size"] == 1
    # Re-queueing image 3's embedded prompt reproduces image 3 on its own.
    rerun = _run(embedded)
    assert rerun["sampler"]["seeds"] == [102]
    a, b = _pixels(output_dir / names[2]), _pixels(output_dir / rerun["save"]["images"][0])
    diff = max(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb))
    assert diff <= 2, "batched vs single GPU kernels may differ by rounding only"


def test_internal_seed_info_is_not_written_into_the_png(output_dir):
    outputs = _run(_graph())
    with Image.open(output_dir / outputs["save"]["images"][0]) as img:
        assert "pixelflow_seed" not in img.text and "pixelflow_seed" not in img.info


def test_schedule_options_derive_from_the_checkpoints_own_config():
    from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler

    class _Pipe:
        scheduler = EulerDiscreteScheduler()

    base = dict(_Pipe.scheduler.config)
    karras = generation.make_scheduler(_Pipe, "dpm_multistep", "karras", base)
    assert isinstance(karras, DPMSolverMultistepScheduler) and karras.config.use_karras_sigmas
    trailing = generation.make_scheduler(_Pipe, "euler", "trailing", base)
    assert trailing.config.timestep_spacing == "trailing"
    # A later "normal" run starts from the original config again, not from
    # whatever the previous run swapped onto the shared pipeline.
    _Pipe.scheduler = karras
    normal = generation.make_scheduler(_Pipe, "dpm_multistep", "normal", base)
    assert not normal.config.use_karras_sigmas


def test_karras_schedule_runs_end_to_end(output_dir):
    outputs = _run(_graph(sampler="dpm_multistep", scheduler="karras"))
    assert len(outputs["save"]["images"]) == 1


# ---- B9: prompt weighting, long prompts, clip skip ----


def test_weight_syntax_is_parsed_like_a1111():
    parsed = prompt_weights.parse_prompt_weights(r"a (red:1.3) [blue] ((green)) \(lit\)")
    assert parsed == [("a ", 1.0), ("red", 1.3), (" ", 1.0), ("blue", 0.909091), (" ", 1.0), ("green", 1.21), (" (lit)", 1.0)]
    assert prompt_weights.parse_prompt_weights("(open") == [("open", 1.1)]
    assert prompt_weights.plain_text("a (red:1.3) dog") == "a red dog"
    assert prompt_weights.parse_prompt_weights("") == [("", 1.0)]


def test_chunking_keeps_every_token():
    chunks = prompt_weights.chunk_tokens(list(range(160)), [1.0] * 160, 75)
    assert [len(ids) for ids, _ in chunks] == [75, 75, 10]
    assert prompt_weights.chunk_tokens([], [], 75) == [([], [])]


@pytest.fixture
def tiny_pipe():
    from backend.nodes import io_nodes

    try:
        return io_nodes._load_pipeline(TINY)
    except Exception as exc:
        skip_if_hub_unreachable(exc)
        raise


def test_long_prompts_are_encoded_in_chunks_not_truncated(tiny_pipe):
    long = " ".join(["red circle"] * 60)
    assert prompt_weights.needs_embeddings(tiny_pipe.tokenizer, long)
    positive, negative = prompt_weights.encode_prompt_pair(tiny_pipe, long, "blurry", 0)
    assert positive.shape[1] > 77 and positive.shape[1] % 77 == 0
    assert negative.shape == positive.shape, "negative padded to the same length for CFG"


def test_unweighted_short_prompt_matches_diffusers_own_encoding(tiny_pipe):
    ours, _ = prompt_weights.encode_prompt_pair(tiny_pipe, "a cat", "", 0)
    theirs = tiny_pipe.encode_prompt("a cat", tiny_pipe.device, 1, False)[0]
    assert torch.allclose(ours.float(), theirs.float(), atol=1e-3)
    weighted, _ = prompt_weights.encode_prompt_pair(tiny_pipe, "a (cat:1.5)", "", 0)
    skipped, _ = prompt_weights.encode_prompt_pair(tiny_pipe, "a cat", "", 1)
    assert not torch.allclose(ours, weighted) and not torch.allclose(ours, skipped)


def test_weighted_long_prompt_with_clip_skip_samples(output_dir):
    graph = _graph(text="a (red:1.4) circle, " + ", ".join(["detailed"] * 90))
    graph["clip"]["inputs"]["clip_skip"] = 1
    graph["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "[blurry]"}}
    graph["sampler"]["inputs"]["negative"] = ["neg", 0]
    outputs = _run(graph)
    assert (output_dir / outputs["save"]["images"][0]).is_file()


def test_other_architectures_get_plain_text_without_weight_syntax():
    class _Pipe:
        tokenizer = object()
        tokenizer_2 = object()  # SDXL-style: two encoders
        text_encoder = object()

    kwargs = generation._prompt_kwargs({"arch": "sdxl"}, _Pipe(), {"text": "a (cat:1.3)"}, {"text": "[ugly]"}, 1)
    assert kwargs["prompt"] == "a cat" and kwargs["negative_prompt"] == "ugly"


# ---- B6: LoRAs ----


class _LoraPipe:
    def __init__(self):
        self.calls = []

    def load_lora_weights(self, path, adapter_name):
        self.calls.append(("load", Path(path).name, adapter_name))

    def set_adapters(self, names, adapter_weights):
        self.calls.append(("set", list(names), list(adapter_weights)))

    def enable_lora(self):
        self.calls.append(("enable",))

    def disable_lora(self):
        self.calls.append(("disable",))


@pytest.fixture
def lora_files(isolated_data_dirs, monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "peft", types.ModuleType("peft"))  # the real one isn't required here
    folder = isolated_data_dirs["loras"]
    folder.mkdir(parents=True)
    for name in ("style.safetensors", "detail.safetensors"):
        (folder / name).write_bytes(b"x")
    return folder


def test_lora_loader_chains_without_touching_the_pipeline(lora_files):
    pipe = _LoraPipe()
    model = {"name": "m", "pipeline": pipe, "loras": []}
    (one,) = generation.LoraLoader().apply(model, "style.safetensors", 0.8)
    (two,) = generation.LoraLoader().apply(one, "detail.safetensors", 0.5)
    assert two["loras"] == [("style.safetensors", 0.8), ("detail.safetensors", 0.5)]
    assert model["loras"] == [] and pipe.calls == []
    with pytest.raises(FileNotFoundError):
        generation.LoraLoader().apply(model, "missing.safetensors", 1.0)


def test_loras_are_loaded_once_and_switched_per_run(lora_files):
    pipe = _LoraPipe()
    generation.apply_loras(pipe, [("style.safetensors", 0.8)])
    generation.apply_loras(pipe, [("style.safetensors", 0.5), ("detail.safetensors", 1.0)])
    generation.apply_loras(pipe, [])
    loads = [c for c in pipe.calls if c[0] == "load"]
    assert [c[1] for c in loads] == ["style.safetensors", "detail.safetensors"], "each file loaded once"
    sets = [c for c in pipe.calls if c[0] == "set"]
    assert sets[0][2] == [0.8] and sets[1][2] == [0.5, 1.0]
    assert pipe.calls[-1] == ("disable",), "a run without LoRAs switches them all off"


def test_lora_without_peft_is_a_clear_error(isolated_data_dirs, monkeypatch):
    import sys

    folder = isolated_data_dirs["loras"]
    folder.mkdir(parents=True)
    (folder / "style.safetensors").write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "peft", None)
    with pytest.raises(RuntimeError, match="peft"):
        generation.apply_loras(_LoraPipe(), [("style.safetensors", 1.0)])


def test_lora_node_lists_models_loras(lora_files):
    spec = generation.LoraLoader.INPUT_TYPES()["required"]["lora_name"]
    assert spec[0] == ["detail.safetensors", "style.safetensors"]


# ---- F10: live previews and progress detail ----


def test_latent_preview_projects_four_channel_latents():
    image = latent_preview.latents_to_image(torch.zeros(1, 4, 8, 12))
    assert image.mode == "RGB" and max(image.size) == latent_preview.MAX_PREVIEW_SIDE
    assert latent_preview.latents_to_image(torch.zeros(1, 16, 8, 8)) is None
    assert latent_preview.to_data_url(image).startswith("data:image/jpeg;base64,")


def test_progress_detail_carries_previews_and_speed(output_dir, monkeypatch):
    monkeypatch.setattr(generation, "PREVIEW_INTERVAL_S", 0.0)
    events = []
    _run(_graph(), on_progress_detail=lambda *a: events.append(a))
    assert events and all(node == "sampler" for node, *_ in events)
    assert any("preview" in info for *_, info in events), "previews only when on_progress_detail asks"
    assert any("it_per_s" in info and "eta_s" in info for *_, info in events)


def test_plain_on_progress_gets_no_previews(output_dir):
    calls = []
    _run(_graph(), on_progress=lambda node_id, step, total: calls.append((node_id, step, total)))
    assert calls[-1] == ("sampler", 2, 2)


# ---- B14: renamed class keeps old prompts working ----


def test_old_ksamplerstub_prompts_still_validate_and_run(output_dir):
    graph = _graph()
    graph["sampler"]["class_type"] = "KSamplerStub"
    assert validate_prompt(graph)["sampler"]["class_type"] == "KSampler"
    outputs = _run(graph)
    assert outputs["sampler"]["seed"] == 7
