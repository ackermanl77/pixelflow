"""Wildcards in prompts, resolved per seed (report item F21)."""

import json

import pytest
from PIL import Image

from backend.engine.graph import execute_prompt, validate_prompt
from backend.nodes import generation, wildcards

from _hub import skip_if_hub_unreachable

TINY = "hf-internal-testing/tiny-stable-diffusion-torch"


@pytest.fixture
def wildcard_dir(isolated_data_dirs):
    folder = isolated_data_dirs["wildcards"]
    folder.mkdir()
    return folder


def test_plain_prompts_and_literal_braces_are_left_alone():
    for text in ("a cat", "a {single} word", "(red:1.2) [blue]", "", None):
        assert not wildcards.has_wildcards(text)
        assert wildcards.resolve(text, 5) == text


def test_alternatives_are_chosen_per_seed_and_reproducibly():
    text = "a {red|green|blue|yellow} {cat|dog|fox} on a {hill|beach}"
    picks = {wildcards.resolve(text, seed) for seed in range(40)}
    assert len(picks) > 5, "different seeds pick different alternatives"
    assert all("{" not in pick and "|" not in pick for pick in picks)
    assert wildcards.resolve(text, 17) == wildcards.resolve(text, 17)


def test_nested_and_empty_alternatives():
    seen = {wildcards.resolve("x{a|{b|c}|}y", seed) for seed in range(200)}
    assert seen == {"xay", "xby", "xcy", "xy"}


def test_file_wildcards_read_one_line_skipping_blanks_and_comments(wildcard_dir):
    (wildcard_dir / "animal.txt").write_text("# animals\ncat\n\n  dog  \n", encoding="utf-8")
    (wildcard_dir / "colors").mkdir()
    (wildcard_dir / "colors" / "warm.txt").write_text("\ufeffred\norange __animal__\n", encoding="utf-8")
    seen = {wildcards.resolve("a __colors/warm__", seed) for seed in range(100)}
    assert seen == {"a red", "a orange cat", "a orange dog"}
    assert wildcards.referenced_files("__colors/warm__ and __animal__ and __animal__") == ["animal", "colors/warm"]


def test_missing_empty_and_self_referencing_files_are_clear_errors(wildcard_dir):
    with pytest.raises(wildcards.WildcardError, match="no wildcard file nope.txt"):
        wildcards.resolve("__nope__", 1)
    (wildcard_dir / "empty.txt").write_text("# nothing\n", encoding="utf-8")
    with pytest.raises(wildcards.WildcardError, match="has no options"):
        wildcards.resolve("__empty__", 1)
    (wildcard_dir / "loop.txt").write_text("again __loop__\n", encoding="utf-8")
    with pytest.raises(wildcards.WildcardError, match="unresolved"):
        wildcards.resolve("__loop__", 1)


def test_names_cannot_leave_the_wildcard_folder():
    assert not wildcards.has_wildcards("__../secrets__")
    assert not wildcards.has_wildcards("__C:/x__")


def test_positive_and_negative_choose_independently():
    text = "{a|b|c|d|e|f|g|h}"
    same = sum(wildcards.resolve(text, s) == wildcards.resolve(text, s, "negative") for s in range(64))
    assert same < 32


def test_each_image_of_a_batch_gets_its_own_seed_and_text():
    positive = {"text": "a {red|blue|green|black} circle", "clip_skip": 0, "controls": ["kept"]}
    negative = {"text": "blurry"}
    seed, runs = generation._wildcard_runs(positive, negative, 100, 3)
    assert seed == 100
    assert [s for s, _p, _n in runs] == [100, 101, 102]
    for image_seed, image_positive, image_negative in runs:
        assert image_positive["text"] == wildcards.resolve(positive["text"], image_seed)
        assert image_positive["controls"] == ["kept"]
        assert image_negative == {"text": "blurry"}
    assert generation._wildcard_runs({"text": "a circle"}, None, 1, 3) is None


def test_progress_of_the_single_image_runs_reads_as_one_bar():
    calls = []

    def progress(current, total, **info):
        calls.append((current, total, info))

    progress.wants_previews = True
    second = generation._run_progress(progress, 1, 3)
    second(2, 4, preview="p")
    assert calls == [(6, 12, {"preview": "p"})]
    assert second.wants_previews is True


def test_editing_a_wildcard_file_changes_the_text_nodes_signature(wildcard_dir):
    path = wildcard_dir / "animal.txt"
    path.write_text("cat\n", encoding="utf-8")
    before = generation.CLIPTextEncode.IS_CHANGED(text="a __animal__")
    path.write_text("cat\ndog\n", encoding="utf-8")
    assert generation.CLIPTextEncode.IS_CHANGED(text="a __animal__") != before
    assert generation.CLIPTextEncode.IS_CHANGED(text="no wildcards") == []


# ---- with the tiny real pipeline ----


def _graph(text, seed, batch_size):
    return {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": text}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#000000"}},
        "sampler": {
            "class_type": "KSampler",
            "inputs": {"model": ["ckpt", 0], "positive": ["clip", 0], "latent_image": ["latent", 0],
                       "sampler": "euler", "seed": seed, "steps": 2, "cfg": 7.0, "batch_size": batch_size},
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "wc"}},
    }


def _run(prompt):
    try:
        return execute_prompt(validate_prompt(prompt))
    except Exception as exc:
        skip_if_hub_unreachable(exc)
        raise


def _pixels(path):
    with Image.open(path) as img:
        return list(img.convert("RGB").getdata())


def test_a_wildcard_batch_varies_and_each_image_reproduces_alone(output_dir, monkeypatch):
    texts = []
    original = generation._prompt_kwargs

    def recording(model, pipe, positive, negative, repeat):
        texts.append(positive["text"])
        return original(model, pipe, positive, negative, repeat)

    monkeypatch.setattr(generation, "_prompt_kwargs", recording)
    prompt = "a {red|blue|green|black|white|pink} circle"
    outputs = _run(_graph(prompt, seed=100, batch_size=3))
    assert outputs["sampler"]["seeds"] == [100, 101, 102]
    assert texts == [wildcards.resolve(prompt, s) for s in (100, 101, 102)]
    names = outputs["save"]["images"]
    assert len(names) == 3
    with Image.open(output_dir / names[2]) as img:
        embedded = json.loads(img.text["prompt"])
    assert embedded["sampler"]["inputs"]["seed"] == 102
    assert embedded["clip"]["inputs"]["text"] == prompt
    rerun = _run(embedded)
    assert texts[-1] == texts[2]
    a, b = _pixels(output_dir / names[2]), _pixels(output_dir / rerun["save"]["images"][0])
    assert max(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb)) <= 2
