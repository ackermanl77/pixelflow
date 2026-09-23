import pytest

from backend.engine.graph import GraphError, execute_prompt
from backend.paths import OUTPUT_DIR


def test_execute_prompt_runs_save_image():
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#ff0000"}},
        "2": {"class_type": "SaveImage", "inputs": {"image": ["1", 0], "filename_prefix": "test"}},
    }
    outputs = execute_prompt(prompt)
    assert "2" in outputs
    filename = outputs["2"]["images"][0]
    assert (OUTPUT_DIR / filename).exists()


def test_execute_prompt_stub_pipeline():
    prompt = {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": "stub-v1"}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red circle"}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "color": "#000000"}},
        "sampler": {
            "class_type": "KSamplerStub",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["clip", 0],
                "latent_image": ["latent", 0],
                "seed": 42,
                "steps": 10,
                "cfg": 7.0,
            },
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "pipeline"}},
    }
    outputs = execute_prompt(prompt)
    filename = outputs["save"]["images"][0]
    assert (OUTPUT_DIR / filename).exists()


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
