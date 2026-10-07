"""P3: opt-in torch.compile of the UNet (backend.nodes.models.maybe_compile)."""
import json
import types

import pytest

from backend import paths
from backend.nodes import models


@pytest.fixture
def settings_root(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXELFLOW_TORCH_COMPILE", raising=False)
    monkeypatch.setattr(paths, "ROOT_DIR", tmp_path)
    return tmp_path


@pytest.mark.parametrize("value, mode", [(None, None), (False, None), (True, "default"), ("max-autotune", "max-autotune-no-cudagraphs"), ("nonsense", None)])
def test_the_setting_picks_the_mode(settings_root, value, mode):
    if value is not None:
        (settings_root / paths.SETTINGS_FILENAME).write_text(json.dumps({"torch_compile": value}))
    assert models.torch_compile_mode() == mode


def test_the_environment_overrides_the_setting(settings_root, monkeypatch):
    (settings_root / paths.SETTINGS_FILENAME).write_text(json.dumps({"torch_compile": True}))
    monkeypatch.setenv("PIXELFLOW_TORCH_COMPILE", "0")
    assert models.torch_compile_mode() is None


def test_off_or_without_triton_nothing_is_compiled(settings_root, monkeypatch):
    unet = object()
    pipe = types.SimpleNamespace(unet=unet)
    assert models.maybe_compile(pipe) is False and pipe.unet is unet
    monkeypatch.setenv("PIXELFLOW_TORCH_COMPILE", "1")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None if name == "triton" else object())
    assert models.maybe_compile(pipe) is False and pipe.unet is unet


def test_a_compiled_unet_is_not_compiled_again(settings_root, monkeypatch):
    monkeypatch.setenv("PIXELFLOW_TORCH_COMPILE", "1")
    unet = types.SimpleNamespace(_orig_mod=object())
    pipe = types.SimpleNamespace(unet=unet)
    assert models.maybe_compile(pipe) is True and pipe.unet is unet


def test_with_triton_the_unet_is_wrapped(settings_root, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    monkeypatch.setenv("PIXELFLOW_TORCH_COMPILE", "1")
    pipe = types.SimpleNamespace(unet=torch.nn.Linear(2, 2))
    assert models.maybe_compile(pipe) is True and hasattr(pipe.unet, "_orig_mod")
