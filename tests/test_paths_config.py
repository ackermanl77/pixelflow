"""Data-folder resolution (R3), shared port/app-id config (T7), lazy torch
import (P6), Save Image naming (B17) and the .gitignore entries (R7)."""

import datetime
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from backend import config, paths
from backend.nodes import io_nodes

ROOT = Path(__file__).resolve().parent.parent


# ---- R3 ----


def test_pixelflow_home_wins(tmp_path):
    assert paths.resolve_data_root({"PIXELFLOW_HOME": str(tmp_path)}, frozen=True) == tmp_path.resolve()


def test_running_from_source_uses_the_project_folder():
    assert paths.resolve_data_root({}, frozen=False) == ROOT


def test_frozen_app_defaults_to_local_app_data(tmp_path, monkeypatch):
    exe = tmp_path / "Program Files" / "PixelFlow" / "PixelFlow.exe"
    exe.parent.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "local"))
    root = paths.resolve_data_root({}, frozen=True, executable=exe)
    assert root.parent == tmp_path / "local"
    assert root.name.lower() == "pixelflow"


@pytest.mark.parametrize("marker", ["portable.txt", "outputs", "history.jsonl"])
def test_portable_marker_or_existing_data_keeps_the_exe_folder(tmp_path, marker):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    if marker == "outputs":
        (app_dir / "outputs").mkdir()
    else:
        (app_dir / marker).write_text("")
    assert paths.resolve_data_root({}, frozen=True, executable=app_dir / "PixelFlow.exe") == app_dir.resolve()


def test_models_dir_from_env_or_settings(tmp_path):
    assert paths.resolve_models_dir(tmp_path, {}) == tmp_path / "models"
    (tmp_path / "settings.json").write_text(json.dumps({"models_dir": str(tmp_path / "big-drive")}))
    assert paths.resolve_models_dir(tmp_path, {}) == (tmp_path / "big-drive").resolve()
    env = {"PIXELFLOW_MODELS_DIR": str(tmp_path / "env-models")}
    assert paths.resolve_models_dir(tmp_path, env) == (tmp_path / "env-models").resolve()
    (tmp_path / "settings.json").write_text("{broken")
    assert paths.resolve_models_dir(tmp_path, {}) == tmp_path / "models"


def test_ensure_data_dirs_never_raises(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(paths, "OUTPUT_DIR", blocker / "outputs")  # can't mkdir under a file
    paths.ensure_data_dirs()


# ---- T7 ----


def test_port_comes_from_one_place(monkeypatch):
    monkeypatch.delenv("PIXELFLOW_PORT", raising=False)
    assert config.port() == 8288
    monkeypatch.setenv("PIXELFLOW_PORT", "9123")
    assert config.port() == 9123
    monkeypatch.setenv("PIXELFLOW_PORT", "not-a-port")
    assert config.port() == 8288
    run_py = (ROOT / "run.py").read_text(encoding="utf-8")
    assert "port=8288" not in run_py and "backend.config" in run_py


def test_index_html_has_no_manual_cache_busting():
    assert "?v=" not in (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


# ---- P6 ----


def test_importing_the_server_does_not_import_torch():
    code = "import sys, backend.main; print('torch' in sys.modules, 'diffusers' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


# ---- B17 ----


def test_save_image_numbers_files_in_creation_order(output_dir):
    names = [io_nodes.SaveImage().save(Image.new("RGB", (4, 4)), "shot")["images"][0] for _ in range(3)]
    assert names == ["shot_00001_.png", "shot_00002_.png", "shot_00003_.png"]
    (output_dir / "shot_00010_.png").write_bytes(b"")
    assert io_nodes.SaveImage().save(Image.new("RGB", (4, 4)), "shot")["images"] == ["shot_00011_.png"]


def test_prefix_tokens_and_subfolders(output_dir):
    image = Image.new("RGB", (4, 4))
    image.info[io_nodes.SEED_INFO_KEY] = 42
    image.info[io_nodes.MODEL_INFO_KEY] = "org/model"
    (name,) = io_nodes.SaveImage().save(image, "runs/%date%/s%seed%_%model%")["images"]
    today = datetime.date.today().isoformat()
    assert name == f"runs/{today}/s42_org_model_00001_.png"
    assert (output_dir / name).is_file()


@pytest.mark.parametrize("prefix", ["../../escape", "a/../../b", "/abs/x", "C:\\Windows\\x", "..\\..\\y"])
def test_prefix_can_never_leave_the_output_folder(output_dir, prefix):
    (name,) = io_nodes.SaveImage().save(Image.new("RGB", (4, 4)), prefix)["images"]
    assert (output_dir / name).resolve().is_relative_to(output_dir.resolve())


# ---- R7 ----


def test_gitignore_covers_models_workflows_and_temp():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in ("/models/", "/workflows/*", "!/workflows/.gitkeep", "/temp/"):
        assert entry in lines, entry
