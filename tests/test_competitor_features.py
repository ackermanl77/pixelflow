"""The 2026-10-21 competitor rows: the ComfyUI model library (B25), "Connect an
assistant" (F27), Image Compare / Remove Background (F28) and crash and
file-lock diagnostics (R8). Resumable downloads (F26) are in
test_model_downloads.py."""

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import diagnostics, main, mcp_server, model_roots
from backend.nodes import background, controlnet, io_nodes, models, upscale

BASE = "http://127.0.0.1:8288"

YAML = """
dmodels:
    base_path: {base}
    is_default: true
    checkpoints: checkpoints
    diffusion_models: |
        diffusion_models
        unet
    clip: |
        clip
        text_encoders
    loras: loras
    controlnet: controlnet
    upscale_models: upscale_models
relative:
    loras: more_loras
"""


@pytest.fixture
def comfy_library(tmp_path, monkeypatch):
    library = tmp_path / "comfy"
    for folder, name in (("checkpoints", "sdxl.safetensors"), ("unet", "flux.safetensors"), ("diffusion_models", "qwen.safetensors"),
                         ("text_encoders", "t5.safetensors"), ("loras", "style.safetensors"), ("controlnet", "pose.safetensors"),
                         ("upscale_models", "4x.pth")):
        (library / folder).mkdir(parents=True, exist_ok=True)
        (library / folder / name).write_bytes(b"x")
    (tmp_path / "more_loras").mkdir()
    (tmp_path / "more_loras" / "ink.safetensors").write_bytes(b"x")
    config = tmp_path / "extra_model_paths.yaml"
    config.write_text(YAML.format(base=library.as_posix()), encoding="utf-8")
    monkeypatch.setenv("PIXELFLOW_COMFYUI_MODEL_PATHS", str(config))
    model_roots.reset_cache()
    yield library
    model_roots.reset_cache()


# ---- B25: ComfyUI's model library ----


def test_comfyui_yaml_is_parsed_like_comfyui(tmp_path):
    parsed = model_roots.parse_extra_model_paths(YAML.format(base="lib"), tmp_path)
    assert parsed["diffusion_models"] == [tmp_path / "lib" / "diffusion_models", tmp_path / "lib" / "unet"]
    assert parsed["loras"] == [tmp_path / "lib" / "loras", tmp_path / "more_loras"]
    assert "is_default" not in parsed and "base_path" not in parsed


def test_the_library_adds_models_to_every_kind(comfy_library, isolated_data_dirs):
    isolated_data_dirs["loras"].mkdir(parents=True)
    (isolated_data_dirs["loras"] / "style.safetensors").write_bytes(b"own")
    assert models.list_models("loras") == ["ink.safetensors", "style.safetensors"]
    assert models.local_model_path("loras", "style.safetensors") == isolated_data_dirs["loras"] / "style.safetensors", "own folder wins"
    assert models.local_model_path("checkpoints", "sdxl.safetensors") == comfy_library / "checkpoints" / "sdxl.safetensors"
    assert models.list_models("diffusion_models") == ["flux.safetensors", "qwen.safetensors"]
    assert models.list_models("text_encoders") == ["t5.safetensors"]
    assert controlnet.list_controlnets() == ["pose.safetensors"]
    assert upscale.list_upscale_models() == ["4x.pth"]
    assert models.local_model_path("checkpoints", "../../etc/passwd") is None


def test_the_library_can_be_turned_off_and_is_found_automatically(tmp_path):
    install = tmp_path / "Comfy-Desktop" / "ComfyUI-Installs" / "ComfyUI (1)" / "ComfyUI"
    install.mkdir(parents=True)
    (install / "extra_model_paths.yaml").write_text("x: {}", encoding="utf-8")
    assert model_roots.config_files(environ={}, settings={}, local_appdata=str(tmp_path)) == [install / "extra_model_paths.yaml"]
    assert model_roots.config_files(environ={"PIXELFLOW_COMFYUI_MODEL_PATHS": "off"}, settings={}, local_appdata=str(tmp_path)) == []
    assert model_roots.config_files(environ={}, settings={"comfyui_model_paths": False}, local_appdata=str(tmp_path)) == []
    assert model_roots.config_files(environ={}, settings={"comfyui_model_paths": "D:/x.yaml"}) == [__import__("pathlib").Path("D:/x.yaml")]


def test_a_broken_yaml_is_skipped(tmp_path, monkeypatch):
    config = tmp_path / "bad.yaml"
    config.write_text("a: [unclosed", encoding="utf-8")
    monkeypatch.setenv("PIXELFLOW_COMFYUI_MODEL_PATHS", str(config))
    model_roots.reset_cache()
    assert model_roots.extra_roots("checkpoints") == []


# ---- F27: connect an assistant ----


def test_the_mcp_config_for_source_and_the_packaged_app():
    packaged = mcp_server.client_config(frozen=True, executable=r"C:\Program Files\PixelFlow\PixelFlow.exe")
    assert packaged["json"] == {"mcpServers": {"pixelflow": {"command": r"C:\Program Files\PixelFlow\PixelFlow.exe", "args": ["--mcp"]}}}
    assert packaged["claude_code"] == r'claude mcp add pixelflow -- "C:\Program Files\PixelFlow\PixelFlow.exe" --mcp'
    source = mcp_server.client_config(frozen=False, executable="python")
    assert source["args"] == ["-m", "backend.mcp_server"] and "PYTHONPATH" in source["env"]
    assert source["claude_code"].startswith("claude mcp add pixelflow -e PYTHONPATH=")


def test_the_api_serves_the_mcp_config():
    client = TestClient(main.app, base_url=BASE)
    body = client.get("/api/mcp").json()
    assert body["name"] == "pixelflow" and "mcpServers" in body["json"]


# ---- F28: Image Compare and Remove Background ----


def test_image_compare_saves_both_sides(temp_dir):
    a, b = Image.new("RGB", (8, 8), "red"), Image.new("RGB", (8, 8), "blue")
    result = io_nodes.ImageCompare().compare([a, a], b)
    assert result["compare"] is True and result["type"] == "temp" and len(result["images"]) == 2
    first = Image.open(temp_dir / result["images"][0]).convert("RGB")
    assert first.getpixel((0, 0)) == (255, 0, 0)


def test_remove_background_cuts_out_the_subject(monkeypatch):
    def stand_in(image):
        mask = Image.new("L", image.size, 0)
        mask.paste(255, (4, 4, 12, 12))
        return mask

    monkeypatch.setattr(background.RemoveBackground, "segment", staticmethod(stand_in))
    image = Image.new("RGB", (16, 16), (200, 50, 50))
    cut, mask = background.RemoveBackground().run(image, "transparent", 0.0)
    assert cut.mode == "RGBA" and cut.getpixel((8, 8))[3] == 255 and cut.getpixel((0, 0))[3] == 0
    assert mask.getpixel((8, 8)) == 255
    green, _ = background.RemoveBackground().run([image], "green", 0.0)
    assert green[0].getpixel((0, 0)) == (0, 177, 64) and green[0].getpixel((8, 8)) == (200, 50, 50)
    with pytest.raises(ValueError):
        background.RemoveBackground().run(image, "purple")


def test_the_new_nodes_are_listed():
    nodes = TestClient(main.app, base_url=BASE).get("/api/nodes").json()
    assert nodes["ImageCompare"]["output_node"] is True
    assert nodes["RemoveBackground"]["return_types"] == ["IMAGE", "MASK"]


# ---- R8: diagnostics ----


def test_missing_vc_runtime_dlls_are_named():
    def loader(name):
        if name == "vcruntime140_1.dll":
            raise OSError("not found")

    assert diagnostics.missing_vc_runtime_dlls(loader) == ["vcruntime140_1.dll"]
    assert "vcruntime140_1.dll" in diagnostics.vc_runtime_report(loader)
    assert diagnostics.vc_runtime_report(lambda name: None) == ""


@pytest.mark.skipif(sys.platform != "win32", reason="the Restart Manager is Windows-only")
def test_the_process_holding_a_file_is_named(tmp_path):
    path = tmp_path / "held.png"
    path.write_bytes(b"x")
    with open(path, "rb"):
        holders = diagnostics.locking_processes(path)
    assert any(f"pid {os.getpid()}" in holder for holder in holders)
    assert diagnostics.locking_processes(tmp_path / "missing.png") == []


def test_a_locked_delete_is_a_409_naming_the_holder(monkeypatch, output_dir):
    (output_dir / "a.png").write_bytes(b"x")

    def refuse(self, *args, **kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(type(output_dir), "unlink", refuse)
    monkeypatch.setattr(diagnostics, "locking_processes", lambda path: ["Photoshop.exe (pid 42)"])
    response = TestClient(main.app, base_url=BASE).delete("/api/outputs/a.png")
    assert response.status_code == 409
    assert "Photoshop.exe (pid 42)" in json.dumps(response.json())
