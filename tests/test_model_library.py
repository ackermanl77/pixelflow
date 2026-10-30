"""The Model Library panel's API (UI13): installed models with size, preview and loader."""

from fastapi.testclient import TestClient

from backend import main, model_library


def _write(path, size=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_the_library_lists_each_kind_with_size_preview_and_loader(isolated_data_dirs):
    dirs = isolated_data_dirs
    _write(dirs["checkpoints"] / "dreamy.safetensors", 1234)
    _write(dirs["checkpoints"] / "dreamy.preview.png", 5)
    _write(dirs["loras"] / "styles" / "ink.safetensors", 40)
    _write(dirs["loras"] / "styles" / "ink.jpg", 3)
    _write(dirs["vae"] / "sdxl_vae.safetensors", 77)
    _write(dirs["controlnet"] / "canny" / "config.json", 2)
    _write(dirs["controlnet"] / "canny" / "diffusion_pytorch_model.safetensors", 8)
    _write(dirs["controlnet"] / "canny" / "preview.webp", 1)

    lib = model_library.library()
    assert set(lib) == {"checkpoints", "loras", "vae", "controlnet"}
    assert (lib["checkpoints"]["node"], lib["checkpoints"]["input"]) == ("LoadCheckpoint", "ckpt_name")
    checkpoints = {item["name"]: item for item in lib["checkpoints"]["items"]}
    assert checkpoints["dreamy.safetensors"] == {"name": "dreamy.safetensors", "size": 1234, "preview": True}
    assert any(item["size"] is None for item in lib["checkpoints"]["items"])  # the default Hub model
    assert lib["loras"]["items"] == [{"name": "styles/ink.safetensors", "size": 40, "preview": True}]
    assert lib["vae"]["items"] == [{"name": "sdxl_vae.safetensors", "size": 77, "preview": False}]
    assert lib["controlnet"]["items"] == [{"name": "canny", "size": 11, "preview": True}]  # a diffusers folder


def test_previews_are_served_and_nothing_outside_the_model_folders(isolated_data_dirs):
    dirs = isolated_data_dirs
    _write(dirs["loras"] / "ink.safetensors")
    _write(dirs["loras"] / "ink.png", 6)
    client = TestClient(main.app, base_url="http://127.0.0.1:8288")
    response = client.get("/api/model-library/preview", params={"kind": "loras", "name": "ink.safetensors"})
    assert response.status_code == 200 and response.content == b"x" * 6
    for params in ({"kind": "loras", "name": "../../secret.safetensors"}, {"kind": "embeddings", "name": "ink.safetensors"},
                   {"kind": "loras", "name": "missing.safetensors"}):
        assert client.get("/api/model-library/preview", params=params).status_code == 404
    assert client.get("/api/model-library").json()["loras"]["items"][0]["name"] == "ink.safetensors"
