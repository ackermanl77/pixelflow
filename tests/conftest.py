import pytest

from backend import main, paths
from backend.nodes import io_nodes


def pytest_collection_modifyitems(config, items):
    """`gpu`-marked tests (report item T4) need a CUDA device; skip them
    elsewhere (CI, CPU-only machines)."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has_cuda = False
    if has_cuda:
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_data_dirs(tmp_path, monkeypatch):
    """Point every writable data folder at a per-test tmp dir, so the suite
    never writes into (or deletes from) the real outputs/, uploads/,
    workflows/, temp/ or models/ folders. backend.paths' constants are
    imported by name into the modules that use them, so those module
    attributes are what gets patched."""
    dirs = {name: tmp_path / name for name in ("outputs", "uploads", "workflows", "temp")}
    for path in dirs.values():
        path.mkdir()
    monkeypatch.setattr(io_nodes, "OUTPUT_DIR", dirs["outputs"])
    monkeypatch.setattr(io_nodes, "TEMP_DIR", dirs["temp"])
    monkeypatch.setattr(io_nodes, "UPLOAD_DIR", dirs["uploads"])
    monkeypatch.setattr(main, "UPLOAD_DIR", dirs["uploads"])
    monkeypatch.setattr(main, "WORKFLOW_DIR", dirs["workflows"])
    # Model folders are read through backend.paths at call time.
    models_dir = tmp_path / "models"
    dirs["checkpoints"] = models_dir / "checkpoints"
    dirs["loras"] = models_dir / "loras"
    monkeypatch.setattr(paths, "MODELS_DIR", models_dir)
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", dirs["checkpoints"])
    monkeypatch.setattr(paths, "LORA_DIR", dirs["loras"])
    dirs["embeddings"] = models_dir / "embeddings"
    monkeypatch.setattr(paths, "EMBEDDING_DIR", dirs["embeddings"])
    dirs["controlnet"] = models_dir / "controlnet"
    monkeypatch.setattr(paths, "CONTROLNET_DIR", dirs["controlnet"])
    dirs["upscale_models"] = models_dir / "upscale_models"
    monkeypatch.setattr(paths, "UPSCALE_MODEL_DIR", dirs["upscale_models"])
    monkeypatch.setattr(paths, "LOG_DIR", tmp_path / "logs")
    dirs["wildcards"] = tmp_path / "wildcards"
    monkeypatch.setattr(paths, "WILDCARD_DIR", dirs["wildcards"])
    dirs["custom_nodes"] = tmp_path / "custom_nodes"
    monkeypatch.setattr(paths, "CUSTOM_NODES_DIR", dirs["custom_nodes"])
    for name, key in (("OUTPUT_DIR", "outputs"), ("UPLOAD_DIR", "uploads"), ("WORKFLOW_DIR", "workflows"), ("TEMP_DIR", "temp")):
        monkeypatch.setattr(paths, name, dirs[key])
    # The app-wide queue persists finished prompts; keep that out of the repo too.
    monkeypatch.setattr(main.queue, "history_path", tmp_path / "history.jsonl")
    monkeypatch.setattr(paths, "OUTPUT_CACHE_DIR", tmp_path / "cache" / "outputs")
    return dirs


@pytest.fixture
def output_dir(isolated_data_dirs):
    return isolated_data_dirs["outputs"]


@pytest.fixture
def temp_dir(isolated_data_dirs):
    return isolated_data_dirs["temp"]


@pytest.fixture
def upload_dir(isolated_data_dirs):
    return isolated_data_dirs["uploads"]
