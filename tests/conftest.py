import pytest

from backend import main
from backend.nodes import io_nodes


@pytest.fixture(autouse=True)
def isolated_data_dirs(tmp_path, monkeypatch):
    """Point every writable data folder at a per-test tmp dir, so the suite
    never writes into (or deletes from) the real outputs/, uploads/ or
    workflows/ folders. backend.paths' constants are imported by name into
    the modules that use them, so those module attributes are what gets
    patched."""
    dirs = {name: tmp_path / name for name in ("outputs", "uploads", "workflows")}
    for path in dirs.values():
        path.mkdir()
    monkeypatch.setattr(io_nodes, "OUTPUT_DIR", dirs["outputs"])
    monkeypatch.setattr(io_nodes, "UPLOAD_DIR", dirs["uploads"])
    monkeypatch.setattr(main, "UPLOAD_DIR", dirs["uploads"])
    monkeypatch.setattr(main, "WORKFLOW_DIR", dirs["workflows"])
    return dirs


@pytest.fixture
def output_dir(isolated_data_dirs):
    return isolated_data_dirs["outputs"]


@pytest.fixture
def upload_dir(isolated_data_dirs):
    return isolated_data_dirs["uploads"]
