"""The output cache on disk (engine.disk_cache): node outputs survive a
restart, as plain files (no pickle), only for the same app version."""

import json

import numpy as np
import pytest
from PIL import Image

from backend.engine import disk_cache
from backend.engine.graph import OutputCache, execute_prompt
from backend.nodes import NODE_REGISTRY
from backend.nodes.base import NodeBase, register_node


@pytest.fixture
def counting_nodes():
    calls = []
    for name in ("_DiskCounter", "_DiskModel"):
        NODE_REGISTRY.pop(name, None)

    @register_node()
    class _DiskCounter(NodeBase):
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("INT", {"default": 0})}}

        def run(self, value):
            calls.append(value)
            image = Image.new("RGB", (8, 8), (value % 256, 10, 20))
            image.info["pixelflow_seed"] = value
            return ([image],)

    @register_node()
    class _DiskModel(NodeBase):
        """An output that can't go to disk (like a loaded MODEL)."""

        RETURN_TYPES = ("MODEL",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        def run(self):
            calls.append("model")
            return ({"pipeline": object()},)

    yield calls
    for name in ("_DiskCounter", "_DiskModel"):
        NODE_REGISTRY.pop(name, None)


def _graph(value=1, radius=1.0):
    return {
        "src": {"class_type": "_DiskCounter", "inputs": {"value": value}},
        "blur": {"class_type": "ImageBlur", "inputs": {"image": ["src", 0], "radius": radius}},
        "out": {"class_type": "PreviewImage", "inputs": {"image": ["blur", 0]}},
    }


def _cache(root, version="1.0"):
    cache = OutputCache()
    cache.attach_disk(root, version)
    return cache


def test_outputs_are_reused_after_a_restart(counting_nodes, temp_dir, tmp_path):
    root = tmp_path / "cache"
    first = _cache(root)
    execute_prompt(_graph(value=5), output_cache=first)
    first.flush()
    assert counting_nodes == [5]

    restarted = _cache(root)  # a new process: nothing in memory
    assert len(restarted) == 2 and "src" in restarted
    cached = []
    execute_prompt(_graph(value=5, radius=2.0), output_cache=restarted, on_node_cached=lambda nid, cls: cached.append(nid))
    assert cached == ["src"] and counting_nodes == [5], "the source node must come from disk, not run again"
    image = restarted.get("src", restarted._entries["src"][0])[0][0][0]
    assert image.getpixel((0, 0)) == (5, 10, 20) and image.info["pixelflow_seed"] == 5


def test_another_version_or_changed_inputs_start_fresh(counting_nodes, temp_dir, tmp_path):
    root = tmp_path / "cache"
    cache = _cache(root)
    execute_prompt(_graph(value=1), output_cache=cache)
    cache.flush()
    assert len(_cache(root, version="2.0")) == 0
    again = _cache(root)
    execute_prompt(_graph(value=2), output_cache=again)
    again.flush()
    assert counting_nodes == [1, 2]
    # Only the last run's entries stay on disk.
    folders = [p for p in root.iterdir() if p.is_dir()]
    assert len(folders) == 2 and {p.name for p in folders} == set(json.loads((root / "manifest.json").read_text())["entries"].values())


def test_outputs_that_are_not_plain_data_stay_in_memory(tmp_path):
    root = tmp_path / "cache"
    cache = _cache(root)
    cache.commit({"m": ("a" * 64, ({"pipeline": object()},), None), "n": ("b" * 64, (np.arange(6).reshape(2, 3),), {"x": [1]})}, True)
    cache.flush()
    assert "m" in cache
    restarted = _cache(root)
    assert "m" not in restarted and "n" in restarted
    outputs, ui = restarted.get("n", "b" * 64)
    assert np.array_equal(outputs[0], np.arange(6).reshape(2, 3)) and ui == {"x": [1]}


def test_clear_empties_the_folder(counting_nodes, temp_dir, tmp_path):
    root = tmp_path / "cache"
    cache = _cache(root)
    execute_prompt(_graph(), output_cache=cache)
    cache.clear()
    cache.flush()
    assert len(cache) == 0 and not list(root.iterdir())
    assert len(_cache(root)) == 0


def test_a_tampered_cache_is_ignored_not_trusted(tmp_path):
    root = tmp_path / "cache"
    sig = "c" * 64
    (root / sig).mkdir(parents=True)
    (root / sig / "entry.json").write_text(json.dumps({"outputs": {"tuple": [{"png": "../../evil.png", "info": {}}]}}))
    (root / "manifest.json").write_text(json.dumps({"format": disk_cache.FORMAT, "version": "1.0", "entries": {"n": sig, "x": "../.."}}))
    cache = _cache(root)
    assert "x" not in cache and "n" in cache
    assert cache.get("n", sig) is None and "n" not in cache
    (root / "manifest.json").write_text("not json")
    assert len(_cache(root)) == 0


def test_free_memory_clears_the_disk_copy_too(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend import main

    root = tmp_path / "free-cache"
    cache = _cache(root)
    cache.commit({"n": ("d" * 64, (1,), None)}, True)
    cache.flush()
    monkeypatch.setattr(main.queue, "output_cache", cache)
    assert TestClient(main.app, base_url="http://127.0.0.1:8288").post("/api/free").status_code == 200
    cache.flush()
    assert len(_cache(root)) == 0
