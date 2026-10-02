"""Queue/interrupt/history management (B11, F8), node-level validation and
the /api/nodes contract (B14, T2), previews in temp/ (B15), gallery file
actions (F9), request size limits and embedded metadata (R6)."""

import asyncio
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import main
from backend.engine.graph import PromptValidationError, execute_prompt, validate_prompt
from backend.engine.queue import ConnectionManager, ExecutionQueue
from backend.nodes import NODE_REGISTRY, io_nodes

BASE = "http://127.0.0.1:8288"
PROMPT = {
    "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#000000"}},
    "2": {"class_type": "SaveImage", "inputs": {"image": ["1", 0], "filename_prefix": "api"}},
}


@pytest.fixture
def fresh_queue(monkeypatch):
    q = ExecutionQueue(ConnectionManager())
    monkeypatch.setattr(main, "queue", q)
    return q


@pytest.fixture
def client():
    return TestClient(main.app, base_url=BASE)


# ---- B11: interrupt by prompt id ----


def test_interrupt_only_stops_the_named_prompt(client, fresh_queue):
    fresh_queue.running_prompt_id = "b"
    res = client.post("/api/interrupt", json={"prompt_id": "a"})
    assert res.status_code == 409
    assert not fresh_queue._interrupt_event.is_set(), "a stale Cancel must not stop prompt b"
    assert client.post("/api/interrupt", json={"prompt_id": "b"}).json() == {"interrupted": "b"}
    assert fresh_queue._interrupt_event.is_set()


def test_interrupt_without_a_body_still_stops_whatever_runs(client, fresh_queue):
    fresh_queue.running_prompt_id = "x"
    assert client.post("/api/interrupt").json() == {"interrupted": "x"}


# ---- F8: queue front, cancel all, clear finished ----


def test_front_flag_runs_a_prompt_before_the_pending_ones(client, fresh_queue):
    first = client.post("/api/queue", json={"prompt": PROMPT}).json()["prompt_id"]
    second = client.post("/api/queue", json={"prompt": PROMPT}).json()["prompt_id"]
    urgent = client.post("/api/queue", json={"prompt": PROMPT, "front": True}).json()["prompt_id"]
    assert client.get("/api/queue").json()["pending"] == [urgent, first, second]
    assert fresh_queue.pending.get_nowait()[0] == urgent


def test_cancel_all_pending_and_clear_finished(client, fresh_queue):
    ids = [client.post("/api/queue", json={"prompt": PROMPT}).json()["prompt_id"] for _ in range(3)]
    fresh_queue._record_history("done", {"status": "completed", "outputs": {}})
    assert client.delete("/api/queue").json() == {"cancelled": ids}
    assert client.get("/api/queue").json()["pending"] == []
    deleted = client.delete("/api/history").json()["deleted"]
    assert set(deleted) == set(ids) | {"done"}
    assert client.get("/api/history").json() == {}


def test_history_lifecycle_through_the_real_worker(output_dir):
    """queued -> running -> completed with timestamps, outputs and the
    executing/progress messages (report item T2)."""
    messages = []

    class _Manager(ConnectionManager):
        async def broadcast(self, message):
            messages.append(message)

    queue = ExecutionQueue(_Manager())

    async def _run():
        queue.start()
        prompt_id = await queue.submit(validate_prompt(PROMPT))
        for _ in range(300):
            if queue.history[prompt_id]["status"] == "completed":
                break
            await asyncio.sleep(0.01)
        return prompt_id

    prompt_id = asyncio.run(_run())
    entry = queue.history[prompt_id]
    assert entry["status"] == "completed"
    assert entry["queued_at"] <= entry["started_at"] <= entry["finished_at"]
    assert entry["outputs"]["2"]["type"] == "output"
    assert (output_dir / entry["outputs"]["2"]["images"][0]).is_file()
    kinds = [m["type"] for m in messages]
    assert kinds[0] == "queued" and kinds[-1] == "execution_success"
    executing = [m for m in messages if m["type"] == "executing"]
    assert [m["node_index"] for m in executing] == [1, 2] and executing[0]["node_count"] == 2


# ---- B14: VALIDATE_INPUTS, descriptions, tooltips ----


def test_bad_colour_is_rejected_before_queueing(client, fresh_queue):
    bad = json.loads(json.dumps(PROMPT))
    bad["1"]["inputs"]["color"] = "not-a-colour"
    res = client.post("/api/queue", json={"prompt": bad})
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert detail["node_id"] == "1" and detail["input"] == "color"
    assert "not a colour" in detail["message"]
    assert fresh_queue.history == {}


def test_text_overlay_colour_is_validated_too():
    graph = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#000"}},
        "2": {
            "class_type": "ImageTextOverlay",
            "inputs": {"image": ["1", 0], "text": "hi", "x": 0, "y": 0, "size": 12, "color": "nope"},
        },
        "3": {"class_type": "PreviewImage", "inputs": {"image": ["2", 0]}},
    }
    with pytest.raises(PromptValidationError, match="not a colour") as info:
        validate_prompt(graph)
    assert info.value.node_id == "2" and info.value.class_type == "ImageTextOverlay"


def test_a_crashing_validate_inputs_hook_is_a_400_not_a_500(monkeypatch):
    def boom(cls, **_):
        raise RuntimeError("hook bug")

    monkeypatch.setattr(NODE_REGISTRY["EmptyImage"], "VALIDATE_INPUTS", classmethod(boom))
    with pytest.raises(PromptValidationError, match="hook bug"):
        validate_prompt(PROMPT)


def test_node_info_contract(client):
    """The shape the frontend builds its nodes from. A change here must be
    deliberate (report item T2)."""
    info = client.get("/api/nodes").json()
    assert {"LoadCheckpoint", "KSampler", "KSamplerImage", "KSamplerInpaint", "LoraLoader", "SaveImage"} <= set(info)
    assert "KSamplerStub" not in info
    for name, node in info.items():
        assert set(node) == {
            "display_name",
            "category",
            "description",
            "input_types",
            "return_types",
            "return_names",
            "output_node",
        }, name
        assert set(node["input_types"]) <= {"required", "optional"}, "hidden inputs are not sent"
        assert len(node["return_types"]) == len(node["return_names"])
    sampler = info["KSampler"]
    assert sampler["description"]
    seed_config = sampler["input_types"]["required"]["seed"][1]
    assert seed_config["control_after_generate"] is True and "tooltip" in seed_config
    assert sampler["input_types"]["optional"]["scheduler"][0] == ["normal", "karras", "trailing"]
    assert client.get("/api/node_aliases").json() == {"KSamplerStub": "KSampler"}


# ---- B15: previews in temp/ ----


def test_previews_go_to_temp_and_temp_is_cleared_at_startup(temp_dir, output_dir, client):
    graph = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "color": "#000000"}},
        "2": {"class_type": "PreviewImage", "inputs": {"image": ["1", 0]}},
    }
    outputs = execute_prompt(graph)
    (name,) = outputs["2"]["images"]
    assert outputs["2"]["type"] == "temp"
    assert (temp_dir / name).is_file() and list(output_dir.iterdir()) == []
    served = client.get(f"/temp/{name}")
    assert served.status_code == 200 and served.headers["cache-control"] == "no-store"
    assert main.clear_temp_dir(temp_dir) == 1
    assert list(temp_dir.iterdir()) == []


# ---- F9: gallery file actions ----


def test_delete_output_and_send_to_uploads(client, output_dir, upload_dir):
    outputs = execute_prompt(PROMPT)
    (name,) = outputs["2"]["images"]
    sent = client.post("/api/outputs/to-upload", json={"filename": name}).json()["filename"]
    assert (upload_dir / sent).is_file() and sent in client.get("/api/uploads").json()
    assert client.get(f"/outputs/{name}").status_code == 200
    assert client.delete(f"/api/outputs/{name}").json() == {"deleted": name}
    assert not (output_dir / name).exists()
    assert client.delete(f"/api/outputs/{name}").status_code == 404
    assert client.delete("/api/outputs/..%2Fhistory.jsonl").status_code == 404
    assert client.delete(f"/api/outputs/{name}?type=bogus").status_code == 400


def test_outputs_in_subfolders_are_served(client, output_dir):
    graph = json.loads(json.dumps(PROMPT))
    graph["2"]["inputs"]["filename_prefix"] = "sub/dir/pic"
    (name,) = execute_prompt(graph)["2"]["images"]
    assert name == "sub/dir/pic_00001_.png"
    assert client.get(f"/outputs/{name}").status_code == 200


# ---- R6: request size cap and embedded metadata ----


def test_oversized_request_bodies_are_413(client, fresh_queue, monkeypatch):
    big = {"prompt": PROMPT, "extra_data": {"extra_pnginfo": {"workflow": {"x": "a" * (main.MAX_REQUEST_BYTES + 10)}}}}
    res = client.post("/api/queue", json=big)
    assert res.status_code == 413
    assert fresh_queue.history == {}


def test_streamed_oversized_body_without_length_is_413(client, fresh_queue):
    def chunks():
        for _ in range(6):
            yield b"x" * (1024 * 1024)

    res = client.post("/api/queue", content=chunks(), headers={"content-type": "application/json"})
    assert res.status_code == 413


def test_upload_limit_is_larger_than_the_general_one(client, upload_dir):
    buf = io.BytesIO()
    Image.effect_noise((1400, 1400), 100).convert("RGB").save(buf, "BMP")  # ~5.9 MB, above 5 MB
    data = buf.getvalue()
    assert len(data) > main.MAX_REQUEST_BYTES
    res = client.post("/api/upload", files={"file": ("big.bmp", data, "image/bmp")})
    assert res.status_code == 200, res.text


def test_only_the_workflow_is_embedded_from_extra_pnginfo(output_dir):
    outputs = execute_prompt(PROMPT, extra_pnginfo={"workflow": {"nodes": []}, "junk": "x" * 1000})
    with Image.open(output_dir / outputs["2"]["images"][0]) as img:
        assert json.loads(img.text["workflow"]) == {"nodes": []}
        assert "junk" not in img.text


def test_health_reports_the_version(client):
    assert client.get("/api/health").json()["version"]


def test_frontend_files_are_revalidated(client):
    res = client.get("/app.js")
    assert res.status_code == 200 and res.headers["cache-control"] == "no-cache"


def test_open_log_folder(client, monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(main, "_open_folder", opened.append)
    res = client.post("/api/logs/open")
    assert res.status_code == 200 and opened and opened[0].name == "logs"
    assert client.post("/api/logs/open", headers={"origin": "https://evil.example"}).status_code == 403
