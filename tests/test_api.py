"""HTTP-level tests through FastAPI's TestClient: Host/Origin hardening, the
health endpoint, prompt validation on /api/queue, and /api/interrupt."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import main
from backend.engine.queue import ConnectionManager, ExecutionQueue

BASE = "http://127.0.0.1:8288"
SAME_ORIGIN = {"origin": BASE}
# TestClient.websocket_connect() ignores base_url (defaults to ws://testserver), so pass the full URL.
WS_URL = "ws://127.0.0.1:8288/ws"

VALID_PROMPT = {
    "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#000000"}},
    "2": {"class_type": "SaveImage", "inputs": {"image": ["1", 0], "filename_prefix": "api"}},
}


@pytest.fixture
def fresh_queue(monkeypatch):
    """A queue that's never started, so submitted prompts just sit in it."""
    q = ExecutionQueue(ConnectionManager())
    monkeypatch.setattr(main, "queue", q)
    return q


@pytest.fixture
def client():
    # Not used as a context manager, so the app's startup hook (which starts
    # the real global queue's worker) doesn't run.
    return TestClient(main.app, base_url=BASE)


def test_health_identifies_pixelflow(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["app"] == "pixelflow"


def test_foreign_host_header_is_rejected():
    # DNS rebinding: attacker.example resolved to 127.0.0.1 still sends its own Host.
    res = TestClient(main.app, base_url="http://attacker.example:8288").get("/api/health")
    assert res.status_code == 400


def test_localhost_host_is_allowed():
    assert TestClient(main.app, base_url="http://localhost:8288").get("/api/health").status_code == 200


def test_cross_origin_post_is_rejected(client, fresh_queue):
    res = client.post("/api/queue", json={"prompt": VALID_PROMPT}, headers={"origin": "https://evil.example"})
    assert res.status_code == 403
    assert fresh_queue.history == {}


@pytest.mark.parametrize("origin", ["http://127.0.0.1:9999", "null", "http://localhost:8288"])
def test_other_local_origins_are_not_same_origin(client, fresh_queue, origin):
    res = client.post("/api/queue", json={"prompt": VALID_PROMPT}, headers={"origin": origin})
    assert res.status_code == 403


def test_cross_origin_upload_is_rejected(client):
    res = client.post(
        "/api/upload",
        files={"file": ("x.png", b"data", "image/png")},
        headers={"origin": "https://evil.example"},
    )
    assert res.status_code == 403


def test_same_origin_and_no_origin_posts_are_allowed(client, fresh_queue):
    assert client.post("/api/queue", json={"prompt": VALID_PROMPT}, headers=SAME_ORIGIN).status_code == 200
    assert client.post("/api/queue", json={"prompt": VALID_PROMPT}).status_code == 200
    assert len(fresh_queue.history) == 2


def test_cross_origin_get_is_not_blocked_by_the_origin_check(client):
    # Reads are protected by the browser's same-origin policy + Host check;
    # only state-changing methods need the Origin check.
    assert client.get("/api/nodes", headers={"origin": "https://evil.example"}).status_code == 200


def test_websocket_rejects_cross_origin(client):
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(WS_URL, headers={"origin": "https://evil.example"}) as ws:
            ws.receive_text()
    assert exc_info.value.code == 1008  # our Origin check, not the Host check


def test_websocket_rejects_foreign_host(client):
    with pytest.raises(Exception):
        with client.websocket_connect("ws://attacker.example:8288/ws") as ws:
            ws.receive_text()


def test_websocket_accepts_same_origin_and_no_origin(client):
    with client.websocket_connect(WS_URL, headers=SAME_ORIGIN):
        pass
    with client.websocket_connect(WS_URL):
        pass


def test_queue_rejects_an_invalid_graph_with_400_naming_the_node(client, fresh_queue):
    bad = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 999999, "height": 64, "color": "#000"}},
        "2": {"class_type": "PreviewImage", "inputs": {"image": ["1", 0]}},
    }
    res = client.post("/api/queue", json={"prompt": bad})
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert "Node '1'" in detail["message"]
    assert "maximum" in detail["message"]
    # Structured so the frontend can outline the node and mark the input.
    assert detail["node_id"] == "1"
    assert detail["class_type"] == "EmptyImage"
    assert detail["input"] == "width"
    assert fresh_queue.history == {}


def test_queue_rejects_a_prompt_without_output_nodes(client, fresh_queue):
    only_image = {"1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#000"}}}
    res = client.post("/api/queue", json={"prompt": only_image})
    assert res.status_code == 400
    assert "no output nodes" in res.json()["detail"]["message"]
    assert res.json()["detail"]["node_id"] is None


def test_queue_ignores_an_unconnected_half_built_branch(client, fresh_queue):
    prompt = dict(VALID_PROMPT)
    prompt["stray"] = {"class_type": "ImageBlur", "inputs": {"radius": 2.0}}  # 'image' unconnected
    res = client.post("/api/queue", json={"prompt": prompt})
    assert res.status_code == 200
    _, queued, _ = fresh_queue.pending.get_nowait()
    assert set(queued) == {"1", "2"}


def test_get_queue_reports_running_and_pending_in_order(client, fresh_queue):
    first = client.post("/api/queue", json={"prompt": VALID_PROMPT}).json()["prompt_id"]
    second = client.post("/api/queue", json={"prompt": VALID_PROMPT}).json()["prompt_id"]
    fresh_queue.running_prompt_id = "xyz"
    assert client.get("/api/queue").json() == {"running": "xyz", "pending": [first, second]}
    client.delete(f"/api/queue/{first}")
    assert client.get("/api/queue").json()["pending"] == [second]


def test_history_endpoint_supports_limit_before_and_single_entry(client, fresh_queue):
    for i in range(5):
        fresh_queue._record_history(f"p{i}", {"status": "completed", "outputs": {}})
    assert list(client.get("/api/history").json()) == ["p0", "p1", "p2", "p3", "p4"]
    assert list(client.get("/api/history?limit=2").json()) == ["p3", "p4"]
    assert list(client.get("/api/history?limit=2&before=p3").json()) == ["p1", "p2"]
    assert client.get("/api/history?limit=0").status_code == 422
    assert client.get("/api/history/p2").json()["status"] == "completed"
    assert client.get("/api/history/nope").status_code == 404


def test_queued_history_entry_records_prompt_workflow_and_time(client, fresh_queue):
    body = {"prompt": VALID_PROMPT, "extra_data": {"extra_pnginfo": {"workflow": {"nodes": [1]}}}}
    prompt_id = client.post("/api/queue", json=body).json()["prompt_id"]
    entry = fresh_queue.history[prompt_id]
    assert entry["status"] == "queued"
    assert set(entry["prompt"]) == {"1", "2"}
    assert entry["workflow"] == {"nodes": [1]}
    assert isinstance(entry["queued_at"], float)


def test_queue_accepts_workflow_extra_data(client, fresh_queue):
    body = {"prompt": VALID_PROMPT, "extra_data": {"extra_pnginfo": {"workflow": {"nodes": []}}}}
    res = client.post("/api/queue", json=body)
    assert res.status_code == 200
    _, _, extra = fresh_queue.pending.get_nowait()
    assert extra == {"workflow": {"nodes": []}}


def test_interrupt_with_nothing_running(client, fresh_queue):
    res = client.post("/api/interrupt")
    assert res.status_code == 200
    assert res.json() == {"interrupted": None}


def test_interrupt_sets_the_flag_for_the_running_prompt(client, fresh_queue):
    fresh_queue.running_prompt_id = "abc"
    res = client.post("/api/interrupt")
    assert res.json() == {"interrupted": "abc"}
    assert fresh_queue._interrupt_event.is_set()


def test_pending_prompt_can_be_deleted(client, fresh_queue):
    prompt_id = client.post("/api/queue", json={"prompt": VALID_PROMPT}).json()["prompt_id"]
    assert client.delete(f"/api/queue/{prompt_id}").status_code == 200
    assert fresh_queue.history[prompt_id]["status"] == "cancelled"
