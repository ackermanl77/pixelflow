"""/api/workflows endpoints (report items T1 and R1)."""

from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from backend import main

BASE = "http://127.0.0.1:8288"
GRAPH = {"last_node_id": 2, "nodes": [{"id": 1, "type": "PixelFlow/image/Blur"}], "links": [], "note": "héllo ✓"}


@pytest.fixture
def client():
    return TestClient(main.app, base_url=BASE)


def test_save_list_load_round_trip(client, isolated_data_dirs):
    assert client.post("/api/workflows/My flow (v2)", json=GRAPH).json() == {"saved": "My flow (v2)"}
    listed = client.get("/api/workflows").json()
    assert [w["name"] for w in listed] == ["My flow (v2)"]
    assert listed[0]["size"] > 0 and listed[0]["modified"] > 0
    assert client.get("/api/workflows/My flow (v2)").json() == GRAPH
    # Stored as UTF-8 regardless of the OS code page.
    raw = (isolated_data_dirs["workflows"] / "My flow (v2).json").read_bytes()
    assert "héllo ✓".encode("utf-8") in raw


def test_saving_again_overwrites(client):
    client.post("/api/workflows/flow", json={"v": 1})
    client.post("/api/workflows/flow", json={"v": 2})
    assert client.get("/api/workflows/flow").json() == {"v": 2}
    assert [w["name"] for w in client.get("/api/workflows").json()] == ["flow"]


def test_missing_workflow_is_404(client):
    assert client.get("/api/workflows/nope").status_code == 404


@pytest.mark.parametrize(
    "name",
    ["x?y", "x*y", "<b>", "sub\\x", "a" * 101, "trail.", "trail ", "CON", "nul", "com1", "LPT9.backup", ".hidden", "-dash", "a:b", "tab\tname"],
)
def test_invalid_names_are_rejected_with_400(client, isolated_data_dirs, name):
    url = f"/api/workflows/{quote(name, safe='')}"
    res = client.post(url, json=GRAPH)
    assert res.status_code == 400, res.text
    assert "Invalid workflow name" in res.json()["detail"]
    assert client.get(url).status_code == 400
    assert list(isolated_data_dirs["workflows"].iterdir()) == []


def test_traversal_is_rejected(client):
    # An encoded '/' never reaches the handler as one path segment; '..' alone is refused.
    assert client.get("/api/workflows/..%2F..%2Fsecret").status_code in (400, 404)
    assert client.post("/api/workflows/..", json=GRAPH).status_code in (400, 404, 405)


def test_corrupt_workflow_file_is_422_not_500(client, isolated_data_dirs):
    (isolated_data_dirs["workflows"] / "broken.json").write_text("{not json", encoding="utf-8")
    res = client.get("/api/workflows/broken")
    assert res.status_code == 422


def test_list_hides_files_with_names_the_api_cannot_load(client, isolated_data_dirs):
    (isolated_data_dirs["workflows"] / "good.json").write_text("{}", encoding="utf-8")
    (isolated_data_dirs["workflows"] / "bad name?.json".replace("?", "_") ).write_text("{}", encoding="utf-8")
    (isolated_data_dirs["workflows"] / "trail .json").write_text("{}", encoding="utf-8")
    assert [w["name"] for w in client.get("/api/workflows").json()] == ["bad name_", "good"]


# ---- delete / rename / overwrite protection (B12) ----


def test_overwrite_false_refuses_an_existing_workflow(client):
    assert client.post("/api/workflows/flow?overwrite=false", json={"v": 1}).status_code == 200
    res = client.post("/api/workflows/flow?overwrite=false", json={"v": 2})
    assert res.status_code == 409
    assert client.get("/api/workflows/flow").json() == {"v": 1}
    assert client.post("/api/workflows/flow?overwrite=true", json={"v": 3}).status_code == 200
    assert client.get("/api/workflows/flow").json() == {"v": 3}


def test_delete_workflow(client, isolated_data_dirs):
    client.post("/api/workflows/gone", json={})
    assert client.delete("/api/workflows/gone").json() == {"deleted": "gone"}
    assert client.get("/api/workflows").json() == []
    assert client.delete("/api/workflows/gone").status_code == 404
    assert client.delete("/api/workflows/CON").status_code == 400


def test_rename_workflow(client):
    client.post("/api/workflows/old", json={"v": 1})
    client.post("/api/workflows/taken", json={"v": 2})
    assert client.post("/api/workflows/old/rename", json={"new_name": "taken"}).status_code == 409
    assert client.post("/api/workflows/old/rename", json={"new_name": "bad?name"}).status_code == 400
    assert client.post("/api/workflows/missing/rename", json={"new_name": "x"}).status_code == 404
    res = client.post("/api/workflows/old/rename", json={"new_name": "new name"})
    assert res.json() == {"renamed": "old", "to": "new name"}
    assert [w["name"] for w in client.get("/api/workflows").json()] == ["new name", "taken"]
    assert client.get("/api/workflows/new name").json() == {"v": 1}
    # A case-only rename of the same workflow is not a conflict.
    assert client.post("/api/workflows/new name/rename", json={"new_name": "New Name"}).status_code == 200


def test_delete_and_rename_are_origin_checked(client):
    client.post("/api/workflows/keep", json={})
    evil = {"origin": "https://evil.example"}
    assert client.delete("/api/workflows/keep", headers=evil).status_code == 403
    assert client.post("/api/workflows/keep/rename", json={"new_name": "x"}, headers=evil).status_code == 403
