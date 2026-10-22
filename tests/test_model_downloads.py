"""Model downloads from a Hugging Face or Civitai link, SHA-256 checked (report item F23)."""

import hashlib
import http.server
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import main, model_downloads as md

SHA = "a" * 64
BODY = b"pretend safetensors " * 4096
BODY_SHA = hashlib.sha256(BODY).hexdigest()


def _head(status=302, headers=None, seen=None):
    def head(url, request_headers):
        if seen is not None:
            seen.append((url, dict(request_headers)))
        return SimpleNamespace(status_code=status, headers=headers or {})

    return head


# ---- resolving links ----


def test_a_hugging_face_file_page_resolves_to_its_lfs_file_and_hash(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    seen = []
    plan = md.resolve(
        "https://huggingface.co/acme/style-lora/blob/main/folder/My%20Style.safetensors",
        head=_head(headers={"X-Linked-Etag": f'"{SHA.upper()}"', "X-Linked-Size": "1234"}, seen=seen),
    )
    assert plan.url == "https://huggingface.co/acme/style-lora/resolve/main/folder/My%20Style.safetensors"
    assert (plan.filename, plan.kind, plan.sha256, plan.size) == ("My_Style.safetensors", "loras", SHA, 1234)
    assert seen[0][1] == {"Authorization": "Bearer hf_secret"}


@pytest.mark.parametrize("path, kind", [
    ("org/sd15/resolve/main/v1-5-pruned.safetensors", "checkpoints"),
    ("lllyasviel/sd-controlnet-canny/resolve/main/diffusion_pytorch_model.safetensors", "controlnet"),
    ("ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth", "upscale_models"),
    ("someone/my-embeddings/resolve/main/bad_hands.pt", "embeddings"),
])
def test_hugging_face_folders_are_guessed_from_the_name(path, kind):
    assert md.resolve(f"https://huggingface.co/{path}", head=_head(headers={})).kind == kind


def test_a_chosen_folder_wins_and_small_non_lfs_files_have_no_hash():
    plan = md.resolve("https://huggingface.co/a/b/resolve/main/x.safetensors", "embeddings",
                      head=_head(headers={"ETag": '"0123abcd"'}))
    assert plan.kind == "embeddings" and plan.sha256 is None


@pytest.mark.parametrize("status, message", [(401, "is gated or private"), (404, "is not in a/b at main")])
def test_hugging_face_errors_are_explained(status, message):
    with pytest.raises(md.DownloadError, match=message) as caught:
        md.resolve("https://huggingface.co/a/b/resolve/main/x.safetensors", head=_head(status=status))
    # F26: a gated repository's page, where the licence is accepted.
    assert caught.value.help_url == ("https://huggingface.co/a/b" if status == 401 else None)


CIVITAI_VERSION = {
    "id": 222,
    "model": {"name": "Ink", "type": "LORA"},
    "files": [
        {"name": "ink_training_data.zip", "type": "Training Data", "hashes": {}},
        {"name": "ink-v2.safetensors", "type": "Model", "primary": True, "sizeKB": 2.5,
         "hashes": {"SHA256": SHA.upper()}, "downloadUrl": "https://civitai.com/api/download/models/222"},
    ],
}


def test_a_civitai_page_uses_the_versions_primary_file_type_and_hash(monkeypatch):
    monkeypatch.setenv("CIVITAI_TOKEN", "civ")
    fetched = []

    def fetch(url):
        fetched.append(url)
        return {"modelVersions": [dict(CIVITAI_VERSION)]} if "/models/" in url else dict(CIVITAI_VERSION)

    plan = md.resolve("https://civitai.com/models/111/ink-style", fetch_json=fetch)
    assert fetched == ["https://civitai.com/api/v1/models/111"]
    assert (plan.filename, plan.kind, plan.sha256, plan.size) == ("ink-v2.safetensors", "loras", SHA, 2560)
    assert plan.url == "https://civitai.com/api/download/models/222"
    assert plan.headers == {"Authorization": "Bearer civ"}
    md.resolve("https://civitai.com/models/111?modelVersionId=222", fetch_json=fetch)
    md.resolve("https://civitai.com/api/download/models/222", fetch_json=fetch)
    assert fetched[1:] == ["https://civitai.com/api/v1/model-versions/222"] * 2


def test_civitai_types_map_to_folders():
    for civitai_type, kind in (("Checkpoint", "checkpoints"), ("TextualInversion", "embeddings"),
                               ("Controlnet", "controlnet"), ("Upscaler", "upscale_models"), ("LoCon", "loras")):
        version = dict(CIVITAI_VERSION, model={"type": civitai_type})
        assert md.resolve("https://civitai.com/api/download/models/222", fetch_json=lambda _u: version).kind == kind


@pytest.mark.parametrize("link, message", [
    ("http://huggingface.co/a/b/resolve/main/x.safetensors", "https link"),
    ("https://example.com/model.safetensors", "huggingface.co or a model on civitai.com"),
    ("https://huggingface.co/a/b", "link of one file"),
    ("https://civitai.com/images/5", "model page"),
    ("https://huggingface.co/a/b/resolve/main/readme.md", "not a model file"),
])
def test_other_links_are_refused(link, message):
    with pytest.raises(md.DownloadError, match=message):
        md.resolve(link, head=_head(headers={}))


def test_a_long_name_keeps_its_extension():
    plan = md.resolve(f"https://huggingface.co/a/b/resolve/main/{'x' * 300}.SafeTensors", head=_head(headers={}))
    assert plan.filename.endswith(".safetensors") and len(plan.filename) <= 120


# ---- downloading ----


class _Handler(http.server.BaseHTTPRequestHandler):
    # /drop sends the first half of the body and closes the connection while the
    # `drop` flag is set (F26); /file and /drop honour Range when If-Range matches.
    drop = False
    ranges: list = []

    def do_GET(self):  # noqa: N802 - http.server's naming
        if self.path == "/login":
            payload, content_type = b"<html>please log in</html>", "text/html; charset=utf-8"
        else:
            payload, content_type = BODY, "application/octet-stream"
        requested = self.headers.get("Range")
        _Handler.ranges.append(requested)
        if requested and self.path != "/login" and self.headers.get("If-Range") in (None, '"v1"'):
            start = int(requested.split("=")[1].rstrip("-"))
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("ETag", '"v1"')
            self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
            self.send_header("Content-Length", str(len(payload) - start))
            self.end_headers()
            self.wfile.write(payload[start:])
            return
        if self.path == "/drop" and _Handler.drop:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("ETag", '"v1"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload[: len(payload) // 2])
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        self.send_response(200)
        self.send_header("ETag", '"v1"')
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def file_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _plan(base, sha=BODY_SHA, path="/file", kind="loras", name="style.safetensors"):
    return md.Plan(url=base + path, filename=name, kind=kind, sha256=sha, size=len(BODY))


def test_a_verified_download_lands_in_its_folder_with_progress(file_server, isolated_data_dirs):
    progress = []
    path = md.download(_plan(file_server), lambda received, total: progress.append((received, total)))
    assert path == isolated_data_dirs["loras"] / "style.safetensors"
    assert path.read_bytes() == BODY
    assert progress[-1] == (len(BODY), len(BODY))
    assert md.download(_plan(file_server)) == path, "the same file again is just found"
    with pytest.raises(md.DownloadError, match="already exists"):
        md.download(_plan(file_server, sha="b" * 64))


def test_a_hash_mismatch_leaves_nothing_behind(file_server, isolated_data_dirs):
    with pytest.raises(md.DownloadError, match="SHA-256 doesn't match"):
        md.download(_plan(file_server, sha="b" * 64, kind="checkpoints", name="bad.safetensors"))
    assert list(isolated_data_dirs["checkpoints"].iterdir()) == []


def test_an_interrupted_download_resumes_from_its_part_file(file_server, isolated_data_dirs, monkeypatch):
    """F26: the connection drops half-way -> .part + sidecar stay; the next attempt
    asks for the rest with Range/If-Range and the hash still covers the whole file."""
    monkeypatch.setattr(md, "CHUNK_BYTES", 4096)  # so half the body arrives before the drop
    _Handler.drop, _Handler.ranges = True, []
    plan = _plan(file_server, path="/drop", kind="vae", name="v.safetensors")
    with pytest.raises(md.DownloadError, match="start it again to resume"):
        md.download(plan)
    folder = isolated_data_dirs["vae"]
    part = folder / "v.safetensors.part"
    assert part.is_file() and 0 < part.stat().st_size < len(BODY)
    assert md.read_resume_meta(part, plan)["etag"] == '"v1"'
    _Handler.drop = False
    progress = []
    path = md.download(plan, lambda received, total: progress.append(received))
    assert path.read_bytes() == BODY
    assert _Handler.ranges[-1] == f"bytes={len(BODY) // 2}-"
    assert sorted(p.name for p in folder.iterdir()) == ["v.safetensors"], "the sidecar is gone too"


def test_a_part_file_from_another_download_is_not_resumed(file_server, isolated_data_dirs):
    folder = isolated_data_dirs["loras"]
    folder.mkdir(parents=True)
    (folder / "style.safetensors.part").write_bytes(b"stale bytes")
    (folder / "style.safetensors.part.dl-meta").write_text('{"version": 1, "url": "https://elsewhere"}')
    _Handler.ranges = []
    assert md.download(_plan(file_server)).read_bytes() == BODY
    assert _Handler.ranges == [None]


def test_a_login_page_and_a_cancel_are_refused_cleanly(file_server, isolated_data_dirs):
    with pytest.raises(md.DownloadError, match="web page instead of the file"):
        md.download(_plan(file_server, path="/login", kind="embeddings", name="e.pt"))
    with pytest.raises(md.DownloadCancelled):
        md.download(_plan(file_server, kind="embeddings", name="e.pt"), cancelled=lambda: True)
    assert list(isolated_data_dirs["embeddings"].iterdir()) == []


# ---- jobs and the API ----


def _wait(predicate, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline and not predicate():
        time.sleep(0.02)
    return predicate()


def test_jobs_run_in_the_background_and_report_errors(file_server, isolated_data_dirs):
    def resolver(url, kind):
        if "bad" in url:
            raise md.DownloadError("nope")
        return _plan(file_server, kind="loras" if kind == "auto" else kind)

    manager = md.DownloadManager(resolver=resolver)
    ok = manager.start("https://civitai.com/models/1")
    bad = manager.start("https://civitai.com/models/bad")
    assert _wait(lambda: {j["id"]: j["status"] for j in manager.list()} == {ok["id"]: "done", bad["id"]: "error"})
    done = next(j for j in manager.list() if j["id"] == ok["id"])
    assert (done["filename"], done["folder"], done["verified"], done["bytes"]) == (
        "style.safetensors", "loras", True, len(BODY))
    assert next(j for j in manager.list() if j["id"] == bad["id"])["error"] == "nope"
    assert manager.cancel(ok["id"]) is False, "a finished job can't be cancelled"


def test_the_api_starts_lists_and_cancels_downloads(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_download(plan, on_progress, cancelled):
        started.set()
        while not cancelled():
            if release.wait(0.01):
                break
        if cancelled():
            raise md.DownloadCancelled()

    manager = md.DownloadManager(resolver=lambda url, kind: md.Plan(url=url, filename="m.safetensors",
                                                                     kind="checkpoints"),
                                 downloader=slow_download)
    monkeypatch.setattr(md, "MANAGER", manager)
    client = TestClient(main.app, base_url="http://127.0.0.1:8288")
    assert client.post("/api/downloads", json={"url": "https://x", "kind": "weird"}).status_code == 400
    job = client.post("/api/downloads", json={"url": "https://civitai.com/models/1"}).json()
    assert started.wait(5)
    assert client.get("/api/downloads").json()[0]["status"] == "downloading"
    assert client.delete(f"/api/downloads/{job['id']}").json() == {"cancelled": job["id"]}
    assert _wait(lambda: client.get("/api/downloads").json()[0]["status"] == "cancelled")
    assert client.delete(f"/api/downloads/{job['id']}").status_code == 404
