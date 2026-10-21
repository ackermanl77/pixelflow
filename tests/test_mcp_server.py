"""PixelFlow's MCP server on stdio (report item F24)."""

import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from backend import mcp_server, paths

ROOT = Path(__file__).resolve().parent.parent
PROMPT = {
    "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#336699"}},
    "2": {"class_type": "SaveImage", "inputs": {"image": ["1", 0], "filename_prefix": "mcp"}},
}


def _call(message, connect=None):
    return mcp_server.handle(message, connect or (lambda: pytest.fail("no server needed")))


def test_initialize_negotiates_the_protocol_and_lists_tools():
    reply = _call({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t"}}})
    assert reply["result"]["protocolVersion"] == "2025-03-26"
    assert reply["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert reply["result"]["serverInfo"]["name"] == "pixelflow"
    unknown = _call({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}})
    assert unknown["result"]["protocolVersion"] == mcp_server.PROTOCOL_VERSIONS[0]
    tools = _call({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})["result"]["tools"]
    assert [t["name"] for t in tools] == ["list_workflows", "run_workflow", "queue_status", "get_outputs"]
    assert all(t["inputSchema"]["type"] == "object" for t in tools)


def test_pings_notifications_and_errors_follow_json_rpc():
    assert _call({"jsonrpc": "2.0", "id": 4, "method": "ping"})["result"] == {}
    assert _call({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert _call({"jsonrpc": "2.0", "id": 5, "method": "resources/list"})["error"]["code"] == -32601
    bad_tool = _call({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "rm_rf"}})
    assert bad_tool["error"]["code"] == -32602
    out = io.StringIO()
    mcp_server.serve(io.StringIO('not json\n\n{"jsonrpc": "2.0", "id": 7, "method": "ping"}\n'), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert replies[0]["error"]["code"] == -32700 and replies[1] == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_tool_failures_are_tool_results_not_protocol_errors():
    def unreachable():
        raise mcp_server.ToolError("No running PixelFlow found")

    reply = _call({"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "queue_status"}}, unreachable)
    assert reply["result"]["isError"] is True
    assert reply["result"]["content"][0]["text"] == "No running PixelFlow found"


def test_inputs_target_one_node_or_every_node_of_a_class():
    prompt = {
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "a"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "b"}},
        "6": {"class_type": "KSampler", "inputs": {"seed": 1, "positive": ["4", 0]}},
    }
    changed = mcp_server.apply_inputs(prompt, {"CLIPTextEncode.text": "fox", "6.seed": 9})
    assert [changed[n]["inputs"]["text"] for n in ("4", "5")] == ["fox", "fox"]
    assert changed["6"]["inputs"]["seed"] == 9
    assert prompt["4"]["inputs"]["text"] == "a", "the saved prompt is not modified"
    with pytest.raises(mcp_server.ToolError, match=r"its nodes are: 4 \(CLIPTextEncode\), 5"):
        mcp_server.apply_inputs(prompt, {"Nope.text": 1})
    with pytest.raises(mcp_server.ToolError, match="is a link"):
        mcp_server.apply_inputs(prompt, {"6.positive": "x"})
    with pytest.raises(mcp_server.ToolError, match="must look like"):
        mcp_server.apply_inputs(prompt, {"seed": 1})


def test_the_server_is_found_through_the_desktop_apps_note(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "ROOT_DIR", tmp_path)
    monkeypatch.setenv("PIXELFLOW_PORT", "8288")
    mcp_server.record_server(9123)
    assert mcp_server.candidate_urls({"PIXELFLOW_URL": "http://127.0.0.1:7000"}) == [
        "http://127.0.0.1:7000", "http://127.0.0.1:9123", "http://127.0.0.1:8288"]
    mcp_server.forget_server(9123)
    assert not (tmp_path / "server.json").exists()


# ---- end to end: a real server, the real stdio process ----


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    port = _free_port()
    env = dict(os.environ, PIXELFLOW_TEST_MODE="1", PIXELFLOW_HOME=str(tmp_path / "data"),
               PIXELFLOW_DISABLE_CUSTOM_NODES="1")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if requests.get(url + "/api/health", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(0.2)
    else:
        process.kill()
        pytest.fail("test server did not start")
    yield url, tmp_path / "data"
    process.terminate()
    process.wait(10)


def test_an_assistant_runs_a_saved_workflow_over_stdio(server):
    url, data_dir = server
    graph = {"nodes": [], "links": [], "extra": {mcp_server.SAVED_PROMPT_KEY: PROMPT}}
    assert requests.post(f"{url}/api/workflows/Swatch", json=graph, timeout=5).ok
    assert requests.post(f"{url}/api/workflows/Old", json={"nodes": [], "links": []}, timeout=5).ok

    process = subprocess.Popen(
        [sys.executable, "-m", "backend.mcp_server"], cwd=ROOT, env=dict(os.environ, PIXELFLOW_URL=url),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
    )

    def rpc(request_id, method, params=None):
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method,
                                        "params": params or {}}) + "\n")
        process.stdin.flush()
        return json.loads(process.stdout.readline())

    try:
        assert rpc(1, "initialize", {"protocolVersion": "2025-06-18"})["result"]["serverInfo"]["name"] == "pixelflow"
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        listed = rpc(2, "tools/call", {"name": "list_workflows"})["result"]["structuredContent"]["workflows"]
        assert {(w["name"], w["runnable"]) for w in listed} == {("Swatch", True), ("Old", False)}

        old = rpc(3, "tools/call", {"name": "run_workflow", "arguments": {"name": "Old"}})["result"]
        assert old["isError"] and "open it in PixelFlow and save it once" in old["content"][0]["text"]

        ran = rpc(4, "tools/call", {"name": "run_workflow", "arguments": {
            "name": "Swatch", "inputs": {"1.width": 96, "SaveImage.filename_prefix": "from_mcp"}}})["result"]
        assert ran["isError"] is False, ran
        result = ran["structuredContent"]
        assert result["status"] == "completed"
        (image,) = result["images"]
        assert image["filename"].startswith("from_mcp") and image["url"].startswith(url + "/outputs/")
        from PIL import Image

        with Image.open(image["path"]) as img:
            assert img.size == (96, 64)
        assert Path(image["path"]).parent == data_dir / "outputs"

        again = rpc(5, "tools/call", {"name": "get_outputs", "arguments": {"prompt_id": result["prompt_id"]}})
        assert again["result"]["structuredContent"]["images"] == result["images"]
        status = rpc(6, "tools/call", {"name": "queue_status"})["result"]["structuredContent"]
        assert status == {"running": None, "pending": []}
    finally:
        process.stdin.close()
        process.wait(10)
