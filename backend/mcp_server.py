"""PixelFlow as an MCP server (report item F24), so an assistant such as
Claude can run your saved workflows locally.

    python -m backend.mcp_server          (from source)
    PixelFlow.exe --mcp                   (the packaged app)

It speaks the Model Context Protocol over stdio (newline-delimited JSON-RPC
2.0) -- there is no network listener -- and drives the PixelFlow that is
already running on this machine through its loopback HTTP API, found at
PIXELFLOW_URL, else the port the desktop app recorded in
<data folder>/server.json, else the configured port (8288). Tools:

- list_workflows: the saved workflows, and whether each can be run.
- run_workflow(name, inputs, wait): queue a saved workflow, optionally
  overriding widget values, e.g. {"6.text": "a lighthouse at dusk",
  "KSampler.seed": 42} -- "<node id>.<input>" for one node, or
  "<ClassType>.<input>" for every node of that class -- then (by default)
  wait for it and return the images.
- queue_status: the running prompt and the pending ones.
- get_outputs(prompt_id): a prompt's status and its images (file paths and
  URLs).

A workflow runs exactly what PixelFlow's Queue button would send: Save
Workflow stores that prompt alongside the graph (extra.pixelflow_prompt,
frontend/logic.js), so a workflow saved by an older version needs to be
opened and saved once. Standard library only: no MCP SDK needed."""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from . import paths
from .config import APP_ID, HOST, app_version, port

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MCP_SERVER_NAME = "pixelflow"
SAVED_PROMPT_KEY = "pixelflow_prompt"
SERVER_INFO_FILE = "server.json"
DEFAULT_TIMEOUT_S = 600
POLL_INTERVAL_S = 0.5
TERMINAL_STATUSES = ("completed", "error", "interrupted", "cancelled")

def client_config(frozen: Optional[bool] = None, executable: Optional[str] = None) -> Dict[str, Any]:
    """How an MCP client starts this server (F27, the System panel's "Connect an
    assistant"): the command and arguments, the JSON block for Claude Desktop's
    claude_desktop_config.json (and other clients using the same "mcpServers"
    shape), and the one-line `claude mcp add` for Claude Code."""
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    executable = executable or sys.executable
    if frozen:
        command, args, env = executable, ["--mcp"], {}
    else:
        command, args, env = executable, ["-m", "backend.mcp_server"], {"PYTHONPATH": str(paths.BUNDLE_DIR)}
    entry: Dict[str, Any] = {"command": command, "args": args}
    if env:
        entry["env"] = env
    quoted = " ".join(f'"{part}"' if " " in part else part for part in [command, *args])
    env_flags = "".join(f" -e {key}={value}" for key, value in env.items())
    return {
        "name": MCP_SERVER_NAME,
        "command": command,
        "args": args,
        "env": env,
        "json": {"mcpServers": {MCP_SERVER_NAME: entry}},
        "claude_code": f"claude mcp add {MCP_SERVER_NAME}{env_flags} -- {quoted}",
    }


TOOLS = [
    {
        "name": "list_workflows",
        "description": "List PixelFlow's saved workflows. 'runnable' is false for a workflow saved by an older "
        "PixelFlow: open it in PixelFlow and save it once.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "run_workflow",
        "description": "Queue a saved PixelFlow workflow, optionally changing widget values, and (by default) wait "
        "for it to finish and return the generated images. Input keys are '<node id>.<input>' (one node) or "
        "'<ClassType>.<input>' (every node of that class), e.g. {\"CLIPTextEncode.text\": \"a red fox\", "
        "\"KSampler.seed\": 7}. Unknown keys are an error that lists the workflow's nodes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The saved workflow's name (see list_workflows)."},
                "inputs": {"type": "object", "description": "Widget values to override.", "additionalProperties": True},
                "wait": {"type": "boolean", "description": "Wait for the result (default true).", "default": True},
                "timeout_s": {"type": "number", "description": "How long to wait (default 600).", "default": 600},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "queue_status",
        "description": "The prompt PixelFlow is running now and the ones waiting, in run order.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_outputs",
        "description": "A queued prompt's status and, once completed, its images (file paths and URLs).",
        "inputSchema": {
            "type": "object",
            "properties": {"prompt_id": {"type": "string"}},
            "required": ["prompt_id"],
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    """A tool failure reported to the assistant as an isError result."""


# ------------------------------------------------------------------ HTTP


class PixelFlowClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310 - loopback only
                return json.loads(response.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail")
            except (ValueError, AttributeError):
                detail = None
            if isinstance(detail, dict):
                detail = detail.get("message") or json.dumps(detail)
            raise ToolError(f"PixelFlow answered {exc.code}: {detail or exc.reason}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise ToolError(f"PixelFlow is not reachable at {self.base_url} ({exc}). Is the app running?") from None


def _is_pixelflow(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/health", timeout=2) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8")).get("app") == APP_ID
    except (OSError, ValueError, AttributeError):
        return False


def candidate_urls(environ=None) -> List[str]:
    environ = os.environ if environ is None else environ
    urls = []
    if environ.get("PIXELFLOW_URL"):
        urls.append(environ["PIXELFLOW_URL"])
    try:
        info = json.loads((paths.ROOT_DIR / SERVER_INFO_FILE).read_text(encoding="utf-8"))
        urls.append(f"http://{HOST}:{int(info['port'])}")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    urls.append(f"http://{HOST}:{port()}")
    return list(dict.fromkeys(urls))


def record_server(server_port: int) -> None:
    """The desktop app's note of the port its server listens on (it falls
    back to a free port when another program has 8288), for find_server()."""
    path = paths.ROOT_DIR / SERVER_INFO_FILE
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps({"port": int(server_port), "pid": os.getpid()}), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def forget_server(server_port: int) -> None:
    path = paths.ROOT_DIR / SERVER_INFO_FILE
    try:
        if json.loads(path.read_text(encoding="utf-8")).get("pid") == os.getpid():
            path.unlink()
    except (OSError, ValueError, AttributeError):
        pass


def find_server(environ=None) -> PixelFlowClient:
    urls = candidate_urls(environ)
    for url in urls:
        if _is_pixelflow(url):
            return PixelFlowClient(url)
    raise ToolError(f"No running PixelFlow found (tried {', '.join(urls)}). Start PixelFlow first.")


# ----------------------------------------------------------------- tools


def apply_inputs(prompt: Dict[str, Any], inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A copy of `prompt` with the overrides applied (see run_workflow)."""
    prompt = json.loads(json.dumps(prompt))
    for key, value in (inputs or {}).items():
        target, dot, input_name = str(key).rpartition(".")
        if not dot or not target or not input_name:
            raise ToolError(f"input key '{key}' must look like '<node id>.<input>' or '<ClassType>.<input>'")
        nodes = [prompt[target]] if target in prompt else [n for n in prompt.values() if n.get("class_type") == target]
        if not nodes:
            listing = ", ".join(f"{node_id} ({node.get('class_type')})" for node_id, node in prompt.items())
            raise ToolError(f"no node '{target}' in this workflow; its nodes are: {listing}")
        for node in nodes:
            if isinstance(node["inputs"].get(input_name), list):
                raise ToolError(f"'{key}' is a link from another node, not a value that can be set")
            node["inputs"][input_name] = value
    return prompt


def _images(client: PixelFlowClient, entry: Dict[str, Any], data_dir: Optional[str]) -> List[Dict[str, str]]:
    images = []
    for node_id, output in (entry.get("outputs") or {}).items():
        if not isinstance(output, dict):
            continue
        kind = output.get("type", "output")
        folder = "temp" if kind == "temp" else "outputs"
        for name in output.get("images") or []:
            item = {"node": node_id, "filename": name, "type": kind,
                    "url": f"{client.base_url}/{folder}/{urllib.parse.quote(name)}"}
            if data_dir:
                item["path"] = os.path.join(data_dir, folder, *name.split("/"))
            images.append(item)
    return images


def _describe(client: PixelFlowClient, prompt_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    result = {"prompt_id": prompt_id, "status": entry.get("status")}
    if entry.get("status") == "error":
        result["error"] = {k: entry.get(k) for k in ("error", "node_id", "class_type", "exception_type") if entry.get(k)}
    if entry.get("status") == "completed":
        try:
            data_dir = client.request("GET", "/api/system").get("data_dir")
        except ToolError:
            data_dir = None
        result["images"] = _images(client, entry, data_dir)
    return result


def list_workflows(client: PixelFlowClient, _args) -> Dict[str, Any]:
    workflows = []
    for item in client.request("GET", "/api/workflows"):
        name = item["name"]
        data = client.request("GET", f"/api/workflows/{urllib.parse.quote(name, safe='')}")
        runnable = isinstance((data.get("extra") or {}).get(SAVED_PROMPT_KEY), dict)
        workflows.append({"name": name, "runnable": runnable, "modified": item.get("modified")})
    return {"workflows": workflows}


def run_workflow(client: PixelFlowClient, args) -> Dict[str, Any]:
    name = args.get("name")
    if not isinstance(name, str) or not name:
        raise ToolError("'name' is required")
    data = client.request("GET", f"/api/workflows/{urllib.parse.quote(name, safe='')}")
    saved = (data.get("extra") or {}).get(SAVED_PROMPT_KEY)
    if not isinstance(saved, dict) or not saved:
        raise ToolError(f"workflow '{name}' was saved by an older PixelFlow: open it in PixelFlow and save it once")
    prompt = apply_inputs(saved, args.get("inputs"))
    queued = client.request("POST", "/api/queue", {"prompt": prompt, "client_id": "mcp",
                                                   "extra_data": {"extra_pnginfo": {"workflow": data}}})
    prompt_id = queued["prompt_id"]
    if args.get("wait", True) is False:
        return {"prompt_id": prompt_id, "status": "queued"}
    deadline = time.monotonic() + float(args.get("timeout_s") or DEFAULT_TIMEOUT_S)
    while True:
        entry = client.request("GET", f"/api/history/{prompt_id}")
        if entry.get("status") in TERMINAL_STATUSES:
            return _describe(client, prompt_id, entry)
        if time.monotonic() >= deadline:
            return {"prompt_id": prompt_id, "status": entry.get("status"),
                    "note": "still running; call get_outputs later"}
        time.sleep(POLL_INTERVAL_S)


def queue_status(client: PixelFlowClient, _args) -> Dict[str, Any]:
    return client.request("GET", "/api/queue")


def get_outputs(client: PixelFlowClient, args) -> Dict[str, Any]:
    prompt_id = args.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ToolError("'prompt_id' is required")
    return _describe(client, prompt_id, client.request("GET", f"/api/history/{urllib.parse.quote(prompt_id)}"))


HANDLERS = {
    "list_workflows": list_workflows,
    "run_workflow": run_workflow,
    "queue_status": queue_status,
    "get_outputs": get_outputs,
}


# --------------------------------------------------------------- protocol


def _result(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id, code: int, message: str):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(message: Dict[str, Any], connect=find_server) -> Optional[Dict[str, Any]]:
    """One JSON-RPC message -> its response (None for a notification)."""
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:  # notifications/initialized, notifications/cancelled, ...
        return None
    params = message.get("params") or {}
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return _result(request_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "pixelflow", "title": "PixelFlow", "version": app_version()},
            "instructions": "Runs the user's saved PixelFlow image-generation workflows on this machine. "
            "Call list_workflows first; run_workflow waits for the images and returns their file paths.",
        })
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        return _result(request_id, {"tools": TOOLS})
    if method == "tools/call":
        handler = HANDLERS.get(params.get("name"))
        if handler is None:
            return _error(request_id, -32602, f"Unknown tool: {params.get('name')}")
        try:
            payload = handler(connect(), params.get("arguments") or {})
            return _result(request_id, {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                                        "structuredContent": payload, "isError": False})
        except ToolError as exc:
            return _result(request_id, {"content": [{"type": "text", "text": str(exc)}], "isError": True})
    return _error(request_id, -32601, f"Method not found: {method}")


def serve(stdin=None, stdout=None, connect=find_server) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            response = _error(None, -32700, "Parse error")
        else:
            response = handle(message, connect) if isinstance(message, dict) else _error(None, -32600, "Invalid Request")
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def main() -> int:
    # stdout carries the protocol: make sure nothing else (a stray print) and
    # no platform newline translation corrupt it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
