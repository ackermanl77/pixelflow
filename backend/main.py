import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .engine.queue import ConnectionManager, ExecutionQueue
from .nodes import NODE_REGISTRY
from .paths import FRONTEND_DIR, OUTPUT_DIR, PathTraversalError, UPLOAD_DIR, WORKFLOW_DIR, safe_path
from .security import ALLOWED_HOSTS, LocalOriginMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PixelFlow")
# See backend/security.py. add_middleware() wraps outermost-last, so the
# Host check runs first, then the Origin check.
app.add_middleware(LocalOriginMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
manager = ConnectionManager()
queue = ExecutionQueue(manager)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_UPLOAD_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}


class NodeSpec(BaseModel):
    """One node in a ComfyUI-style prompt graph."""

    class_type: str
    inputs: Dict[str, Any] = Field(default_factory=dict)


class QueueRequest(BaseModel):
    prompt: Dict[str, NodeSpec]


class WorkflowPayload(BaseModel):
    """A litegraph.js workflow graph. Its shape is owned by the frontend, so
    it's kept intentionally open rather than modeled field-by-field."""

    model_config = ConfigDict(extra="allow")


@app.on_event("startup")
async def _startup() -> None:
    queue.start()


def _node_info() -> dict:
    info = {}
    for name, cls in NODE_REGISTRY.items():
        info[name] = {
            "display_name": cls.DISPLAY_NAME,
            "category": cls.CATEGORY,
            "input_types": cls.INPUT_TYPES(),
            "return_types": list(cls.RETURN_TYPES),
            "return_names": list(cls.RETURN_NAMES),
            "output_node": bool(getattr(cls, "OUTPUT_NODE", False)),
        }
    return info


@app.get("/api/nodes")
async def get_nodes():
    return _node_info()


@app.post("/api/queue")
async def post_queue(request: QueueRequest):
    if not request.prompt:
        raise HTTPException(400, "Missing 'prompt'")
    prompt = {node_id: spec.model_dump() for node_id, spec in request.prompt.items()}
    prompt_id = await queue.submit(prompt)
    return {"prompt_id": prompt_id}


@app.get("/api/history")
async def get_history():
    return queue.history


@app.delete("/api/queue/{prompt_id}")
async def cancel_prompt(prompt_id: str):
    if not queue.cancel(prompt_id):
        raise HTTPException(404, "Prompt not queued (unknown, already running, or finished)")
    await manager.broadcast({"type": "execution_cancelled", "prompt_id": prompt_id})
    return {"cancelled": prompt_id}


@app.delete("/api/history/{prompt_id}")
async def delete_history_entry(prompt_id: str):
    if not queue.delete_history(prompt_id):
        raise HTTPException(404, "History entry not found, or still queued/running")
    return {"deleted": prompt_id}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported content type '{file.content_type}'")
    try:
        dest = safe_path(UPLOAD_DIR, Path(file.filename).name)
    except PathTraversalError:
        raise HTTPException(400, "Invalid filename")

    total = 0
    chunks = []
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File exceeds max upload size of {MAX_UPLOAD_BYTES} bytes")
        chunks.append(chunk)

    with open(dest, "wb") as f:
        f.writelines(chunks)
    return {"filename": dest.name}


@app.get("/api/workflows")
async def list_workflows():
    return sorted(p.stem for p in WORKFLOW_DIR.glob("*.json"))


@app.get("/api/workflows/{name}")
async def load_workflow(name: str):
    try:
        path = safe_path(WORKFLOW_DIR, f"{name}.json")
    except PathTraversalError:
        raise HTTPException(400, "Invalid workflow name")
    if not path.exists():
        raise HTTPException(404, "Workflow not found")
    return JSONResponse(json.loads(path.read_text()))


@app.post("/api/workflows/{name}")
async def save_workflow(name: str, payload: WorkflowPayload):
    try:
        path = safe_path(WORKFLOW_DIR, f"{name}.json")
    except PathTraversalError:
        raise HTTPException(400, "Invalid workflow name")
    path.write_text(json.dumps(payload.model_dump(), indent=2))
    return {"saved": name}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
