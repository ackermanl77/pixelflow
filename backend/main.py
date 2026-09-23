import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine.queue import ConnectionManager, ExecutionQueue
from .nodes import NODE_REGISTRY
from .paths import FRONTEND_DIR, OUTPUT_DIR, PathTraversalError, UPLOAD_DIR, WORKFLOW_DIR, safe_path

app = FastAPI(title="PixelFlow")
manager = ConnectionManager()
queue = ExecutionQueue(manager)


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
async def post_queue(payload: dict):
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(400, "Missing 'prompt'")
    prompt_id = await queue.submit(prompt)
    return {"prompt_id": prompt_id}


@app.get("/api/history")
async def get_history():
    return queue.history


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        dest = safe_path(UPLOAD_DIR, Path(file.filename).name)
    except PathTraversalError:
        raise HTTPException(400, "Invalid filename")
    with open(dest, "wb") as f:
        f.write(await file.read())
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
async def save_workflow(name: str, payload: dict):
    try:
        path = safe_path(WORKFLOW_DIR, f"{name}.json")
    except PathTraversalError:
        raise HTTPException(400, "Invalid workflow name")
    path.write_text(json.dumps(payload, indent=2))
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
