import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import contact_sheet, model_downloads, model_library, paths, plugins, update_check
from .config import APP_ID, app_version, test_mode
from .engine.graph import PromptValidationError, validate_prompt
from .engine.queue import ConnectionManager, ExecutionQueue
from .logging_setup import log_runtime_info, setup_logging
from .nodes import NODE_ALIASES, NODE_REGISTRY, io_nodes, models, runtime
from .nodes.io_nodes import list_uploads
from .paths import (
    FRONTEND_DIR,
    HISTORY_PATH,
    PathTraversalError,
    UPLOAD_DIR,
    WORKFLOW_DIR,
    safe_path,
    sanitize_filename,
)
from .security import ALLOWED_HOSTS, LocalOriginMiddleware, MaxBodySizeMiddleware

# Handlers (stderr + a rotating logs/pixelflow.log) are attached by
# setup_logging() at startup, not at import, so importing this module
# (e.g. from the tests) never creates a log file.
logger = logging.getLogger(__name__)

manager = ConnectionManager()
# Finished prompts are appended to history.jsonl and restored at startup.
queue = ExecutionQueue(manager, history_path=HISTORY_PATH)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
# Cap decoded images too: compressed size alone does not bound memory use.
MAX_UPLOAD_DIMENSION = 16_384
MAX_UPLOAD_PIXELS = 40_000_000
# Any other request body (prompts, workflows) - see security.MaxBodySizeMiddleware.
MAX_REQUEST_BYTES = 5 * 1024 * 1024
ALLOWED_UPLOAD_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
}


def clear_temp_dir(folder: Optional[Path] = None) -> int:
    """Delete Preview Image's throwaway files (temp/ is emptied at every
    start). Returns how many files were removed."""
    folder = Path(folder or io_nodes.TEMP_DIR)
    removed = 0
    if not folder.is_dir():
        return 0
    for path in folder.iterdir():
        try:
            if path.is_file() and not path.name.startswith("."):
                path.unlink()
                removed += 1
        except OSError:
            logger.debug("could not remove %s", path, exc_info=True)
    return removed


def load_plugins() -> None:
    """F22: arbitrary code from the user's custom_nodes folder, imported at
    startup before the first /api/nodes (a broken plugin is logged, never
    fatal). Off with PIXELFLOW_DISABLE_CUSTOM_NODES=1 or settings.json."""
    if plugins.custom_nodes_enabled():
        plugins.load_custom_nodes()
    else:
        logger.info("custom node plugins disabled")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log_file = setup_logging()
    if log_file is not None:
        logger.info("logging to %s", log_file)
    logger.info("PixelFlow %s, data folder %s, models folder %s", app_version(), paths.ROOT_DIR, paths.MODELS_DIR)
    paths.ensure_data_dirs()
    clear_temp_dir()
    load_plugins()
    restored = queue.load_history()
    if restored:
        logger.info("restored %d history entries from %s", restored, queue.history_path)
    cached = queue.output_cache.attach_disk(paths.OUTPUT_CACHE_DIR, app_version())
    if cached:
        logger.info("found %d cached node outputs in %s", cached, paths.OUTPUT_CACHE_DIR)
    queue.start()
    if test_mode():
        logger.warning("test mode (PIXELFLOW_TEST_MODE): torch is not loaded and models can't be")
    else:
        # torch/diffusers load in the background (report item P6): the UI is
        # usable right away and the first prompt doesn't pay for the import.
        runtime.warm_up_in_background(lambda: log_runtime_info(logger))
    # Does nothing until update_check.UPDATE_FEED_URL is set; /api/update reports the result.
    update_check.check_in_background(app_version(), _remember_update)
    yield


app = FastAPI(title="PixelFlow", lifespan=lifespan)
# See backend/security.py. add_middleware() wraps outermost-last, so the
# Host check runs first, then the Origin check, then the body-size cap.
app.add_middleware(
    MaxBodySizeMiddleware,
    default_limit=MAX_REQUEST_BYTES,
    limits={"/api/upload": lambda: MAX_UPLOAD_BYTES + 1024 * 1024},
)
app.add_middleware(LocalOriginMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


class NodeSpec(BaseModel):
    """One node in a ComfyUI-style prompt graph."""

    class_type: str
    inputs: Dict[str, Any] = Field(default_factory=dict)


class QueueRequest(BaseModel):
    prompt: Dict[str, NodeSpec]
    # ComfyUI's shape: {"extra_pnginfo": {"workflow": <litegraph graph>}} -
    # embedded into saved PNGs next to the prompt itself.
    extra_data: Dict[str, Any] = Field(default_factory=dict)
    # Run before everything already pending (Ctrl+Shift+Enter).
    front: bool = False
    # The browser window queueing it (each window shows its own prompts).
    client_id: Optional[str] = Field(None, max_length=64)


class WorkflowPayload(BaseModel):
    """A litegraph.js workflow graph. Its shape is owned by the frontend, so
    it's kept intentionally open rather than modeled field-by-field."""

    model_config = ConfigDict(extra="allow")


class RenameRequest(BaseModel):
    new_name: str


class InterruptRequest(BaseModel):
    prompt_id: Optional[str] = None


class OutputRef(BaseModel):
    filename: str
    type: str = "output"


def _node_info() -> dict:
    info = {}
    for name, cls in NODE_REGISTRY.items():
        info[name] = {
            "display_name": cls.DISPLAY_NAME,
            "category": cls.CATEGORY,
            "description": getattr(cls, "DESCRIPTION", "") or "",
            "input_types": {
                key: value for key, value in cls.INPUT_TYPES().items() if key in ("required", "optional")
            },
            "return_types": list(cls.RETURN_TYPES),
            "return_names": list(cls.RETURN_NAMES),
            "output_node": bool(getattr(cls, "OUTPUT_NODE", False)),
            # The custom_nodes plugin that registered it (F22), or None for a built-in node.
            "plugin": getattr(cls, "PLUGIN", None),
        }
    return info


@app.get("/api/health")
async def health():
    return {"app": APP_ID, "status": "ok", "version": app_version()}


_found_update: Optional[update_check.UpdateInfo] = None


def _remember_update(info: update_check.UpdateInfo) -> None:
    global _found_update
    _found_update = info


@app.get("/api/update")
async def update():
    """A newer release found by the once-per-launch check (backend/update_check.py), or {}."""
    info = _found_update
    return {} if info is None else {"version": info.version, "url": info.url, "notes": info.notes}


@app.get("/api/nodes")
def get_nodes():
    return _node_info()


@app.get("/api/node_aliases")
async def get_node_aliases():
    """Old node class names -> current ones (for loading old workflows)."""
    return dict(NODE_ALIASES)


@app.post("/api/queue")
async def post_queue(request: QueueRequest):
    if not request.prompt:
        raise HTTPException(400, "Missing 'prompt'")
    prompt = {node_id: spec.model_dump() for node_id, spec in request.prompt.items()}
    try:
        prompt = validate_prompt(prompt)
    except PromptValidationError as exc:
        # {"detail": {"message", "node_id", "class_type", "input"}} so the
        # frontend can mark the offending node and input.
        raise HTTPException(400, exc.to_dict()) from None
    extra_pnginfo = request.extra_data.get("extra_pnginfo") or {}
    if not isinstance(extra_pnginfo, dict):
        raise HTTPException(400, "'extra_data.extra_pnginfo' must be an object")
    prompt_id = await queue.submit(prompt, extra_pnginfo, front=request.front, client_id=request.client_id)
    return {"prompt_id": prompt_id}


@app.post("/api/interrupt")
async def interrupt_prompt(request: Optional[InterruptRequest] = Body(None)):
    """Stop the running prompt at its next node or diffusion step. With
    {"prompt_id"}, only if that prompt is the one running (409 otherwise),
    so a stale Cancel button can't stop a different prompt."""
    prompt_id = request.prompt_id if request is not None else None
    interrupted = queue.interrupt(prompt_id)
    if prompt_id is not None and interrupted is None:
        raise HTTPException(409, f"Prompt '{prompt_id}' is not running")
    return {"interrupted": interrupted}


@app.get("/api/queue")
async def get_queue():
    """{"running": prompt_id | null, "pending": [prompt_id, ...] in run order}
    - lets a reloaded or reconnected page rebuild its queue panel."""
    return queue.queue_state()


@app.delete("/api/queue")
async def cancel_all_pending(client_id: Optional[str] = Query(None, max_length=64)):
    """Cancel every prompt that hasn't started yet (only `client_id`'s
    prompts, plus those queued without one, when given)."""
    cancelled = queue.cancel_all(client_id)
    for prompt_id in cancelled:
        await manager.broadcast({"type": "execution_cancelled", "prompt_id": prompt_id})
    return {"cancelled": cancelled}


@app.get("/api/history")
async def get_history(
    limit: Optional[int] = Query(None, ge=1, le=1000),
    before: Optional[str] = Query(None, max_length=64),
):
    """History entries keyed by prompt id, oldest first. `limit` keeps only
    the newest N; `before` (a prompt id) pages further back."""
    return queue.history_page(limit=limit, before=before)


@app.delete("/api/history")
async def clear_finished_history(client_id: Optional[str] = Query(None, max_length=64)):
    """Delete every finished entry (queued/running ones stay); only
    `client_id`'s, plus those queued without one, when given."""
    return {"deleted": queue.clear_finished(client_id)}


@app.get("/api/history/{prompt_id}")
async def get_history_entry(prompt_id: str):
    entry = queue.history.get(prompt_id)
    if entry is None:
        raise HTTPException(404, "History entry not found")
    return entry


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


def _open_unique(stem: str, suffix: str):
    """Create (exclusively) a new file in UPLOAD_DIR named stem+suffix, or
    stem_1+suffix, stem_2+suffix, ... if taken, so an upload never silently
    overwrites an earlier one. Returns (path, open binary file)."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(10_000):
        name = f"{stem}{suffix}" if attempt == 0 else f"{stem}_{attempt}{suffix}"
        dest = safe_path(UPLOAD_DIR, name)
        try:
            return dest, open(dest, "xb")
        except FileExistsError:
            continue
    raise HTTPException(409, "Too many uploads with this filename")


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported content type '{file.content_type}'")
    raw_name = Path(file.filename or "").name.strip()
    if not raw_name:
        raise HTTPException(400, "Missing filename")
    # Sanitize stem and extension separately so a long name can't truncate
    # the extension away, and characters like ':' can't create an NTFS
    # alternate data stream.
    raw_path = Path(raw_name)
    stem = sanitize_filename(raw_path.stem, default="upload")
    suffix = sanitize_filename(raw_path.suffix, default="")
    suffix = f".{suffix.lower()}" if suffix else ""

    source = file.file
    source.seek(0, io.SEEK_END)
    total = source.tell()
    source.seek(0)
    if total > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds max upload size of {MAX_UPLOAD_BYTES} bytes")
    if not total:
        raise HTTPException(400, "Empty file")

    # The content-type header is client-controlled; make sure the bytes
    # really are an image Pillow can decode before storing them.
    try:
        with Image.open(source) as img:
            width, height = img.size
            if (
                width > MAX_UPLOAD_DIMENSION
                or height > MAX_UPLOAD_DIMENSION
                or width * height > MAX_UPLOAD_PIXELS
            ):
                raise HTTPException(
                    413,
                    f"Image dimensions exceed the {MAX_UPLOAD_DIMENSION}px or {MAX_UPLOAD_PIXELS}-pixel limit",
                )
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "File is not a valid image") from None
    finally:
        source.seek(0)

    try:
        dest, handle = _open_unique(stem, suffix)
    except PathTraversalError:
        raise HTTPException(400, "Invalid filename") from None
    with handle:
        shutil.copyfileobj(source, handle, length=1024 * 1024)
    return {"filename": dest.name}


@app.get("/api/uploads")
async def get_uploads():
    """Filenames available to Load Image / Load Mask (sorted)."""
    return list_uploads()


@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    """Serve an uploaded image (the Load Image node's thumbnail). A route
    rather than a StaticFiles mount so it follows UPLOAD_DIR at request time."""
    try:
        path = safe_path(UPLOAD_DIR, filename)
    except PathTraversalError:
        raise HTTPException(404, "Not found") from None
    if not path.is_file() or path.name.startswith("."):
        raise HTTPException(404, "Not found")
    return FileResponse(path)


# ---- generated images (outputs/ and the temp/ previews) ----


def _image_folder(kind: str) -> Path:
    if kind == "output":
        return io_nodes.OUTPUT_DIR
    if kind == "temp":
        return io_nodes.TEMP_DIR
    raise HTTPException(400, "type must be 'output' or 'temp'")


def _generated_file(kind: str, filename: str) -> Path:
    try:
        path = safe_path(_image_folder(kind), filename)
    except PathTraversalError:
        raise HTTPException(404, "Not found") from None
    if not path.is_file() or path.name.startswith(".") or path.suffix.lower() != ".png":
        raise HTTPException(404, "Not found")
    return path


@app.get("/outputs/{filename:path}")
async def get_output(filename: str):
    return FileResponse(_generated_file("output", filename))


@app.get("/temp/{filename:path}")
async def get_temp(filename: str):
    return FileResponse(_generated_file("temp", filename), headers={"Cache-Control": "no-store"})


def _unlink(path: Path) -> None:
    """Delete `path`; a file another program holds open becomes a 409 that
    names that program (R8, Windows Restart Manager)."""
    try:
        path.unlink()
    except PermissionError as error:
        from .diagnostics import explain_delete_failure

        raise HTTPException(409, explain_delete_failure(path, error)) from error


@app.delete("/api/outputs/{filename:path}")
async def delete_output(filename: str, type: str = Query("output")):  # noqa: A002 - API field name
    """Delete a generated image (gallery "Delete")."""
    path = _generated_file(type, filename)
    _unlink(path)
    return {"deleted": filename}


@app.post("/api/outputs/to-upload")
async def output_to_upload(ref: OutputRef):
    """Copy a generated image into uploads/ (gallery "Send to Load Image")
    and return its upload filename."""
    source = _generated_file(ref.type, ref.filename)
    stem = sanitize_filename(Path(ref.filename).stem, default="output")
    dest, handle = _open_unique(stem, ".png")
    with handle, open(source, "rb") as src:
        shutil.copyfileobj(src, handle)
    return {"filename": dest.name}


class ContactSheetRequest(BaseModel):
    """An X/Y plot's results (F18): one cell per grid point, row-major (null:
    the run failed), plus the axis labels."""

    cells: list[Optional[OutputRef]] = Field(..., min_length=1, max_length=contact_sheet.MAX_CELLS)
    columns: int = Field(..., ge=1, le=contact_sheet.MAX_CELLS)
    x_values: list[str] = Field(..., max_length=contact_sheet.MAX_CELLS)
    y_values: list[str] = Field(..., max_length=contact_sheet.MAX_CELLS)
    x_label: str = Field("", max_length=200)
    y_label: str = Field("", max_length=200)


@app.post("/api/contact_sheet")
def make_contact_sheet(request: ContactSheetRequest):
    """Lay an X/Y plot's images out in a labelled grid and save it to
    outputs/xy_plot/ (the gallery picks it up like any output)."""
    images = []
    for ref in request.cells:
        if ref is None:
            images.append(None)
            continue
        with Image.open(_generated_file(ref.type, ref.filename)) as opened:
            images.append(opened.convert("RGB"))
    try:
        sheet = contact_sheet.compose(
            images, request.columns, [v[:60] for v in request.x_values], [v[:60] for v in request.y_values],
            request.x_label, request.y_label,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from None
    filename = io_nodes.save_numbered(sheet, "xy_plot/xy_plot", io_nodes.OUTPUT_DIR)
    return {"filename": filename, "type": "output"}


# ---- models and system ----


class DownloadRequest(BaseModel):
    url: str = Field(..., max_length=2000)
    kind: str = "auto"


@app.post("/api/downloads")
def start_download(request: DownloadRequest):
    """F23: download a model from a Hugging Face file link or a Civitai model
    page into models/<kind>/ ("auto" picks the folder), SHA-256 checked, in
    the background. Returns the job; GET /api/downloads reports progress."""
    if request.kind != "auto" and request.kind not in model_downloads.MODEL_FOLDERS:
        raise HTTPException(400, f"Unknown model folder '{request.kind}'")
    return model_downloads.MANAGER.start(request.url, request.kind)


@app.get("/api/downloads")
def list_downloads():
    return model_downloads.MANAGER.list()


@app.delete("/api/downloads/{job_id}")
def cancel_download(job_id: str):
    if not model_downloads.MANAGER.cancel(job_id):
        raise HTTPException(404, "No running download with that id")
    return {"cancelled": job_id}


@app.get("/api/model-library")
def get_model_library():
    """The Model Library panel (UI13): installed models by kind, with size, preview and loader."""
    return model_library.library()


@app.get("/api/model-library/preview")
def get_model_preview(kind: str = Query(...), name: str = Query(...)):
    path = model_library.preview_path(kind, name)
    if path is None:
        raise HTTPException(404, "No preview image for that model")
    return FileResponse(path, headers={"Cache-Control": "max-age=300"})


@app.get("/api/models/{kind}")
def get_models(kind: str):
    """Installed models of a kind (models.MODEL_KINDS): e.g. checkpoints
    (files and diffusers folders in models/checkpoints/, plus diffusers
    pipelines in the Hugging Face cache), loras, embeddings, and the files a
    ComfyUI library adds (backend/model_roots.py)."""
    try:
        return models.list_models(kind)
    except KeyError:
        raise HTTPException(404, f"Unknown model kind '{kind}' (use one of {sorted(models.MODEL_KINDS)})") from None


def _system_info() -> Dict[str, Any]:
    if test_mode():  # never import torch in the lightweight test server
        return {
            "app": APP_ID,
            "version": app_version(),
            "python": platform.python_version(),
            "torch": None,
            "cuda": None,
            "diffusers": None,
            "device": "cpu",
            "device_name": "test mode",
            "vram_free": None,
            "vram_total": None,
            "loaded_models": [],
            "data_dir": str(paths.ROOT_DIR),
            "models_dir": str(paths.MODELS_DIR),
            "log_dir": str(paths.LOG_DIR),
            "custom_nodes": plugins.describe(),
            "test_mode": True,
        }
    t = runtime.torch()
    info: Dict[str, Any] = {
        "app": APP_ID,
        "version": app_version(),
        "python": platform.python_version(),
        "torch": t.__version__,
        "cuda": t.version.cuda,
        "device": "cpu",
        "device_name": platform.processor() or "CPU",
        "vram_free": None,
        "vram_total": None,
        "loaded_models": models.PIPELINES.describe(),
        "data_dir": str(paths.ROOT_DIR),
        "models_dir": str(paths.MODELS_DIR),
        "log_dir": str(paths.LOG_DIR),
        "custom_nodes": plugins.describe(),
    }
    try:
        import diffusers

        info["diffusers"] = diffusers.__version__
    except Exception:  # noqa: BLE001 - informational
        info["diffusers"] = None
    if t.cuda.is_available():
        free, total = t.cuda.mem_get_info()
        info.update(
            device="cuda",
            device_name=t.cuda.get_device_name(0),
            vram_free=int(free),
            vram_total=int(total),
            vram_allocated=int(t.cuda.memory_allocated()),
        )
    return info


@app.get("/api/system")
def get_system():
    """Versions, device, VRAM and the loaded models (top bar / System panel)."""
    return _system_info()


@app.get("/api/mcp")
def get_mcp_config():
    """F27: how to connect an assistant (Claude Desktop, Claude Code...) to
    PixelFlow's MCP server -- the command, a copyable config block and the
    Claude Code command line."""
    from . import mcp_server

    return mcp_server.client_config()


@app.post("/api/free")
def free_models():
    """Unload every cached pipeline and node output and give the memory back
    (e.g. to use the GPU for something else)."""
    unloaded = models.PIPELINES.clear()
    queue.output_cache.clear()
    from .nodes import generation

    generation.clear_prompt_cache()
    runtime.free_memory()
    return {"unloaded": unloaded}


def _open_folder(folder: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(folder))  # noqa: S606 - a fixed local folder, not user input
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])  # noqa: S603,S607
    else:
        subprocess.Popen(["xdg-open", str(folder)])  # noqa: S603,S607


@app.post("/api/logs/open")
def open_log_folder():
    """Open the log folder in the file manager (desktop use; the server
    only listens on loopback)."""
    folder = paths.LOG_DIR
    try:
        folder.mkdir(parents=True, exist_ok=True)
        _open_folder(folder)
    except OSError as exc:
        raise HTTPException(500, f"Could not open {folder}: {exc}") from None
    return {"opened": str(folder)}


# ---- workflows ----

# Workflow names become filenames, so they're validated rather than trusted:
# letters/digits/underscore first, then also spaces, '.', '-', '(' and ')',
# at most 100 characters, no trailing dot or space (Windows strips those,
# which made "a." and "a" look-alike files), and none of Windows' reserved
# device names. Anything else is a 400 instead of an OS-level 500.
_WORKFLOW_NAME = re.compile(r"[\w][\w .()-]{0,99}")
_RESERVED_WINDOWS_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)}
INVALID_WORKFLOW_NAME = (
    "Invalid workflow name: use 1-100 letters, digits, spaces, '_', '-', '.', '(' or ')', "
    "starting with a letter or digit and not ending in a dot or space"
)


def is_valid_workflow_name(name: str) -> bool:
    if not _WORKFLOW_NAME.fullmatch(name) or name.endswith((".", " ")):
        return False
    return name.split(".")[0].rstrip().upper() not in _RESERVED_WINDOWS_NAMES


def _workflow_path(name: str) -> Path:
    if not is_valid_workflow_name(name):
        raise HTTPException(400, INVALID_WORKFLOW_NAME)
    try:
        return safe_path(WORKFLOW_DIR, f"{name}.json")
    except PathTraversalError:
        raise HTTPException(400, INVALID_WORKFLOW_NAME) from None


@app.get("/api/workflows")
async def list_workflows():
    """[{name, modified (unix time), size (bytes)}], sorted by name."""
    if not WORKFLOW_DIR.is_dir():
        return []
    items = []
    for path in WORKFLOW_DIR.glob("*.json"):
        if not is_valid_workflow_name(path.stem):
            continue
        stat = path.stat()
        items.append({"name": path.stem, "modified": stat.st_mtime, "size": stat.st_size})
    return sorted(items, key=lambda item: item["name"].lower())


@app.get("/api/workflows/{name}")
async def load_workflow(name: str):
    path = _workflow_path(name)
    if not path.is_file():
        raise HTTPException(404, "Workflow not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(422, f"Workflow '{name}' is not valid JSON") from None
    return JSONResponse(data)


@app.post("/api/workflows/{name}")
async def save_workflow(name: str, payload: WorkflowPayload, overwrite: bool = Query(True)):
    """Save a workflow. With ?overwrite=false an existing one is a 409, so
    the UI can ask before replacing it."""
    path = _workflow_path(name)
    if path.exists() and not overwrite:
        raise HTTPException(409, f"A workflow named '{name}' already exists")
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"saved": name}


@app.delete("/api/workflows/{name}")
async def delete_workflow(name: str):
    path = _workflow_path(name)
    if not path.is_file():
        raise HTTPException(404, "Workflow not found")
    _unlink(path)
    return {"deleted": name}


@app.post("/api/workflows/{name}/rename")
async def rename_workflow(name: str, request: RenameRequest, overwrite: bool = Query(False)):
    source = _workflow_path(name)
    target = _workflow_path(request.new_name)
    if not source.is_file():
        raise HTTPException(404, "Workflow not found")
    if target.exists() and not overwrite and target.name.lower() != source.name.lower():
        raise HTTPException(409, f"A workflow named '{request.new_name}' already exists")
    source.replace(target)
    return {"renamed": name, "to": request.new_name}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


class NoCacheStaticFiles(StaticFiles):
    """The frontend's own files are revalidated on every load (a 304 when
    unchanged), so an updated app.js is never served from a stale browser
    cache - no more manual ?v=N cache-busting in index.html."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=str(FRONTEND_DIR), html=True, check_dir=False), name="frontend")
