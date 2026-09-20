# PixelFlow — Project Status

Authoritative reference for architecture, entry points, and known limitations.
The root [README.md](../README.md) only covers day-to-day commands.

## What this is

A small ComfyUI-style node-graph app: a Python backend executes a graph of
image nodes, and a browser-based canvas UI (litegraph.js — the same library
ComfyUI's own frontend uses) lets you wire nodes together and run them.
It runs either as a local web app (`python run.py`) or as a native desktop
window (`python desktop.py`, or the packaged `PixelFlow.exe`).

## Entry points

| Command | What it does |
|---|---|
| `python run.py` | Runs the FastAPI/uvicorn server with `--reload` at http://127.0.0.1:8188, opened in a normal browser tab. Use this while developing. |
| `python desktop.py` | Runs the same server in a background thread and opens it in a native window via pywebview (no `--reload`; used for the packaged app). |
| `release\PixelFlow.exe` | The packaged desktop app — same as `desktop.py`, but frozen with PyInstaller and bundling its own WebView2 Runtime. Built by `scripts\build_release.ps1`. |

There is exactly one backend process and one frontend in this project — unlike
photo_professional_tool's split Streamlit-frontend/Flask-backend design,
PixelFlow's FastAPI process serves both the JSON API and the static frontend
files directly (see `backend/main.py`'s two `app.mount(...)` calls).

## Architecture

- **`backend/nodes/`** — node definitions. Each node is a class with
  `INPUT_TYPES()`, `RETURN_TYPES`, and a `FUNCTION` name, registered into
  `NODE_REGISTRY` via `@register_node`. This is ComfyUI's own node contract.
  Adding a new `.py` file's classes requires importing it in
  `backend/nodes/__init__.py` for its `@register_node` decorators to run.
- **`backend/engine/graph.py`** — `execute_prompt()` takes a flattened
  "prompt" dict (`{node_id: {class_type, inputs}}`) and resolves it
  recursively with per-node output caching. No separate topological-sort pass.
- **`backend/engine/queue.py`** — a single-worker async job queue. Submitted
  prompts run one at a time in a background thread; progress
  (`queued`/`executing`/`execution_success`/`execution_error`) streams to the
  browser over `/ws`.
- **`backend/main.py`** — the FastAPI app: `GET /api/nodes` (registry as
  JSON), `POST /api/queue`, `GET /api/history`, `POST /api/upload`,
  `GET|POST /api/workflows[/{name}]`, and the `/ws` websocket. Mounts
  `/outputs` (generated images) and `/` (the frontend) as static files.
- **`backend/paths.py`** — resolves two different roots so the packaged exe
  works correctly: `BUNDLE_DIR` (read-only bundled resources — `frontend/`,
  extracted by PyInstaller into a temp dir when frozen) vs. `ROOT_DIR`
  (writable data — `outputs/`, `uploads/`, `workflows/`, kept next to the
  actual `.exe` so they persist across runs and aren't lost with the temp
  extraction dir).
- **`frontend/app.js`** — the entire client. Fetches `/api/nodes` and builds a
  LiteGraph node class per entry (nodes are never hard-coded here — a new
  Python node shows up in the UI automatically). Handles the node-library
  sidebar, prompt building/queueing, the WebSocket-driven queue/gallery
  panels, undo/redo (Ctrl+Z/Ctrl+Y, debounced graph snapshots), and
  workflow save/load (backend-persisted JSON, plus file export/import).
- **`frontend/vendor/`** — litegraph.js and its CSS, vendored locally (not
  loaded from a CDN) so the packaged desktop app works fully offline.
- **`desktop.py`** — runs the FastAPI server in a background thread (uvicorn
  only installs OS signal handlers on the main thread, so this is safe) and
  opens it in a pywebview native window.

## Nodes included

- **loaders**: Load Checkpoint (stub), Load Image
- **conditioning**: CLIP Text Encode (stub)
- **sampling**: KSampler (stub)
- **image** (real, CPU-only, Pillow-based): Empty Image, Resize, Blur,
  Brightness/Contrast, Hue/Saturation, Blend, Text Overlay
- **output**: Preview Image, Save Image

The "stub" nodes (`backend/nodes/generation.py`, plus `LoadCheckpoint` in
`backend/nodes/io_nodes.py`) stand in for real diffusion model inference —
see [NODE_AUTHORING.md](NODE_AUTHORING.md) for what to replace to wire up
real GPU-based generation (e.g. `diffusers`) later. This is a deliberate
scope decision, not a missing feature: there's no GPU on this dev machine, so
the graph shape and execution engine are built and tested end to end without
needing one.

## Known limitations

- **No node validation.** The engine doesn't check that a socket's declared
  type matches what's connected (e.g. wiring an `IMAGE` output into a
  `CONDITIONING` input fails at execution time with whatever Python error
  the node function raises, not with a friendlier upfront graph-validation
  error).
- **`window.prompt()`** is used for naming a workflow on **Save Workflow**.
  This doesn't render in every embedded/automated browser context (it did
  not appear in this project's own automated browser-based testing), but
  works normally in a real browser tab and in the packaged desktop app's
  WebView2 window.
- **Single global queue.** There's no per-client isolation — every connected
  browser tab/window sees every queued prompt's progress.
- **No authentication.** The server binds to `127.0.0.1` only; it isn't meant
  to be exposed beyond localhost.

## Desktop packaging

See [BUILD.md](BUILD.md) for the full build/package walkthrough, including
the WebView2 Runtime bundling rationale (same approach as the sibling
photo_professional_tool project, and the same pinned runtime version reused
from its build cache when present).
