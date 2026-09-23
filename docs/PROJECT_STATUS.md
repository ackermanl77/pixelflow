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
  (`queued`/`executing`/`execution_success`/`execution_error`, plus
  per-step `progress` messages — see below) streams to the browser over
  `/ws`.
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
  sidebar (with a live substring-filter search box over `#node-list`'s
  items, `filterNodeList()`), prompt building/queueing, the WebSocket-driven
  queue/gallery panels (`#gallery` is a responsive CSS grid, not one
  full-width image per row; clicking a thumbnail opens a click-to-close
  full-screen lightbox), undo/redo (Ctrl+Z/Ctrl+Y, debounced graph
  snapshots), and workflow save/load (backend-persisted JSON, plus file
  export/import).
- **`frontend/vendor/`** — litegraph.js and its CSS, vendored locally (not
  loaded from a CDN) so the packaged desktop app works fully offline.
- **`desktop.py`** — runs the FastAPI server in a background thread (uvicorn
  only installs OS signal handlers on the main thread, so this is safe) and
  opens it in a pywebview native window.

## Nodes included

- **loaders**: Load Checkpoint, Load Image
- **conditioning**: CLIP Text Encode
- **sampling**: KSampler
- **image** (real, CPU-only, Pillow-based): Empty Image, Resize, Blur,
  Brightness/Contrast, Hue/Saturation, Blend, Text Overlay
- **output**: Preview Image, Save Image

`LoadCheckpoint`/`CLIPTextEncode`/`KSampler` (`backend/nodes/io_nodes.py`,
`backend/nodes/generation.py`) run real Stable Diffusion inference via
`diffusers`, no longer a deterministic placeholder. `LoadCheckpoint`
resolves `ckpt_name` to, in order: a local `.safetensors`/`.ckpt` file under
`models/checkpoints/`, a local diffusers-format folder there, or otherwise a
Hugging Face Hub repo id (downloaded and cached on first use — needs
internet access the first time only). Loaded pipelines are cached in-process
per `(ckpt_name, device)` since constructing one is expensive. Runs on GPU
automatically if CUDA is available (`torch.cuda.is_available()`), CPU
otherwise — CPU inference works but is slow (expect well over a minute per
image for a full-size SD1.x model at 20 steps; the default checkpoint,
`stabilityai/sd-turbo`, is a distilled few-step model chosen so the
out-of-the-box experience is usable on CPU). See
[NODE_AUTHORING.md](NODE_AUTHORING.md) for the node internals.

`CLIPTextEncode` still just wraps the prompt string as `{"text": ...}`
rather than running a separate embedding step: `KSampler` passes that string
straight to the loaded pipeline's own `prompt=` argument, which does its own
tokenization/CLIP-encoding internally. The same node class doubles as the
negative prompt: wire a second `CLIPTextEncode` into `KSampler`'s optional
`negative` input and its text reaches the pipeline's `negative_prompt=`, the
same two-nodes-into-one-sampler shape real ComfyUI uses. `KSampler` also
exposes `sampler` (a dropdown swapping the pipeline's diffusers scheduler via
`SchedulerClass.from_config(pipe.scheduler.config)` — Euler, Euler
Ancestral, DPM-Solver Multistep, PNDM, DDIM, LMS, Heun), `seed` (>= 0
reproduces the same output for a given sampler/prompt/steps/cfg; the default
`-1` picks a fresh random seed each run), and `batch_size` (passed as
`num_images_per_prompt=`, generating more than one image per queue
submission). `KSampler`'s output is always a list of one or more PIL images
now (even at `batch_size == 1`), and `SaveImage`/`PreviewImage` save every
image in that list; other IMAGE-producing nodes (`EmptyImage`, `LoadImage`,
the `image_ops` nodes) are unaffected and still hand those two nodes a
single `PIL.Image` directly. `latent_image` (from `EmptyImage`) only
supplies the output width/height — there's no real VAE latent or
image-to-image/inpainting support, every run is txt2img from scratch.

`KSamplerStub.sample()` also accepts an optional `on_progress` kwarg, passed
as diffusers' `callback_on_step_end` to the pipeline call and invoked once
per denoising step as `on_progress(step, total_steps)` (1-based). This isn't
part of every node's call signature — `execute_prompt()`
(`backend/engine/graph.py`) only injects it, via `inspect.signature()`, into
nodes whose `FUNCTION` declares an `on_progress` parameter, so most nodes
(which finish in one shot) are unaffected. `ExecutionQueue`
(`backend/engine/queue.py`) wires this into a `{"type": "progress",
"prompt_id", "node_id", "step", "total"}` broadcast alongside the existing
`queued`/`executing`/`execution_success`/`execution_error` messages.
`frontend/app.js`'s `connectWebSocket()` renders it as a small progress bar
docked to the bottom of the canvas (`#progress-wrap`), shown for the
currently-executing node and hidden again on the next `executing`/terminal
message (a node that never reports progress just leaves the bar hidden).

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
- **Generation is CPU-slow and first-run needs internet.** Without a CUDA
  GPU, `KSampler` runs on CPU, which is fine for a distilled few-step model
  like the default `stabilityai/sd-turbo` but slow (minutes per image) for a
  full-size SD1.x/SDXL checkpoint. The first use of any given `ckpt_name`
  that isn't already a local file under `models/checkpoints/` triggers a
  download from the Hugging Face Hub.
- **txt2img only.** There's no image-to-image or inpainting — `latent_image`
  only supplies width/height, not real pixel/latent content, so every
  `KSampler` run generates from scratch.

## Desktop packaging

See [BUILD.md](BUILD.md) for the full build/package walkthrough, including
the WebView2 Runtime bundling rationale (same approach as the sibling
photo_professional_tool project, and the same pinned runtime version reused
from its build cache when present).
