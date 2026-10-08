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
| `python run.py` | Runs the FastAPI/uvicorn server with `--reload` at http://127.0.0.1:8288, opened in a normal browser tab. Use this while developing. |
| `python desktop.py` | Runs the same server in a background thread and opens it in a native window via pywebview (no `--reload`; used for the packaged app). |
| `release\PixelFlow.exe` | The packaged desktop app — same as `desktop.py`, but frozen with PyInstaller and bundling its own WebView2 Runtime. Built by `scripts\build_release.ps1`. |
| `python tests/e2e/server.py [port]` | The backend in **test mode** (`PIXELFLOW_TEST_MODE=1`) on a throwaway data folder (`<temp>\pixelflow-e2e-<port>`, default port 8299) — what `npm run e2e` starts. |

**Test mode** (`PIXELFLOW_TEST_MODE=1`, `backend/config.py::test_mode()`):
torch/diffusers are never imported (no background warm-up, `/api/system`
reports device `cpu` / "test mode" and `test_mode: true`), loading a model
fails with a clear error, and torch-free graphs (Empty Image → Save Image, …)
run normally. It needs only the web dependencies and starts in about a second.

The port (8288, not ComfyUI's 8188) and the app id live in
`backend/config.py`; `PIXELFLOW_PORT` overrides the port. In desktop mode, if
the port is taken by another program a free port is used instead, and if
another PixelFlow already serves it, the second launch brings the running
instance's window to the front (restoring it if minimized) instead of opening
another window (`desktop.choose_port()`, `_focus_existing_window()`). Closing the window interrupts the running
prompt and lets uvicorn shut down (up to 10 s) so no PNG is cut off.

There is exactly one backend process and one frontend in this project —
PixelFlow's FastAPI process serves both the JSON API and the static frontend
files directly.

## Where data lives (`backend/paths.py`)

- **Bundled/read-only** (`frontend/`) — `sys._MEIPASS` when frozen, the repo
  otherwise.
- **Writable data** (`outputs/`, `uploads/`, `workflows/`, `temp/`,
  `history.jsonl`, `logs/`, `settings.json`) — the first of:
  `PIXELFLOW_HOME`; the project folder when running from source; the exe's
  folder when it holds a `portable.txt` marker (or data from an older
  version — an `outputs` folder or `history.jsonl`); otherwise
  `%LOCALAPPDATA%\PixelFlow`. So an install under `C:\Program Files` still
  starts. Nothing is created at import time; `ensure_data_dirs()` runs at
  startup and only logs failures.
- **Models** (`models/checkpoints/`, `models/loras/`) — under the data
  folder, unless `PIXELFLOW_MODELS_DIR` or `{"models_dir": "..."}` in
  `settings.json` points elsewhere (e.g. a big, non-synced drive).

## Architecture

- **`backend/nodes/`** — node definitions. Each node is a class with
  `INPUT_TYPES()`, `RETURN_TYPES`, and a `FUNCTION` name, registered into
  `NODE_REGISTRY` via `@register_node` (ComfyUI's node contract), plus the
  optional `DESCRIPTION`, `IS_CHANGED`, `VALIDATE_INPUTS` and per-input
  `tooltip`s. `NODE_ALIASES` (`base.py`) maps old class names to new ones
  (`KSamplerStub` → `KSampler`), applied by validation, execution and the
  frontend, so saved workflows and PNG metadata keep working after a rename.
  Node declarations don't import torch/diffusers (`nodes/runtime.py`), so the
  server is up in well under a second; `main.py` warms the import up on a
  background thread at startup.
- **`backend/nodes/models.py`** — model discovery (`list_models()`: files and
  diffusers folders under `models/<kind>/`, plus diffusers pipelines in the
  Hugging Face cache), architecture detection (model_index.json's pipeline
  class for folders/hub repos; state-dict keys via diffusers'
  `infer_diffusers_model_type` for single files: SD 1.x/2.x, SDXL, SD3,
  Flux), Hub downloads (`ensure_hub_snapshot()`: only the files the pipeline
  loads, byte progress to the node's `on_progress`, a clear error when
  offline), precision (`auto`/`fp16`/`bf16`/`fp32`) and the pipeline cache
  `PIPELINES`: pipelines stay on the GPU while they fit into free VRAM minus
  a reserve (`PIXELFLOW_VRAM_RESERVE_GB`, default 3), older ones move to a
  CPU-RAM tier (`PIXELFLOW_RAM_CACHE_GB`, default 8) and back on reuse, at
  most 4 in total; a model bigger than the GPU budget on its own runs with
  `enable_model_cpu_offload()`. UNet/VAE use `channels_last` on CUDA.
- **`backend/engine/graph.py`** — `validate_prompt()` keeps only the output
  nodes (`OUTPUT_NODE = True`) and their ancestors, rejects a prompt with no
  output node, checks structure and values, then calls each node's
  `VALIDATE_INPUTS()` (e.g. colours, missing uploads), and raises
  `PromptValidationError` carrying `node_id`, `class_type` and `input_name`.
  A combo with `allow_custom` (Load Checkpoint's `ckpt_name`) also accepts
  values outside its list. `execute_prompt()` resolves the prompt from the
  output nodes; with an `OutputCache` a node whose signature (class, widget
  values, upstream signatures, `IS_CHANGED()`) matches the previous run is
  reused. Node exceptions become `NodeExecutionError`. Nodes that take
  `on_progress` get a callback that adds `it_per_s`/`eta_s` and forwards
  live previews when the caller passes `on_progress_detail`.
- **`backend/engine/queue.py`** — a single-worker async queue (a deque, so
  `front=True` jumps ahead). Messages over `/ws`: `queued`, `execution_start`,
  `executing` (with `node_index`/`node_count`), `execution_cached`,
  `progress` (`step`, `total`, `it_per_s`, `eta_s`, `unit`), `preview`
  (a JPEG data URL of the latents), `execution_success`,
  `execution_interrupted`, `execution_cancelled`, `execution_error`
  (`node_id`, `class_type`, `exception_type`, `traceback`). History entries
  record the prompt, the workflow, status, outputs/error and timestamps;
  finished entries are appended to `history.jsonl` and the newest 200 are
  restored at startup. `interrupt(prompt_id)` only stops that prompt.
- **`backend/main.py`** — the FastAPI app (lifespan startup: logging, data
  folders, emptying `temp/`, restoring history, warming torch up):
  - `GET /api/health` (`{"app", "status", "version"}`), `GET /api/nodes`,
    `GET /api/node_aliases`, `GET /api/system` (versions, device, VRAM,
    loaded models, data/models/log folders), `POST /api/free` (unload every
    model and the output cache), `POST /api/logs/open` (open the log folder),
    `GET /api/models/{checkpoints|loras|controlnet|upscale_models}`.
  - `POST /api/queue` (`{"prompt", "extra_data", "front"}`; 400 with
    `{"detail": {"message", "node_id", "class_type", "input"}}`),
    `GET /api/queue`, `DELETE /api/queue` (cancel all pending),
    `DELETE /api/queue/{id}`, `POST /api/interrupt` (optional
    `{"prompt_id"}`; 409 when that prompt isn't the one running),
    `GET /api/history?limit=&before=`, `GET|DELETE /api/history/{id}`,
    `DELETE /api/history` (clear finished).
  - `POST /api/upload`, `GET /api/uploads`, `GET /uploads/{name}`,
    `GET /outputs/{path}`, `GET /temp/{name}`, `DELETE /api/outputs/{path}?type=`,
    `POST /api/outputs/to-upload`, `POST /api/contact_sheet` (the X/Y plot's
    labelled grid of finished images).
  - `GET /api/workflows` (`[{name, modified, size}]`),
    `GET|POST|DELETE /api/workflows/{name}` (`?overwrite=false` → 409 when it
    exists), `POST /api/workflows/{name}/rename`. Names are validated (1-100
    of letters/digits/space/`_-.()`, no trailing dot/space, no Windows device
    names) else 400; UTF-8; a corrupt file is 422.
  - The frontend is served with `Cache-Control: no-cache` (no `?v=N`).
  Host/Origin checks and a request-body cap (5 MB; uploads 21 MB) live in
  `backend/security.py`. Logging (stderr plus a rotating
  `logs/pixelflow.log`, 5 MB × 3, with a torch/CUDA/device line) is set up by
  `backend/logging_setup.py`.
- **`frontend/app.js`** — the client. Builds a LiteGraph node class per
  `/api/nodes` entry; node palette (buttons: click or drag onto the canvas;
  search filter); prompt building through `Logic.buildPrompt()` (muted nodes
  are left out, bypassed ones pass their same-typed input through);
  queueing (`×N`, Ctrl+Enter, Ctrl+Shift+Enter = front; seeds are never sent
  as -1 and follow each sampler's "control after generate"); the queue panel
  (position, timing, per-row Cancel by id, Cancel all, Clear finished); the
  gallery (from history, grouped by prompt, capped at 300) and image viewer
  (←/→, metadata, Load workflow, Reuse seed, Send to Load Image, Delete);
  progress (overall + per-node bar, it/s, ETA, a live latent preview on the
  running node and in the progress area); output images drawn inside
  Save/Preview nodes; the workflow manager (load/rename/duplicate/delete,
  overwrite prompt, unsaved-changes guard, import of `.json` or a PNG with an
  embedded workflow, also by drop); the System panel (GPU/VRAM in the top
  bar, Free memory, Open log folder); groups (Ctrl+G, mute/bypass a whole
  group); Reroute and Primitive nodes and widget-to-input conversion; a mask
  editor on Load Image (paints the mask a Load Mask feeds to the inpaint
  sampler); the Templates menu (`frontend/templates/*.json`); the X/Y plot
  (queues a fixed-seed grid of two widget values and assembles a labelled
  contact sheet); resizable, collapsible side panels (remembered per
  browser); undo/redo covering nodes, links, moves
  and values (capped at 100, never hijacking Ctrl+Z in text fields); a `?`
  shortcut overlay; accessible dialogs (role, focus trap, focus return).
  Errors are drawn on the failing node; the running-node outline and
  mute/bypass shading are drawn, never written to `node.color`.
- **`frontend/logic.js`** — the DOM-free helpers `app.js` uses, unit-tested
  with `npm test` / `node --test tests/js` (also run from pytest when Node
  is installed); `npm run lint` (ESLint) and `npm run e2e` (Playwright smoke
  test against `PIXELFLOW_TEST_MODE=1`) — see `docs/BUILD.md`.
- **`frontend/vendor/`** — litegraph.js and its CSS, vendored locally.
- **`desktop.py`** — see Entry points.

## Nodes included

- **loaders**: Load Checkpoint (`ckpt_name` combo of installed models plus
  "Other…" for any Hub id; `dtype`), Load LoRA, Load Image (IMAGE + alpha
  MASK, with an in-app mask editor), Load Mask, Load ControlNet
  (`models/controlnet/` or a Hub id), Load Upscale Model
  (`models/upscale_models/`, any architecture spandrel reads)
- **controlnet**: Apply ControlNet (strength, start/end; chainable) on all
  three samplers, through diffusers' ControlNet pipelines derived from the
  loaded one with `from_pipe`
- **conditioning**: CLIP Text Encode (`(word:1.2)` weighting, long prompts,
  `clip_skip`)
- **sampling**: KSampler (txt2img), KSampler (Image) (img2img), KSampler
  (Inpaint) — all with `sampler`, `scheduler` (normal/karras/trailing), `seed`,
  `steps`, `cfg`, `batch_size` and an optional `negative`
- **image** (Pillow/NumPy): Empty Image, Resize, Blur, Brightness/Contrast,
  Hue/Saturation, Blend, Text Overlay, Canny Edges (on cv2's threshold
  scale, for ControlNet); Upscale Image (model) runs a spandrel model tiled
- **utility** (frontend only): Reroute, Primitive
- **output**: Preview Image (to `temp/`), Save Image (`prefix_00001_.png`,
  subfolders, `%date%`/`%time%`/`%seed%`/`%model%`)

Generation details (`backend/nodes/generation.py`):

- Every run builds a fresh scheduler from the checkpoint's *own* scheduler
  config, so a karras/trailing choice never leaks into later runs on the
  shared cached pipeline.
- Image *i* of a batch is sampled from `seed + i` (one generator per image)
  and carries that seed; Save/Preview embed a per-image prompt (`seed + i`,
  `batch_size` 1), so any single image can be reproduced. Outputs report
  `{"seed": N, "seeds": [...]}`.
- Prompts (`prompt_weights.py`): the sampler encodes the prompt itself
  when it has weights, is too long for the plain path or uses `clip_skip`,
  for every supported family (A1111-style weighting with mean restoration,
  per text encoder). SD 1.x/2.x: 75-token CLIP chunks. SDXL: CLIP-L and
  CLIP-G chunked alike (penultimate layer, concatenated per token), pooled
  from CLIP-G's first chunk, an unconnected negative zeroed when the
  checkpoint asks for it. SD3: chunked CLIP-L/CLIP-G zero-padded to T5's
  width, followed by T5 (256 tokens, up to 512 for a longer prompt). Flux:
  weighted T5 (512 tokens), CLIP-L pooled from the first 75 tokens; no clip
  skip, no negative. The results equal each pipeline's own `encode_prompt()`
  for a plain short prompt (unit-tested against tiny random encoders), and
  plain short prompts still go to diffusers as text. Unknown architectures
  get the plain text with the weighting syntax stripped.
- LoRAs: a MODEL carries `loras: [(name, strength)]`; the sampler loads each
  file once per pipeline as a named adapter and activates exactly this run's
  set (`set_adapters`), or disables them for a graph without LoRAs. Needs
  `peft`.
- Call arguments a pipeline class doesn't take (e.g. a negative prompt on
  some Flux versions) are dropped instead of failing.
- VAE slicing is always on; VAE tiling above 1536×1536, with one tiled retry
  after an out-of-memory error.

## Verification

Run on 2026-10-07 in the dev `.venv` (Python 3.14, CUDA torch, RTX 5080):

| Command | Result |
|---|---|
| `.\.venv\Scripts\python.exe -m pytest -q` | 365 passed, 1 skipped (the opt-in `PIXELFLOW_GPU_SMOKE` sd-turbo test) |
| `npm test` | 25/25 node:test unit tests pass |
| `npm run lint` | ESLint clean |
| `npm run e2e` | 7 Playwright tests pass (the smoke test plus templates, groups, side panels, reroute/primitive, mask editor and a real 2×2 X/Y plot) |

ControlNet was tested end to end with the tiny test checkpoint and a ControlNet
built from its UNet (no real SD ControlNet was downloaded). The real
RealESRGAN_x4plus upscales 512 → 2048 px in 0.38 s on the RTX 5080.

`tests/test_frontend_logic.py` also runs the node:test suite, `node --check`
and a check that every element `app.js` looks up exists in `index.html` when
Node is installed. CI (`.github/workflows/ci.yml`): a `test` job on Ubuntu /
Python 3.12 and Windows / Python 3.14 with CPU torch (`PIXELFLOW_REQUIRE_HUB=1`,
`-rs`), and a `frontend` job (Ubuntu, Node 22) running ESLint, the unit tests
and the Playwright smoke test. `release.yml` calls `ci.yml` first.

## Known limitations

- **SDXL/SD3/Flux are detected and loaded, but only unit-tested.** The real
  multi-GB weights haven't been run on this machine. Their prompt
  encoding (weighting, long prompts, clip skip) matches diffusers' own for
  plain prompts with tiny test encoders, but how weighted/long prompts look
  with real SDXL/SD3/Flux checkpoints still needs a manual check. Encoded
  prompts aren't cached between runs. Flux/SD3 latents get no live preview.
- **The output cache is in memory only** and holds the last run's node
  outputs (images); it's lost on restart. Nodes must not mutate their
  inputs in place.
- **Single global queue.** Every connected window sees every prompt.
- **No authentication.** Loopback only, with Host/Origin checks.
- **ControlNet is checked only with tiny test models** — a real SD 1.5/SDXL
  ControlNet checkpoint hasn't been run yet. Only Canny is offered as a
  preprocessor (no depth/pose/lineart).
- **Frontend tests**: unit tests of `logic.js`, ESLint, and one Playwright
  smoke test (page load, node palette, adding and wiring nodes with the
  mouse, running a torch-free graph, saving and loading a workflow) against
  the backend in test mode. Generation itself, the image viewer, uploads and
  drag-and-drop of files still need a manual pass in a real window.
- **`torch.compile` is opt-in and not bundled** (`{"torch_compile": true |
  "max-autotune"}` in settings.json or `PIXELFLOW_TORCH_COMPILE`; needs
  `requirements-compile.txt`'s `triton-windows`, not in the packaged app).
  Measured with the tiny checkpoint: ~29 s first compile (128 s
  max-autotune), then ~40 % faster than eager.
- **Releases aren't code-signed with a trusted certificate yet.**
  `build_release.ps1 -Sign` works (checked with a self-signed test
  certificate on 2026-10-07); a real certificate is what removes SmartScreen's
  warning.

## Desktop packaging

See [BUILD.md](BUILD.md).
