# PixelFlow

A small ComfyUI-style node-graph app: a Python backend that runs a graph of
image-processing/generation nodes, and a web canvas UI (built on
[litegraph.js](https://github.com/jagenjo/litegraph.js), the same library
ComfyUI's own frontend uses) for wiring nodes together and running them.

## Current Status

- `docs/USER_GUIDE.md` walks through the app and every feature in it — start
  there if you just want to use PixelFlow.
- `docs/PROJECT_STATUS.md` is the authoritative reference for architecture,
  entry points, and known limitations — this README only covers day-to-day
  commands.
- `docs/NODE_AUTHORING.md` covers adding a new node type.
- `docs/BUILD.md` covers building the packaged desktop `.exe`.
- The generation nodes (checkpoint loading, CLIP text encode, sampling) run
  real Stable Diffusion inference via `diffusers`/`torch`. Uses a CUDA GPU
  automatically if available; otherwise falls back to CPU, which works but
  is slow for anything beyond a distilled few-step model — see
  `docs/PROJECT_STATUS.md` for details and current limitations. SD 1.x/2.x,
  SDXL, SD3 and Flux checkpoints are detected automatically; txt2img,
  img2img, inpainting, LoRAs, prompt weighting and long prompts (all four
  families), clip skip and karras/trailing schedules are supported, with
  live previews while sampling.
- Only the nodes an output (Save/Preview Image) needs are run, and unchanged
  nodes are reused from the previous run — re-queueing after a Blur tweak
  doesn't re-sample a fixed-seed image.
- The graph auto-saves in the browser; the queue, gallery and history
  survive reloads and restarts (`history.jsonl`). Logs go to
  `logs\pixelflow.log` (System panel → Open log folder).
- The server starts in well under a second: torch and diffusers are
  imported in the background.

## Install

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu130   # NVIDIA GPU (RTX 50-series needs cu128+)
pip install -r requirements.txt
```

`requirements.txt` holds version ranges; `requirements-lock.txt` has the
exact tested versions (`pip install -r requirements.txt -c
requirements-lock.txt`). On Windows, PyPI's own `torch` wheel is CPU-only,
so install the CUDA build first as above (or skip that line for CPU).

## Run (dev web app)

```powershell
python run.py
```

Then open http://127.0.0.1:8288 — click **Load Example** for a starter
checkpoint → prompt → sampler → save graph, then **Queue Prompt**.

## Desktop App

```powershell
pip install -r requirements-desktop.txt
python desktop.py
```

Opens the same app in a native window instead of a browser tab.

**The built app always lands in `release\PixelFlow.exe`** — a fixed-name
folder `scripts\build_release.ps1` overwrites with the latest build on every
run.

```powershell
.\scripts\build_release.ps1 -Version "0.1.0"
```

The release script installs a **CUDA** build of torch (from the PyTorch
cu130 index), refuses to build if the interpreter's torch has no CUDA
(`torch.version.cuda`; pass `-AllowCpu` to override), writes
`release\build-info.json`, runs PyInstaller, and bundles the
Microsoft Edge WebView2 Runtime as a sibling `WebView2Runtime\` folder next to
`PixelFlow.exe` so the packaged app doesn't depend on the target machine
having WebView2 installed — see `docs/BUILD.md` for the full walkthrough,
including how it reuses the sibling `photo_professional_tool` project's
already-downloaded copy of that runtime instead of re-fetching ~300MB.

Generated images, uploads, saved workflows, models and logs live in the
project folder when running from source, and in `%LOCALAPPDATA%\PixelFlow`
for the packaged app (next to `PixelFlow.exe` with a `portable.txt` file
there). `PIXELFLOW_HOME` and `PIXELFLOW_MODELS_DIR` override them — see
`docs/PROJECT_STATUS.md`.

**Or let CI build it for you**: pushing a tag like `v0.1.0` triggers
`.github/workflows/release.yml`, which runs this exact same script on a
`windows-latest` runner after the full test suite passes, and attaches
`PixelFlow-<version>-Windows.7z` (split into `.7z.001`, `.002`, … when it's
over GitHub's 2 GiB asset limit) + its `.sha256` to a new GitHub Release, plus keeps them as a build artifact on the
workflow run either way. It can also be run on demand (Actions tab →
"Release Build" → "Run workflow", enter a version) without needing a tag
push or a real release at all, which just builds and uploads the artifact.

## Nodes included

- **loaders**: Load Checkpoint (installed-model dropdown or any Hub id;
  SD 1.x/2.x, SDXL, SD3, Flux; precision option), Load LoRA, Load Image
  (upload button, drag-and-drop or paste; outputs the image and its alpha
  as a MASK), Load Mask
- **conditioning**: CLIP Text Encode (`(word:1.2)` weighting, long prompts,
  clip skip)
- **sampling**: KSampler (txt2img), KSampler (Image) (img2img / hires fix),
  KSampler (Inpaint) — real diffusion sampling (GPU if available, else CPU),
  per-image seeds in a batch, normal/karras/trailing schedules
- **image**: Empty Image, Resize, Blur, Brightness/Contrast, Hue/Saturation,
  Blend, Text Overlay — real, CPU-only, Pillow-based operations
- **output**: Preview Image (throwaway, `temp\`), Save Image (numbered
  `prefix_00001_.png`, subfolders, `%date%`/`%seed%` tokens)

Adding a new node is a single Python class — see `docs/NODE_AUTHORING.md`.

## Tests

```powershell
pip install pytest httpx
python -m pytest -q
```

`pytest.ini` limits collection to `tests/` (a bare `pytest` otherwise walks
into `release/_internal/` and the venv). Tests write only to per-test temp
folders, never the real `outputs/`/`uploads/`. The real-pipeline tests use
the tiny `hf-internal-testing/tiny-stable-diffusion-torch` checkpoint and
skip themselves if the Hugging Face Hub is unreachable (unless
`PIXELFLOW_REQUIRE_HUB=1`, which CI sets). Tests marked `gpu` need CUDA
(`python -m pytest -q -m gpu`; see `docs/BUILD.md`).
`.github/workflows/ci.yml` runs the suite on every push/PR on Linux/Python
3.12 and Windows/Python 3.14 with CPU `torch`, listing skips (`-rs`).
`tests/test_frontend_logic.py` also runs the frontend's unit tests
(`node --test tests/js/*.test.js`), `node --check`, and a check that every
element `app.js` looks up exists in `index.html`.

Frontend tooling (Node 20.19+; dev only, nothing of it ships):

```powershell
npm ci                 # ESLint + Playwright, into node_modules/ (gitignored)
npm run lint           # ESLint (flat config: eslint.config.js)
npm test               # the node:test unit tests
npm run e2e:install    # once: Chromium for Playwright, under node_modules/
npm run e2e            # browser smoke test against the backend in test mode
```

`npm run e2e` starts the backend with `PIXELFLOW_TEST_MODE=1` (no torch, no
model loading, a throwaway data folder in the temp directory) through
`tests/e2e/server.py`; it uses `.venv`'s Python, or `PIXELFLOW_PYTHON`. CI
runs all three in a separate `frontend` job.

## Security

The server binds to `127.0.0.1` and additionally rejects requests whose
`Host` isn't `127.0.0.1`/`localhost` (DNS rebinding) and state-changing
requests / WebSocket connections whose `Origin` isn't the app's own
(`backend/security.py`), so a web page open in your browser can't drive it.
