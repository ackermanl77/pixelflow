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
  `docs/PROJECT_STATUS.md` for details and current limitations (CPU speed,
  first-run download, txt2img-only).

## Install

```powershell
pip install -r requirements.txt
```

This pulls in `torch`, which is a large download (the CPU wheel is a few
hundred MB). `requirements.txt` installs the default CPU build; if you have
an NVIDIA GPU and want CUDA acceleration, install a CUDA-enabled `torch`
build first (see https://pytorch.org/get-started/locally/ for the right
`--index-url` for your CUDA version) before installing the rest.

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

The release script installs dependencies, runs PyInstaller, and bundles the
Microsoft Edge WebView2 Runtime as a sibling `WebView2Runtime\` folder next to
`PixelFlow.exe` so the packaged app doesn't depend on the target machine
having WebView2 installed — see `docs/BUILD.md` for the full walkthrough,
including how it reuses the sibling `photo_professional_tool` project's
already-downloaded copy of that runtime instead of re-fetching ~300MB.

Generated images, uploads, and saved workflows live in `outputs\`,
`uploads\`, and `workflows\` next to wherever the app is running from (the
project root in dev, next to `PixelFlow.exe` when packaged) — see
`docs/PROJECT_STATUS.md`'s note on `backend/paths.py` for why those two
locations differ.

**Or let CI build it for you**: pushing a tag like `v0.1.0` triggers
`.github/workflows/release.yml`, which runs this exact same script on a
`windows-latest` runner and attaches `PixelFlow-<version>-Windows.7z` + its
`.sha256` to a new GitHub Release, plus keeps them as a build artifact on the
workflow run either way. It can also be run on demand (Actions tab →
"Release Build" → "Run workflow", enter a version) without needing a tag
push or a real release at all, which just builds and uploads the artifact.

## Nodes included

- **loaders**: Load Checkpoint (real `diffusers` Stable Diffusion loading),
  Load Image
- **conditioning**: CLIP Text Encode
- **sampling**: KSampler — real diffusion sampling (GPU if available, else CPU)
- **image**: Empty Image, Resize, Blur, Brightness/Contrast, Hue/Saturation,
  Blend, Text Overlay — real, CPU-only, Pillow-based operations
- **output**: Preview Image, Save Image

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
skip themselves if the Hugging Face Hub is unreachable.
`.github/workflows/ci.yml` runs the suite on every push/PR with CPU `torch`.

## Security

The server binds to `127.0.0.1` and additionally rejects requests whose
`Host` isn't `127.0.0.1`/`localhost` (DNS rebinding) and state-changing
requests / WebSocket connections whose `Origin` isn't the app's own
(`backend/security.py`), so a web page open in your browser can't drive it.
