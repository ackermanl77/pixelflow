# PixelFlow

A small ComfyUI-style node-graph app: a Python backend that runs a graph of
image-processing/generation nodes, and a web canvas UI (built on
[litegraph.js](https://github.com/jagenjo/litegraph.js), the same library
ComfyUI's own frontend uses) for wiring nodes together and running them.

## Current Status

- `docs/PROJECT_STATUS.md` is the authoritative reference for architecture,
  entry points, and known limitations — this README only covers day-to-day
  commands.
- `docs/NODE_AUTHORING.md` covers adding a new node type.
- `docs/BUILD.md` covers building the packaged desktop `.exe`.
- No GPU on this dev machine, so the generation nodes (checkpoint loading,
  CLIP text encode, sampling) are deterministic CPU-only stubs — see
  `docs/PROJECT_STATUS.md` for what that means and `docs/NODE_AUTHORING.md`
  for how to swap in real `diffusers`-based inference later.

## Install

```powershell
pip install -r requirements.txt
```

## Run (dev web app)

```powershell
python run.py
```

Then open http://127.0.0.1:8188 — click **Load Example** for a starter
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

- **loaders**: Load Checkpoint (stub), Load Image
- **conditioning**: CLIP Text Encode (stub)
- **sampling**: KSampler (stub) — deterministic placeholder image generator
- **image**: Empty Image, Resize, Blur, Brightness/Contrast, Hue/Saturation,
  Blend, Text Overlay — real, CPU-only, Pillow-based operations
- **output**: Preview Image, Save Image

Adding a new node is a single Python class — see `docs/NODE_AUTHORING.md`.

## Tests

```powershell
pip install pytest
python -m pytest tests/
```
