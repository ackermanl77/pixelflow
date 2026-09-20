# PixelFlow

A small ComfyUI-style node-graph app: a Python backend that runs a graph of
image-processing/generation nodes, and a web canvas UI (built on
[litegraph.js](https://github.com/jagenjo/litegraph.js), the same library
ComfyUI's own frontend uses) for wiring nodes together and running them.

## Architecture

This mirrors ComfyUI's actual design at small scale:

- **Nodes** (`backend/nodes/`) declare `INPUT_TYPES`, `RETURN_TYPES`, and a
  `FUNCTION` to call — the same contract ComfyUI custom nodes use. Nodes
  register themselves into `NODE_REGISTRY` via the `@register_node` decorator.
- **`GET /api/nodes`** exposes that registry as JSON. The frontend fetches it
  on load and dynamically builds a LiteGraph node class per entry — nodes are
  never hard-coded in the frontend, so adding a Python node makes it show up
  in the UI automatically.
- **Graph execution** (`backend/engine/graph.py`) takes a flattened "prompt"
  (`{node_id: {class_type, inputs}}`, ComfyUI's own execution format) and
  resolves it recursively with per-node output caching — no separate topo-sort
  pass needed.
- **Queue** (`backend/engine/queue.py`) is a single-worker async job queue.
  Submitted prompts run one at a time in a background thread; progress
  (`queued` / `executing` / `execution_success` / `execution_error`) streams
  to the browser over a WebSocket (`/ws`), same event-shape idea as ComfyUI.
- **Frontend** (`frontend/app.js`) is one file: it builds node classes from
  `/api/nodes`, renders a categorized node-library sidebar, serializes the
  canvas into a prompt on "Queue Prompt", and renders queue status + an output
  gallery from the WebSocket stream. Undo/redo (Ctrl+Z / Ctrl+Y) snapshots
  `graph.serialize()`. Workflows save/load as JSON, either to the backend
  (`workflows/`, via `/api/workflows`) or as a downloaded/imported file.

## Nodes included

- **loaders**: Load Checkpoint (stub), Load Image
- **conditioning**: CLIP Text Encode (stub)
- **sampling**: KSampler (stub) — deterministic placeholder image generator
- **image**: Empty Image, Resize, Blur, Brightness/Contrast, Hue/Saturation,
  Blend, Text Overlay — these are real, CPU-only, Pillow-based operations
- **output**: Preview Image, Save Image

There's no GPU here, so the "generation" side (Load Checkpoint, CLIP Text
Encode, KSampler) is intentionally stubbed: it produces a deterministic
gradient+label image from the prompt/seed instead of running a real diffusion
model, so the graph shape and execution flow are real and testable end to end
without needing `torch`/`diffusers`/a GPU.

### Plugging in real generation later

Each stub node's docstring in `backend/nodes/generation.py` and
`backend/nodes/io_nodes.py` says exactly what to replace. In short, once a GPU
is available: install `torch` + `diffusers`, load a real pipeline in
`LoadCheckpoint.load`, and replace `KSamplerStub.sample`'s body with a call to
`StableDiffusionPipeline(...)`. Nothing else in the engine, queue, or frontend
needs to change — they only care about the `INPUT_TYPES`/`RETURN_TYPES`
contract, not what's inside `sample()`.

## Running it

```powershell
pip install -r requirements.txt
python run.py
```

Then open http://127.0.0.1:8188 — click **Load Example** for a starter
checkpoint → prompt → sampler → save graph, then **Queue Prompt**.

## Tests

```powershell
pip install pytest
python -m pytest tests/
```
