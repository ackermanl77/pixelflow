# Adding a node

A node is a Python class in `backend/nodes/`. Nothing in the frontend needs
to change to add one — `frontend/app.js` builds its UI from
`GET /api/nodes` at page load, which reflects `NODE_REGISTRY` automatically.

## Minimal example

```python
from .base import NodeBase, register_node

@register_node("Invert Colors")
class ImageInvert(NodeBase):
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def run(self, image):
        from PIL import ImageOps
        return (ImageOps.invert(image.convert("RGB")),)
```

Then import the module (if it's new) in `backend/nodes/__init__.py`:

```python
from . import io_nodes, image_ops, generation, my_new_module  # noqa: F401
```

That's it — restart the server and "Invert Colors" appears under **image**
in the sidebar.

## Custom node plugins (outside the source tree)

The same class works as a plugin: save it as a `.py` file (or a folder with
an `__init__.py`, which may use relative imports for its own helpers) in the
data folder's `custom_nodes/` and restart PixelFlow (`backend/plugins.py`).
Import the API from the package rather than relatively:

```python
from backend.nodes import NodeBase, register_node
from backend.nodes.base import as_image_list, map_images  # IMAGE helpers
```

- Plugins load at startup, after the built-in nodes, in name order; files
  and folders starting with `_` or `.` are skipped.
- A plugin's nodes carry `"plugin": "<file or folder name>"` in `/api/nodes`;
  `/api/system`'s `custom_nodes` lists every plugin with its nodes or its
  error, which the System panel shows.
- A plugin that raises while importing is logged and reported, never fatal.
  A class name that already exists (a built-in node, or another plugin's) is
  refused and the existing node restored, so a plugin can't change what a
  saved workflow does.
- **Plugins are arbitrary code**: they run with the user's permissions and
  can do anything Python can. The packaged app only bundles its own
  dependencies (torch, diffusers, PIL, numpy, ...), so a plugin can't
  `pip install` more.
- The output cache keys on node signatures, not on code: after editing a
  plugin, use **Free memory** so a stale output isn't reused.

## The contract

- **`CATEGORY`** — groups the node in the sidebar (`loaders`, `image`,
  `conditioning`, `sampling`, `output`, or a new one of your choosing).
- **`INPUT_TYPES()`** — a classmethod returning
  `{"required": {...}, "optional": {...}}`. Each entry is
  `name: (type, config_dict)`:
  - Any type that isn't one of the four primitives below (`"IMAGE"`,
    `"MODEL"`, `"CONDITIONING"`, `"MASK"`, or a new one you invent) →
    rendered as a connectable input socket. The frontend applies the same
    rule as the backend (`frontend/logic.js`'s `widgetKind()`), so a new
    socket type needs no frontend change.
  - `"INT"` / `"FLOAT"` → a number widget; `config` supports `default`,
    `min`, `max`, `step` (the real per-click step; default 1 / 0.1) and,
    for FLOAT, `round` (snap typed/dragged values to this increment). INT
    values are always rounded to whole numbers in the UI.
  - `"BOOLEAN"` → a toggle; `config` supports `default`.
  - `"STRING"` → a text widget; `config` supports `default`,
    `multiline: True`.
  - Any input's `config` may carry a `"tooltip"` (shown in the API for the
    UI). On an INT seed, `"control_after_generate": True` adds the
    fixed/increment/randomize widget (see the samplers).
  - A list of strings, e.g. `["a", "b", "c"]` → a dropdown combo widget.
    `INPUT_TYPES()` is re-read on every `/api/nodes` call and on
    validation, so the list can be dynamic. Add `{"image_upload": True}`
    to the config to get an "Upload image…" button and a thumbnail (see
    `LoadImage`: its list is the files in `uploads/`). Add
    `{"allow_custom": True}` to also accept values outside the list (Load
    Checkpoint: any Hugging Face Hub id; the UI adds an "Other…" entry).
  - `MASK` values are single-channel (`"L"`) PIL images, white = the
    region to repaint.
- **`RETURN_TYPES`** / **`RETURN_NAMES`** — tuples describing each output
  socket's type and label.
- **`FUNCTION`** — the method name the engine calls, given the resolved
  inputs as keyword arguments matching `INPUT_TYPES()`'s names. Return a
  tuple matching `RETURN_TYPES`'s length and order.
- **IMAGE values may be a single `PIL.Image` or a list** (KSampler outputs a
  batch list). Don't assume one: wrap your per-image work with
  `map_images(image, fn)` (or `map_image_pairs()` for two IMAGE inputs, or
  `as_image_list()`) from `backend/nodes/base.py`.
- **`"hidden"`** in `INPUT_TYPES()` (e.g. `{"prompt": "PROMPT",
  "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}`) asks the
  engine to inject the executed prompt / extra PNG info (the frontend
  workflow) / the node's own id — see `SaveImage` and the samplers.
- **`DESCRIPTION`** — one line shown as the node's tooltip in the palette.
- **Validation**: `POST /api/queue` checks every graph against
  `INPUT_TYPES()` (required inputs, socket types, `min`/`max`, combo lists)
  before queueing, so declare accurate `min`/`max` values. For checks the
  types can't express, add a classmethod
  `VALIDATE_INPUTS(cls, **widget_values)` returning `True` or an error
  message (string); it gets the literal widget values only. Examples:
  `EmptyImage`/`ImageTextOverlay` check the colour, `LoadImage` that the
  upload exists. Its message is reported like any other validation error
  (node outlined, the input named when the message quotes its value).
- **Renaming a class**: add `{"OldName": "NewName"}` to `NODE_ALIASES` in
  `backend/nodes/base.py`; saved workflows, queued prompts and PNG metadata
  using the old name keep working (`KSamplerStub` → `KSampler`).
- **Don't import torch/diffusers at module level.** The server starts
  without them (they take seconds to import); import them inside your
  `FUNCTION` (see `backend/nodes/runtime.py`).
- **Execution is output-driven.** Only output nodes and the nodes they
  (transitively) depend on are validated and run; a graph with no output
  node is rejected. A node nothing reaches never runs.
- **Outputs are cached between runs.** A node whose class, widget values,
  upstream signatures and optional `IS_CHANGED` result are unchanged since
  the previous run is served from the queue's `OutputCache` instead of
  running (the UI gets an `execution_cached` message). Treat your inputs as
  read-only (return new images, don't draw into the ones you got), since
  they may be a cached object. Two hooks control this:
  - `IS_CHANGED(cls, **widget_values)` (classmethod, optional): its return
    value is folded into the signature. Return e.g. a file's mtime to
    re-run when it changes, or `float("nan")` to always re-run (KSampler
    does this for `seed = -1`). Only literal widget values are passed.
  - `CACHE_OUTPUTS = False`: never keep this node's outputs (for outputs
    that hold resources with their own lifetime, like LoadCheckpoint's
    pipelines).
- **Errors**: anything your function raises is reported as a
  `NodeExecutionError` naming the node, its class, the exception type and a
  trimmed traceback; the frontend outlines that node in red. Raise with a
  clear message rather than returning a sentinel.
- **`OUTPUT_NODE = True`** — for terminal nodes with no outputs of their own
  (like `SaveImage`/`PreviewImage`). They are what drives execution and
  always run (never served from the cache). Return a dict instead of a tuple (any node
  may: a dict's `"result"` key holds its output tuple, the rest is UI output
  recorded in history) — `{"images": [filename, ...], "type": "output" | "temp"}`
  is the convention the frontend's gallery panel understands (`type` says
  whether the files are in `outputs/` or the throwaway `temp/`; filenames
  may contain `/` subfolders).

## Real generation nodes

`LoadCheckpoint` (`backend/nodes/io_nodes.py`), `CLIPTextEncode`, and
`KSampler` (`backend/nodes/generation.py`) run real Stable Diffusion
inference via `diffusers`/`torch` — see [PROJECT_STATUS.md](PROJECT_STATUS.md#nodes-included)
for the checkpoint-resolution rules, caching, and current limitations.

A few implementation notes for anyone touching these nodes:

- **Pipelines are cached in-process.** `models.PIPELINES.get(ckpt_name,
  dtype)` keys a VRAM-budgeted cache by `(ckpt_name, device, dtype)`: GPU
  while it fits (free VRAM minus a reserve), a CPU-RAM second tier, model
  CPU offload for a model too big on its own, at most
  `MAX_CACHED_PIPELINES` (4) entries. Constructing a pipeline is expensive
  (seconds, multiple GB of weights); don't bypass this cache.
- **`ckpt_name` is untrusted input** (it comes from the graph payload posted
  to `/api/queue`), so local-path resolution goes through `safe_path()`
  (`backend/paths.py`) rather than a raw `Path` join, to stop it from
  reading files outside `models/checkpoints/`. A repo id containing `/`
  (e.g. `stabilityai/sd-turbo`) still resolves to a non-existent local path
  and is treated as a Hugging Face Hub id: `models.ensure_hub_snapshot()`
  uses the cached snapshot or downloads the pipeline's files with progress.
  The architecture decides the pipeline class (`models.py`).
- **`KSampler` calls the full pipeline**, not a separate encode/sample split.
  `CLIPTextEncode` returns `{"text", "clip_skip"}`; the sampler passes the
  text to the pipeline, or — for weighted, long or clip-skipped prompts —
  encodes it itself (`prompt_weights.encode_conditioning()`, one encoder per
  family: SD 1.x/2.x, SDXL, SD3, Flux) and passes `prompt_embeds` (plus
  `pooled_prompt_embeds` and the negatives where the family has them). A new
  sampler should build its prompt arguments with
  `generation._prompt_kwargs()` rather than passing the text itself.
- **GPU is used automatically** when `torch.cuda.is_available()`; the
  precision comes from Load Checkpoint's `dtype` (`models.resolve_dtype_name()`).
- **img2img / inpainting** (`KSampler (Image)`, `KSampler (Inpaint)`) get
  their pipelines from `generation.derived_pipeline(pipe, "img2img" |
  "inpaint")`, which calls `AutoPipelineFor*.from_pipe()` once per loaded
  checkpoint (shared weights, no extra VRAM) and keeps the result in a
  `WeakKeyDictionary` keyed by the base pipeline, so it's freed together
  with it. All three samplers go through `_run_diffusion()` (a fresh
  scheduler from the checkpoint's own config, LoRAs, per-image seeds,
  progress and live previews, VAE tiling, out-of-memory handling).
- **VAE memory**: VAE slicing is always enabled on load; VAE tiling is
  switched on per run above `VAE_TILING_MIN_PIXELS` (1536²), and a run that
  runs out of memory untiled is retried once with tiling before the clear
  "Out of GPU memory" error.

Nothing in `backend/engine/`, `backend/main.py`, or `frontend/app.js` needs
to change for a new node — they only care about the `INPUT_TYPES`/`RETURN_TYPES`
contract, never what happens inside a node's function body. This is the same
separation ComfyUI itself relies on for its custom-node ecosystem.
