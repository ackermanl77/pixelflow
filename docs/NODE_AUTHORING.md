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

## The contract

- **`CATEGORY`** — groups the node in the sidebar (`loaders`, `image`,
  `conditioning`, `sampling`, `output`, or a new one of your choosing).
- **`INPUT_TYPES()`** — a classmethod returning
  `{"required": {...}, "optional": {...}}`. Each entry is
  `name: (type, config_dict)`:
  - `"IMAGE"`, `"MODEL"`, `"CONDITIONING"` → rendered as a connectable input
    socket (see `LINK_TYPES` in `frontend/app.js` — extend that set if you
    introduce a new socket type).
  - `"INT"` / `"FLOAT"` → a number widget; `config` supports `default`,
    `min`, `max`.
  - `"STRING"` → a text widget; `config` supports `default`,
    `multiline: True`.
  - A list of strings, e.g. `["a", "b", "c"]` → a dropdown combo widget.
- **`RETURN_TYPES`** / **`RETURN_NAMES`** — tuples describing each output
  socket's type and label.
- **`FUNCTION`** — the method name the engine calls, given the resolved
  inputs as keyword arguments matching `INPUT_TYPES()`'s names. Return a
  tuple matching `RETURN_TYPES`'s length and order.
- **`OUTPUT_NODE = True`** — for terminal nodes with no outputs of their own
  (like `SaveImage`/`PreviewImage`) that should always run even if nothing
  downstream consumes them. Return a dict instead of a tuple —
  `{"images": [filename, ...]}` is the convention the frontend's gallery
  panel already understands (see `frontend/app.js`'s `execution_success`
  handler).

## Real generation nodes

`LoadCheckpoint` (`backend/nodes/io_nodes.py`), `CLIPTextEncode`, and
`KSampler` (`backend/nodes/generation.py`) run real Stable Diffusion
inference via `diffusers`/`torch` — see [PROJECT_STATUS.md](PROJECT_STATUS.md#nodes-included)
for the checkpoint-resolution rules, caching, and current limitations
(CPU speed, txt2img-only).

A few implementation notes for anyone touching these nodes:

- **Pipelines are cached in-process.** `io_nodes._load_pipeline()` keys a
  module-level cache by `(ckpt_name, device)` under a lock, because
  constructing a `StableDiffusionPipeline` is expensive (seconds, multiple
  GB of weights). Don't remove this cache without replacing it with
  something equivalent — reloading per node execution would make every
  generation take minutes even on a GPU.
- **`ckpt_name` is untrusted input** (it comes from the graph payload posted
  to `/api/queue`), so local-path resolution goes through `safe_path()`
  (`backend/paths.py`) rather than a raw `Path` join, to stop it from
  reading files outside `models/checkpoints/`. A repo id containing `/`
  (e.g. `stabilityai/sd-turbo`) still resolves to a non-existent local path
  and falls through to `StableDiffusionPipeline.from_pretrained(ckpt_name)`,
  which treats it as a Hugging Face Hub id instead.
- **`KSampler` calls the full pipeline**, not a separate encode/sample split.
  `CLIPTextEncode` only wraps the prompt as `{"text": ...}`; the pipeline's
  own `prompt=` argument does its own tokenization and CLIP encoding. This
  keeps the graph's ComfyUI-shaped node layout without reimplementing
  `diffusers`' internal encode/denoise loop.
- **GPU is used automatically** when `torch.cuda.is_available()`, with
  `float16` on GPU and `float32` on CPU. There's no user-facing device
  toggle yet.

Nothing in `backend/engine/`, `backend/main.py`, or `frontend/app.js` needs
to change for this — they only care about the `INPUT_TYPES`/`RETURN_TYPES`
contract, never what happens inside a node's function body. This is the same
separation ComfyUI itself relies on for its custom-node ecosystem.
