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

## Wiring up real generation (GPU)

`backend/nodes/generation.py` (`CLIPTextEncode`, `KSamplerStub`) and
`LoadCheckpoint` in `backend/nodes/io_nodes.py` are deterministic
placeholders — no GPU is available on this dev machine, so they produce a
gradient+label image from the prompt/seed instead of running a real model.
Each one's docstring says exactly what to change. In short, once a GPU is
available:

```powershell
pip install torch diffusers --index-url <the right CUDA wheel index for your GPU>
```

```python
# backend/nodes/io_nodes.py — LoadCheckpoint.load()
def load(self, ckpt_name):
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(ckpt_name).to("cuda")
    return ({"pipe": pipe},)
```

```python
# backend/nodes/generation.py — KSamplerStub.sample()
def sample(self, model, positive, latent_image, seed, steps, cfg):
    import torch
    image = model["pipe"](
        positive["text"],
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=torch.manual_seed(seed),
    ).images[0]
    return (image,)
```

Nothing in `backend/engine/`, `backend/main.py`, or `frontend/app.js` needs
to change — they only care about the `INPUT_TYPES`/`RETURN_TYPES` contract,
never what happens inside a node's function body. This is the same
separation ComfyUI itself relies on for its custom-node ecosystem.
