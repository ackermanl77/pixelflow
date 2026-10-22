# PixelFlow — User Guide

A walkthrough of the app and every feature in it, for someone using PixelFlow
rather than developing it. If you're looking for architecture or how to add
a node, see [PROJECT_STATUS.md](PROJECT_STATUS.md) and
[NODE_AUTHORING.md](NODE_AUTHORING.md) instead.

## What PixelFlow is

PixelFlow lets you build an image-processing pipeline visually: drag nodes
onto a canvas, wire their inputs and outputs together, then run the whole
graph with one click. It's modeled on ComfyUI's node-graph approach, at a
much smaller scale. Its "AI generation" nodes run real Stable Diffusion
inference (see [Generating images](#generating-images) below) — GPU if you
have one, CPU otherwise (slower, but it works).

## Starting the app

- **Dev web app**: `python run.py`, then open http://127.0.0.1:8288 in a
  browser.
- **Desktop app**: double-click `release\PixelFlow.exe` (or run
  `python desktop.py` from source) — same app, opened in its own window
  instead of a browser tab. If port 8288 is taken by another program, the
  desktop app picks a free port; if PixelFlow is already running, a second
  launch brings its window to the front (restoring it if minimized) instead
  of starting again -- or, when that one has no window (started for a
  browser), opens a window on it.

Either way, you get the same interface described below.

## The interface

```
┌──────────────────────────────────── Topbar ─────────────────────────────────────┐
│ PixelFlow ● [Queue Prompt]×[1] │ [Workflows…][Save][Templates…][Load Example]   │
│            [Export JSON][Import…] │ [Undo][Redo][Fit View][?]   GPU · VRAM used  │
├───────────┬───────────────────────────────────────┬─────────────────────────────┤
│  Node     │                                       │  Queue  [Cancel all][Clear] │
│  Library  │            Canvas                     │  Output Gallery             │
│ (sidebar) │       (your node graph)               │  (right panel)              │
└───────────┴───────────────────────────────────────┴─────────────────────────────┘
```

- **Node Library** (left) — every available node, grouped by category, with
  a search box. Click one to drop it in the middle of the view, or drag it
  onto the canvas where you want it. Hover for a one-line description.
- **Canvas** (center) — where you build your graph.
- **Queue** (top right) — your runs (restored after a reload or restart):
  queued (with its position), running (with the node and elapsed time),
  done (with the seed and how long it took), interrupted or errored.
- **Output Gallery** (bottom right) — every image your runs produced, newest
  first, grouped by run. Dashed borders mark previews (not saved).
- **System** (top-right corner) — your GPU and its memory use; click it for
  details.

The bars between the side panels and the canvas resize them: drag one, or
click it and use <kbd>←</kbd>/<kbd>→</kbd>. Its small arrow button (or a
double-click, or <kbd>Enter</kbd>) collapses that panel to give the canvas the
whole width, and brings it back. The layout is remembered in this browser.

## Building a graph

1. **Add a node** — click or drag a node from the sidebar, or right-click
   empty canvas space for litegraph's own "Add Node" menu.
2. **Move it** — drag a node by its title bar.
3. **Set its values** — each node shows its own widgets: numbers, text
   boxes, dropdowns and on/off toggles.
4. **Connect nodes** — inputs like an image, a model or a conditioning show
   as dots on a node's left edge. Drag from an output dot (right edge of the
   upstream node) to an input dot. Wires are coloured by type (image blue,
   model purple, conditioning orange, mask green). Only nodes that lead to a
   Save Image or Preview Image run, so a half-built side branch doesn't
   block a run.
5. **Delete a node** — click it, then press <kbd>Delete</kbd>.
6. **Mute or bypass** — right-click a node (or select nodes and press
   <kbd>Ctrl+M</kbd> / <kbd>Ctrl+B</kbd>). A *muted* node (dark) is left out
   of the run, and so is anything that only it feeds. A *bypassed* node
   (purple) is skipped but its input is passed straight through to whatever
   it was feeding (e.g. bypass a Blur to compare with and without it).
7. **Group nodes** — select several (Ctrl+click, or Ctrl+drag a box) and
   press <kbd>Ctrl+G</kbd>: a titled frame is drawn around them and moves
   them together. Right-click the group's title for **Mute group**, **Bypass
   group**, **Select nodes**, and **Edit Group** (title, colour, remove).
   Groups are saved with the workflow.
8. **Reroute** (utils) — a small pass-through node: wire an output into it
   and its output on to one or more inputs, to tidy long wires.
9. **Share one value** — right-click a node and pick **Convert <widget> to
   input** (e.g. a sampler's `seed`, or a prompt's `text`), then wire a
   **Primitive** (utils) into that new input: its value is used for every
   input it's wired to, so one seed or one size drives several nodes. The
   first wire copies the widget's current value into the Primitive. Convert
   the input back to a widget (once it's unwired) from the same menu.
   Reroute and Primitive only exist in the editor -- they are resolved when
   the prompt is queued, so the server never runs them.
10. **Paint a mask** — right-click a Load Image (with an image chosen) and
   pick **Open in mask editor**: paint over what the inpaint sampler should
   repaint (Erase undoes strokes, <kbd>[</kbd>/<kbd>]</kbd> change the brush,
   Shift for bigger steps). **Save mask** uploads it as `<image>_mask.png`
   into a Load Mask node next to the image (re-opening the editor shows it
   again and updates that node) and wires it into the KSampler (Inpaint) the
   image feeds. The Inpaint template is a good starting point.

## Running a graph

Click **Queue Prompt** (or press <kbd>Ctrl+Enter</kbd>). The number next to
it queues that many runs at once. <kbd>Ctrl+Shift+Enter</kbd> (or
Shift-click) puts a run in front of everything already waiting.

- The **progress area** at the bottom of the canvas shows the overall
  progress (node *n* of *m*), the current node's steps, the speed (it/s) and
  the time left. While a sampler runs you see a rough live preview of the
  image, both there and on the sampler node itself; the running node has a
  blue outline.
- When a run finishes, each Save/Preview Image node shows its newest image
  inside the node, and the images appear in the gallery.
- **Cancel** on a queued row removes it; on the running row it stops that
  run at the next step (it never stops a different run that started in the
  meantime). **Cancel all** cancels everything that hasn't started;
  **Clear finished** removes finished rows (images stay on disk).
- **Several windows.** Each browser tab or window shows only its own runs
  (and their progress and images), though they all share one queue. Tick
  **All windows** above the queue to see every window's runs too (marked
  "another window"); **Cancel all** and **Clear finished** then act on all
  of them.
- Unchanged nodes are reused from the previous run, even after restarting
  PixelFlow: re-queueing a fixed-seed graph after changing only a Blur
  doesn't sample again. **Free memory** forgets them.

If the graph has a problem (an unconnected required input, a wire between
incompatible sockets, an out-of-range number, a colour that isn't one, an
upload that's gone, or no Save/Preview Image node at all), **Queue Prompt**
refuses it straight away: the offending node gets a red outline with the
message under it (a red dot marks the input, when it's a socket), and a
pop-up in the bottom-right repeats it. A node that fails while running is
marked the same way.

### Seeds

Every sampler has a `seed` and a **control after generate** setting:

- **randomize** (default) — a new random seed after each queue;
- **increment** — seed + 1 after each queue;
- **fixed** — keep the seed.

A seed of -1 is replaced by a concrete random seed when you queue (and
shown in the widget), so every run can be reproduced. In a batch, image *i*
uses seed + *i*; the PNG of each image stores its own seed, so you can
re-create any single image of a batch.

## The node library, category by category

### loaders

- **Load Checkpoint** — `ckpt_name` lists the models in
  `models\checkpoints\` (single `.safetensors`/`.ckpt` files and diffusers
  folders) and the diffusers models already in your Hugging Face cache; pick
  **Other (type a Hugging Face Hub id)…** to type any Hub repo id. `dtype`
  sets the precision (see [Generating images](#generating-images)).
  Outputs a `MODEL`.
- **Load LoRA** — adds a LoRA from `models\loras\` to a `MODEL` at a
  `strength` (-5 to 5). Chain several for several LoRAs. Needs the `peft`
  package (included in `requirements.txt`).
- **Load Image** — pick an uploaded image from the `filename` dropdown, or
  click **Upload image…** to add one; a thumbnail shows the chosen file.
  You can also drag an image file onto the canvas, or paste one with
  <kbd>Ctrl+V</kbd>, to upload it and get a Load Image node for it. Outputs
  the `IMAGE` and a `MASK` made from its transparency (transparent areas
  are the ones inpainting repaints). Phone photos are rotated upright.
- **Load Mask** — a grayscale upload as a `MASK` (white = repaint), or its
  alpha channel; `invert` flips it.
- **Load ControlNet** — a ControlNet from `models\controlnet\` (a diffusers
  folder with its `config.json`, or a single `.safetensors` file) or any Hub
  repo id, as a `CONTROL_NET`. It must match the checkpoint's family: for the
  default sd-turbo (SD 2.1), e.g. `thibaud/controlnet-sd21-canny-diffusers`;
  for SD 1.5, `lllyasviel/sd-controlnet-canny`; SDXL ControlNets for SDXL.
- **Apply ControlNet** (conditioning) — attaches a `CONTROL_NET` and its
  control image (e.g. **Canny Edges** of a photo) to the positive
  `CONDITIONING` with a `strength` (0–2) and a `start`/`end` window (share of
  the steps). Chain several for several ControlNets; every sampler (txt2img,
  image and inpaint) follows them. The **ControlNet (Canny)** template wires
  this up.
- **Load Upscale Model** — a super-resolution model from
  `models\upscale_models\` (`.pth`/`.safetensors`: ESRGAN, Real-ESRGAN,
  SwinIR, HAT and others the `spandrel` package recognises, e.g.
  RealESRGAN_x4plus.pth). **Upscale Image (model)** (image) enlarges an
  IMAGE with it (2x/4x as trained), in tiles so big images fit in VRAM —
  e.g. 512 → 2048 px in under half a second on an RTX 5080. Re-sample the
  result at low denoise for a hires fix, or Resize Image to another size.
- **Canny Edges** (image) — white-on-black edges, the control image a Canny
  ControlNet expects; `low`/`high` (default 100/200, OpenCV's scale) keep
  more or fewer edges.
- **Depth Map** (image) — near is white, far is black, for a depth
  ControlNet. Uses Depth Anything V2 Small (about 100 MB, downloaded from
  the Hugging Face Hub the first time, then cached), on the GPU when there
  is one; any transformers depth model id can be typed instead.
- **Pose (OpenPose)** (image) — finds the people in the picture and draws
  their body poses as coloured OpenPose skeletons on black, for an OpenPose
  ControlNet (e.g. `lllyasviel/control_v11p_sd15_openpose`). Two models are
  downloaded from the Hugging Face Hub the first time (a person detector and
  ViTPose, about 250 MB together); `keypoint_threshold` leaves out joints the
  model isn't sure of, `person_threshold` how sure it must be that something
  is a person. Body only -- no hands or face points.
- **Soft Edges** (image) — soft, HED-like contours for a SoftEdge/HED
  ControlNet; higher `detail` keeps finer edges.
- **Lineart** (image) — thin white lines on black for a lineart ControlNet;
  `sigma` sets the line scale, `sharpness` how solid the lines are.
- **Scribble** (image) — thick pure-white strokes from the main contours
  for a scribble ControlNet; `threshold` and `thickness` tune them.
- Soft Edges, Lineart and Scribble are classical filters (no model): close
  enough to guide those ControlNets, though the learned annotators they were
  trained on draw cleaner lines on busy photos. Templates > **ControlNet
  (Depth)** wires Depth Map into an SD 2.1 depth ControlNet.

### conditioning

- **CLIP Text Encode** — a multiline `text` box (your prompt), outputs
  `CONDITIONING`. Weighting works like in A1111/ComfyUI: `(word)` ×1.1,
  `((word))` ×1.21, `[word]` ÷1.1, `(word:1.4)` exactly ×1.4, and `\(` for a
  literal bracket. Long prompts are no longer cut off. `clip_skip` 1 equals
  "clip skip 2" elsewhere. Per model family:
  - **SD 1.x/2.x** — weighting, long prompts (75-token chunks), clip skip.
  - **SDXL** — weighting and long prompts on both text encoders; `clip_skip`
    goes further back than SDXL's default (which already uses the
    second-to-last layer, so 1 here is one layer beyond that). With no
    negative prompt connected, SDXL uses an "empty" (all-zero) negative, as
    diffusers does.
  - **SD3** — weighting and long prompts on CLIP and T5 (T5 reads up to 512
    tokens); clip skip on the CLIP encoders.
  - **Flux** — weighting (on T5, which reads up to 512 tokens). Flux has no
    clip skip and ignores negative prompts.
  A prompt without weights that fits in 75 tokens is passed to the model
  exactly as before.

  **Wildcards** vary a prompt: `a {red|green|blue} car` picks one of the
  alternatives, and `__animal__` picks one line of `wildcards\animal.txt` in
  your data folder (one option per line, `#` for comments; subfolders as
  `__colors/warm__`). Alternatives can nest (`{a|{b|c}}`) or be empty
  (`{very |}`), and a wildcard file's lines can contain more wildcards. They're
  chosen from each image's seed, so a batch of 4 gives 4 different prompts, and
  any one image re-runs to the same prompt from its saved seed. A `{word}`
  without `|` is left as typed.

  **Embeddings** (textual inversion): put `.safetensors`/`.pt`/`.bin` files in
  `models\embeddings\` and write `embedding:name` in a prompt (the extension is
  optional; subfolders as `embedding:styles/name`), e.g. a negative prompt of
  `embedding:bad_hands`. SD 1.x/2.x and SDXL only (an SDXL embedding has
  `clip_l` and `clip_g` parts); using an SD 1.5 embedding with an SDXL model,
  or the other way round, stops with a clear error.

### sampling

- **KSampler** — takes a `MODEL`, `CONDITIONING`, and an `IMAGE` that only
  sets the output size, plus `sampler`, `scheduler` (normal, karras —
  popular for DPM samplers — or trailing), `seed`, `steps`, `cfg` and
  `batch_size`. Wire a second CLIP Text Encode into `negative` for a
  negative prompt.
- **KSampler (Image)** — img2img: like KSampler, but starts from the `image`
  you connect. `denoise` 0–1 sets how much changes (about 0.3–0.6 keeps the
  composition). Chain KSampler → Resize Image → KSampler (Image) at ~0.4
  for a "hires fix".
- **KSampler (Inpaint)** — repaints only the white part of `mask` (e.g. Load
  Image's second output) and keeps the rest of the image.

### image (real, CPU-only operations)

- **Empty Image** — `width`, `height`, and a `color` (any CSS colour: `#202020`,
  `rgb(32,32,32)`, `black`). Creates a solid-colour canvas.
- **Resize Image** — `width`/`height`.
- **Blur** — `radius`. Gaussian blur.
- **Brightness / Contrast** — two sliders, 0–3× each (1.0 = unchanged).
- **Hue / Saturation** — `hue_shift_degrees` (-180 to 180) and `saturation`
  (0–3×).
- **Blend** — a `base` and an `overlay` image, a `mode` (normal / multiply /
  screen), and an `opacity` slider.
- **Text Overlay** — draws `text` at `x`/`y` with a `size` and `color`.

**Remove Background** keeps the subject -- people, animals, objects (COCO
panoptic Mask2Former, downloaded once, ~190 MB) -- and makes the rest
transparent, white, black or green, with a MASK of the subject; **feather**
softens the edge. **Image Compare** (output) shows two images in the node with
a split that follows the mouse: A on the left, B on the right.

### output

- **Preview Image** — shows the image in the gallery and inside the node
  without keeping it: previews go to a `temp\` folder that is emptied every
  time PixelFlow starts.
- **Save Image** — saves to `outputs\` as `<prefix>_00001_.png`,
  `<prefix>_00002_.png`, … (so files sort in creation order). The prefix can
  contain subfolders (`portraits/hero`) and the tokens `%date%`, `%time%`,
  `%seed%` and `%model%`.

Both always run when they're part of your graph — they're the two ways to
actually see a result. Their PNGs embed the graph and the seed.

## Generating images

**Load Checkpoint → CLIP Text Encode → KSampler → Save Image** runs a real
Stable Diffusion pipeline (via `diffusers`) end to end:

- **Models**: SD 1.x/2.x, SDXL, SD3 and Flux checkpoints are recognised
  automatically — from a diffusers folder's `model_index.json`, or from a
  single `.safetensors`/`.ckpt` file's contents. A Hub repo id is downloaded
  the first time (only the files the pipeline needs); the progress area
  shows the download (e.g. "downloading 512 MB / 2.0 GB"). Without internet,
  a model that isn't downloaded yet fails with a clear message.
- **Precision** (`dtype` on Load Checkpoint): `auto` uses fp16 for SD/SDXL
  and bf16 for SD3/Flux on a GPU, fp32 on CPU. `fp32` is slower but avoids
  black/NaN images with some SDXL VAEs.
- **Memory**: models you switch between stay loaded while they fit in the
  GPU's memory (minus a reserve for the actual generation); older ones move
  to system RAM and come back quickly when reused. A model too big for the
  GPU on its own (e.g. Flux on a smaller card) runs with CPU offload —
  slower, but it runs. **System → Free memory** unloads everything.
- PixelFlow uses your GPU automatically if you have a CUDA-capable one and
  a CUDA build of `torch`; otherwise it runs on CPU, which works but is
  slow — the default `stabilityai/sd-turbo` checkpoint is a distilled model
  designed for 1–4 steps at cfg 0.
- Only the nodes that lead to a Save Image or Preview Image run, and a node
  whose settings and inputs haven't changed since the last run is reused:
  with a fixed `seed`, tweaking a Blur after the sampler re-runs just the
  Blur.
- Large images (over about 1536×1536) are decoded in tiles to save GPU
  memory, and a run that runs out of GPU memory is retried once that way.

## X/Y Plot

**X/Y Plot…** (next to Queue Prompt) compares settings side by side. Pick a
node and one of its widgets for X (for example the sampler's `cfg`) and
optionally for Y (its `steps`), and type the values: a comma list
(`3, 5, 7` or `euler, dpm++ 2m`) or, for numbers, a range like `4-20:4`.
**Queue grid** queues the graph once per combination -- every cell with the
same seeds, so only the two values differ -- then lays the results out in a
labelled contact sheet saved to `outputs/xy_plot/` and opened in the viewer.
A cell whose run failed is dark red. Up to 100 cells; your graph's own
values are left as they were.

## Saving and loading your work

- **Save** (<kbd>Ctrl+S</kbd>) — asks for a name the first time, then saves
  to the `workflows\` folder. Saving over an existing name asks first.
  Names can use letters, digits, spaces, `_ - . ( )` (up to 100 characters,
  not ending in a dot or space).
- **Workflows…** (<kbd>Ctrl+O</kbd>) — every saved workflow with its date:
  **Load**, **Rename**, **Duplicate** or **Delete** it.
- **Unsaved changes** — loading, importing, clearing or loading the example
  asks first when your graph has changes you haven't saved.
- **Auto-save** — your current graph is kept in the browser and comes back
  when you reload the page or reopen the app. The queue, gallery and run
  history are reloaded from the server, which keeps them across restarts.
- **Export JSON** — downloads your graph as a `.json` file.
- **Import…** — loads a graph from a `.json` file, or from a **PNG made by
  PixelFlow** (or ComfyUI): the graph that made the image is stored inside
  it. Dropping such a PNG on the canvas asks whether to load its workflow or
  use it as an image.

## The image viewer

Click a gallery image to open it. <kbd>←</kbd>/<kbd>→</kbd> (or the arrow
buttons) move between images, <kbd>Esc</kbd> closes. The side panel shows
the model, seed, sampler, steps, cfg and prompt, and offers:

- **Load workflow** — the graph that made this image;
- **Reuse seed** — puts the seed into the selected sampler (or the first
  one) and sets it to *fixed*;
- **Send to Load Image** — copies the image into the uploads and adds a Load
  Image node for it (e.g. for img2img);
- **Delete** — deletes the file.

## Undo, redo, and finding your way around

- **Undo** / **Redo** (<kbd>Ctrl+Z</kbd> / <kbd>Ctrl+Y</kbd> or
  <kbd>Ctrl+Shift+Z</kbd>) step through
  your recent edits — adding/removing nodes, connecting/disconnecting,
  moving nodes and changing values (last 100 steps). While you type in a
  text field, <kbd>Ctrl+Z</kbd> undoes your typing instead.
- **Fit View** recenters and rescales the canvas so your whole graph is
  visible.
- **Scroll** to zoom; click-drag empty canvas space to pan.
- **?** (or <kbd>?</kbd>) lists every keyboard shortcut.

## The System panel

Click the GPU name in the top-right corner: PixelFlow's version, the device
and its free memory, torch/diffusers/Python versions, where your data,
models and logs are, and which models are loaded (on the GPU, in RAM, or
offloaded). **Free memory** unloads all models and forgets the reused node
outputs and encoded prompts; **Open log folder** opens the
folder with `pixelflow.log` (useful when reporting a problem).

**Download a model**: paste a Hugging Face *file* link (open the model's
"Files" tab and the file, e.g. `https://huggingface.co/<owner>/<repo>/blob/main/model.safetensors`)
or a Civitai model page (`https://civitai.com/models/…`, the newest version
unless the link has `?modelVersionId=`) and press **Download**. The folder is
picked automatically (Civitai says what kind of model it is; for Hugging Face
it's guessed from the name) or chosen from the list. The file is checked
against the SHA-256 the site publishes and only then appears in `models\`;
a cancelled, failed or mismatching download leaves nothing behind, and an
existing file is never overwritten. Reload the page to see it in the model
lists. Gated Hugging Face models use the token `huggingface-cli login` saved
(or `HF_TOKEN`); Civitai models that need an account use `{"civitai_token":
"…"}` in `settings.json` (or `CIVITAI_TOKEN`). When a gated model refuses the
download, the list shows **Open the model page**: log in there and accept its
licence, then download again. If the connection drops, the partial file is
kept: start the same download again (even after a restart) and it carries on
where it stopped, still checked against the full SHA-256. The folder list also
has `diffusion_models`, `text_encoders` and `vae` for ComfyUI-style split models.

**Connect an assistant (MCP)** shows the configuration to paste into Claude
Desktop (Settings > Developer > Edit Config) or another MCP client, with a
**Copy configuration** button, and the one-line `claude mcp add …` for Claude
Code (see "Using PixelFlow from an AI assistant" below).

If a delete fails because another program has the file open (an image viewer,
an editor), the message names that program so you can close it. If torch
can't start because the Microsoft Visual C++ runtime is missing, the error
says which DLL is missing and links the installer.

The **Custom nodes** line lists the plugins loaded from the data folder's
`custom_nodes\` (see "Custom nodes" below).

## Custom nodes

Python files (or folders with an `__init__.py`) in the data folder's
`custom_nodes\` are loaded when PixelFlow starts, and the nodes they add show
up in the palette (how to write one: docs/NODE_AUTHORING.md). **A plugin is
ordinary Python code that runs with your permissions: only add files you
trust**, exactly as with ComfyUI's custom nodes. A plugin that fails to load
is reported in the System panel and the log without stopping PixelFlow, and
one that would replace a built-in node is refused. Turn plugins off with
`{"custom_nodes": false}` in `settings.json` or `PIXELFLOW_DISABLE_CUSTOM_NODES=1`.

## Using PixelFlow from an AI assistant (MCP)

PixelFlow can be an MCP server, so an assistant such as Claude can run your
saved workflows: "run my *Portrait* workflow with the prompt 'an astronaut
reading', seed 7". Add it to the assistant's MCP settings as a local (stdio)
server whose command is `PixelFlow.exe --mcp` (from source:
`python -m backend.mcp_server` in the project folder), and keep PixelFlow
open -- the MCP server drives the running app over its local connection and
has no network port of its own.

The assistant gets four tools: `list_workflows`, `run_workflow` (a saved
workflow, optionally with changed values such as `"CLIPTextEncode.text"` or
`"KSampler.seed"`, waiting for the images and returning their file paths),
`queue_status` and `get_outputs`. A workflow runs exactly as the Queue button
would run it; one saved by an older PixelFlow needs to be opened and saved
once first. PixelFlow looks for the app at `PIXELFLOW_URL`, then at the port
the running app noted in `server.json` in the data folder, then at 8288.

## Where your files are

- Running from source: next to the project (`outputs\`, `uploads\`,
  `workflows\`, `models\`, `logs\`).
- The packaged app: `%LOCALAPPDATA%\PixelFlow` — unless a `portable.txt`
  file sits next to `PixelFlow.exe` (then next to the exe), or an older
  PixelFlow already kept its data there.
- `wildcards\` (prompt wildcard files) and `custom_nodes\` (plugins) are in
  the data folder too; `models\embeddings\` holds textual-inversion embeddings.
- **Sharing ComfyUI's models**: if Comfy Desktop is installed, the folders
  its `extra_model_paths.yaml` lists are added after PixelFlow's own for each
  kind (checkpoints, loras, embeddings, controlnet, upscale_models,
  diffusion_models/unet, text_encoders/clip, vae), so a big library on another
  drive is used where it is. PixelFlow's own `models\` folder wins when a name
  exists in both, and downloads always go there. Point at other yaml files with
  `{"comfyui_model_paths": ["D:/ComfyUI/extra_model_paths.yaml"]}` in
  `settings.json` (or `PIXELFLOW_COMFYUI_MODEL_PATHS`), or turn it off with
  `false` / `0`.
- `PIXELFLOW_HOME` overrides the data folder; `PIXELFLOW_MODELS_DIR` (or
  `{"models_dir": "D:/sd-models"}` in `settings.json` in the data folder)
  keeps big models somewhere else, e.g. on a drive OneDrive doesn't sync.
- **Faster repeated runs (advanced, from source)**: `{"torch_compile": true}` in
  `settings.json` (or `PIXELFLOW_TORCH_COMPILE=1`) compiles the UNet with
  `torch.compile`. On Windows it needs Triton:
  `pip install -r requirements-compile.txt`. The first run at each new size or
  batch size compiles first (tens of seconds, minutes for big models); later
  runs at that size are faster. `"max-autotune"` instead of `true` compiles
  even longer for a little more speed. Without Triton the setting is ignored
  (the log says so).

## Templates and Load Example

**Templates…** lists read-only starter graphs (shipped as small JSON files in
`frontend/templates/`): **Text to image**, **Image to image** (Load Image →
img2img sampler), **Inpaint** (Load Image + Load Mask → inpaint sampler),
**Hires fix** (generate at 512, resize to 1024, re-sample at 0.4 denoise) and
**Upscale** (Load Image → 2× resize → light re-sample). Each is set up for the
default sd-turbo checkpoint (few steps, cfg 0); upload an image/mask into the
Load nodes and adjust the prompt. **Remove background** cuts the subject out of
an uploaded image, and **Compare before / after** shows an image and a
processed copy in an Image Compare node. Loading one replaces the graph (it
asks first when there are unsaved changes); tick **Add to the current graph**
to drop it beside what you have instead, selected, as a reusable block.

**Load Example** is the Text to image template: Load Checkpoint → CLIP Text Encode: Load Checkpoint → CLIP Text Encode
(with a sample prompt) → Empty Image → KSampler → Save Image, fully wired.
Its KSampler is set to 4 steps and cfg 0, which is what the default
`stabilityai/sd-turbo` checkpoint is designed for; raise them (e.g. 20–30
steps, cfg ~7) for a regular SD checkpoint.

## Test mode (trying the UI without a model)

Started with the environment variable `PIXELFLOW_TEST_MODE=1` (for example
`python tests/e2e/server.py`, which also uses a throwaway data folder in your
temp directory, then open http://127.0.0.1:8299), PixelFlow starts in about a
second without loading torch: the System panel shows "test mode", any graph
that loads a checkpoint fails with a clear "model loading is disabled in test
mode" error, and graphs made only of image nodes (Empty Image → Blur → Save
Image, …) run normally. It's what the automated browser test uses; leave it
off for real work.

## Troubleshooting

- **A run says "Error"** — the message names the failure and the node is
  outlined in red. Details, including tracebacks, are in `pixelflow.log`
  (System → Open log folder).
- **The queue/gallery stop updating** — the dot next to the PixelFlow name
  is green while connected and pulses amber while it reconnects
  (automatically). Once it's back, the queue and gallery catch up.
- **Out of GPU memory** — try a smaller size or batch, or System → Free
  memory.
- **"isn't downloaded yet and the Hugging Face Hub can't be reached"** —
  connect once to download the model, or put it in `models\checkpoints\`.
- **A right-click menu is stuck on screen** — click anywhere on the canvas.
- **Load Image can't find your file** — pick it from the dropdown or upload
  it again; the dropdown lists what's in the `uploads\` folder.
