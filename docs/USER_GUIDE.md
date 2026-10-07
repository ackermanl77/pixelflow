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
  launch opens another window on the running one.

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
- **Canny Edges** (image) — white-on-black edges, the control image a Canny
  ControlNet expects; `low`/`high` (default 100/200, OpenCV's scale) keep
  more or fewer edges.

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
offloaded). **Free memory** unloads all models; **Open log folder** opens the
folder with `pixelflow.log` (useful when reporting a problem).

## Where your files are

- Running from source: next to the project (`outputs\`, `uploads\`,
  `workflows\`, `models\`, `logs\`).
- The packaged app: `%LOCALAPPDATA%\PixelFlow` — unless a `portable.txt`
  file sits next to `PixelFlow.exe` (then next to the exe), or an older
  PixelFlow already kept its data there.
- `PIXELFLOW_HOME` overrides the data folder; `PIXELFLOW_MODELS_DIR` (or
  `{"models_dir": "D:/sd-models"}` in `settings.json` in the data folder)
  keeps big models somewhere else, e.g. on a drive OneDrive doesn't sync.

## Templates and Load Example

**Templates…** lists read-only starter graphs (shipped as small JSON files in
`frontend/templates/`): **Text to image**, **Image to image** (Load Image →
img2img sampler), **Inpaint** (Load Image + Load Mask → inpaint sampler),
**Hires fix** (generate at 512, resize to 1024, re-sample at 0.4 denoise) and
**Upscale** (Load Image → 2× resize → light re-sample). Each is set up for the
default sd-turbo checkpoint (few steps, cfg 0); upload an image/mask into the
Load nodes and adjust the prompt. Loading one replaces the graph (it asks
first when there are unsaved changes).

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
