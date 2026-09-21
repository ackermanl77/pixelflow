# PixelFlow — User Guide

A walkthrough of the app and every feature in it, for someone using PixelFlow
rather than developing it. If you're looking for architecture or how to add
a node, see [PROJECT_STATUS.md](PROJECT_STATUS.md) and
[NODE_AUTHORING.md](NODE_AUTHORING.md) instead.

## What PixelFlow is

PixelFlow lets you build an image-processing pipeline visually: drag nodes
onto a canvas, wire their inputs and outputs together, then run the whole
graph with one click. It's modeled on ComfyUI's node-graph approach, at a
much smaller scale, and with its "AI generation" nodes currently running as
CPU-only placeholders rather than real models (see
[Understanding the stub nodes](#understanding-the-stub-nodes) below).

## Starting the app

- **Dev web app**: `python run.py`, then open http://127.0.0.1:8188 in a
  browser.
- **Desktop app**: double-click `release\PixelFlow.exe` (or run
  `python desktop.py` from source) — same app, opened in its own window
  instead of a browser tab.

Either way, you get the same interface described below.

## The interface

```
┌─────────────────────────────── Topbar ───────────────────────────────┐
│ PixelFlow  [Queue Prompt] [Load Example] [Clear] [Undo][Redo][Fit View]│
│            [Save Workflow] [Load workflow ▾] [Export JSON][Import JSON]│
├───────────┬─────────────────────────────────────┬─────────────────────┤
│  Node     │                                     │  Queue              │
│  Library  │           Canvas                    │  Output Gallery     │
│ (sidebar) │      (your node graph)               │  (right panel)      │
└───────────┴─────────────────────────────────────┴─────────────────────┘
```

- **Node Library** (left) — every available node, grouped by category.
  Click one to drop it onto the canvas.
- **Canvas** (center) — where you build your graph: drag nodes around,
  drag from an output dot to an input dot to connect them.
- **Queue** (top right) — live status of everything you've run this
  session: queued, running (with the node currently executing), done, or
  errored.
- **Output Gallery** (bottom right) — every image a run has produced,
  newest first.

## Building a graph

1. **Add a node** — click a node in the sidebar (it appears centered in
   your current view), or right-click empty canvas space for litegraph's
   own "Add Node" menu, browsable by the same categories.
2. **Move it** — drag a node by its title bar.
3. **Set its values** — each node shows its own widgets: number sliders,
   text boxes, dropdowns, or a color field, depending on what that node
   needs.
4. **Connect nodes** — some inputs (an image, a model, a conditioning
   value) show as a small colored dot on the node's left edge instead of a
   widget. Drag from an output dot (right edge of the upstream node) to an
   input dot (left edge of the downstream node) to wire them together. A
   node with unconnected required inputs will fail when you run it.
5. **Delete a node** — click it, then press <kbd>Delete</kbd>.

A blank canvas shows a reminder of steps 1 and 2 until you've added at
least one node.

## Running a graph

Click **Queue Prompt**. This sends your current graph to the backend, which
runs it node by node. Watch the **Queue** panel on the right:

- *Queued* — waiting for the backend's single worker to pick it up (if
  you've queued more than one thing).
- *Running — \<Node Name\>* — which specific node is executing right now.
  That node's title bar also highlights on the canvas itself while it runs.
- *Done* — finished. Any image a **Save Image** or **Preview Image** node
  produced appears in the **Output Gallery** below.
- *Error* — something failed; the message names what went wrong (usually a
  missing/unconnected input, or a bad value like a file that doesn't
  exist).

Nothing runs in parallel — prompts you queue while one is still running
wait their turn.

## The node library, category by category

### loaders

- **Load Checkpoint (Stub)** — a `ckpt_name` text field, outputs a `MODEL`.
  Stands in for loading a real diffusion model checkpoint; see
  [Understanding the stub nodes](#understanding-the-stub-nodes).
- **Load Image** — a `filename` text field, outputs an `IMAGE`. There's no
  upload button in the UI yet: type the exact filename of an image file you
  've placed directly in the project's `uploads\` folder (next to
  `PixelFlow.exe` when packaged, or the project root in dev).

### conditioning

- **CLIP Text Encode (Stub)** — a multiline `text` box (your prompt),
  outputs `CONDITIONING`. Stands in for a real CLIP text encoder.

### sampling

- **KSampler (Stub)** — takes a `MODEL`, `CONDITIONING`, and an `IMAGE` to
  use as the starting canvas size, plus `seed`/`steps`/`cfg` number fields.
  Outputs an `IMAGE`. This is the deterministic placeholder "generator" —
  see below.

### image (real, CPU-only operations)

- **Empty Image** — `width`, `height`, and a `color` hex field. Creates a
  solid-color canvas — the usual starting point for a graph when you're not
  loading an existing photo.
- **Resize Image** — `width`/`height`. Resizes an incoming image.
- **Blur** — `radius`. Gaussian blur.
- **Brightness / Contrast** — two sliders, 0–3× each (1.0 = unchanged).
- **Hue / Saturation** — `hue_shift_degrees` (-180 to 180) and `saturation`
  (0–3×).
- **Blend** — takes a `base` and an `overlay` image, a `mode` dropdown
  (normal / multiply / screen), and an `opacity` slider.
- **Text Overlay** — draws `text` onto the image at a given `x`/`y`
  position, `size`, and `color`.

### output

- **Preview Image** — sends the incoming image to the Output Gallery
  without any other side effect.
- **Save Image** — same, but with a `filename_prefix` used when naming the
  saved file (e.g. `PixelFlow_a1b2c3d4.png`).

Both of these always run when they're part of your graph, even if nothing
else downstream depends on them — they're the two ways to actually see a
result.

## Understanding the stub nodes

**Load Checkpoint**, **CLIP Text Encode**, and **KSampler** are labeled
"(Stub)" because there's no GPU available to run a real diffusion model
here. Instead of generating an image from your prompt with an actual model,
**KSampler (Stub)** produces a deterministic gradient image with your prompt
text, seed, steps, and cfg printed on it — the same prompt and seed always
produce the same gradient. This exists so the *shape* of a generation graph
(checkpoint → prompt → sampler → save) and the queue/execution pipeline are
real and testable, without pretending to generate actual images. See
[NODE_AUTHORING.md](NODE_AUTHORING.md) if you later have a GPU and want to
wire in real generation.

## Saving and loading your work

- **Save Workflow** — opens a small dialog asking for a name, then saves
  your current graph to the backend's `workflows\` folder under that name.
- **Load workflow ▾** — a dropdown of everything saved that way; picking
  one replaces your current graph with it.
- **Export JSON** — downloads your current graph as a `.json` file you can
  keep, share, or move to another PixelFlow install.
- **Import JSON** — loads a graph from a `.json` file on your computer
  (from Export JSON, or handed to you by someone else).

There's currently no way to delete or rename a saved workflow from the UI —
do that by removing/renaming its `.json` file directly in the `workflows\`
folder.

## Undo, redo, and finding your way around

- **Undo** / **Redo** (also <kbd>Ctrl+Z</kbd> / <kbd>Ctrl+Y</kbd>) step
  through your recent graph edits — adding/removing nodes, moving them,
  changing a widget's value, connecting/disconnecting.
- **Fit View** recenters and rescales the canvas so your whole graph is
  visible — useful after you've panned or zoomed away, or after loading a
  workflow you haven't seen laid out yet. On an empty graph, it just resets
  the view to the default position and zoom.
- **Scroll** to zoom in/out on the canvas; click-drag empty canvas space to
  pan.
- **Right-click** the canvas for litegraph's own "Add Node" / "Add Group"
  menu, an alternative to the sidebar.

## Load Example

**Load Example** clears your current graph and builds a starter one:
Load Checkpoint → CLIP Text Encode (with a sample prompt already filled in)
→ Empty Image → KSampler (Stub) → Save Image, fully wired. Use it as a
quick way to see a full run end to end, or as a starting point to build
from.

## Troubleshooting

- **A run says "Error"** — the message names the failure. The most common
  cause is a required input left unconnected (every socket dot on a node
  needs a wire in, unless it has a widget instead).
- **The queue/gallery stop updating** — the browser's connection to the
  backend (a WebSocket) may have dropped, e.g. if the server restarted. Reload
  the page to reconnect.
- **A right-click menu is stuck on screen** — click anywhere on the canvas
  to dismiss it.
- **Load Image can't find your file** — the filename must match exactly
  (including extension) what's actually sitting in the `uploads\` folder.
