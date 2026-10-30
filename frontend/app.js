// PixelFlow — a small ComfyUI-style node graph editor.
// Node types are NOT hard-coded here: they are fetched from /api/nodes and
// used to dynamically build LiteGraph node classes, the same way ComfyUI's
// own web client builds nodes from its backend's object_info. Pure logic
// (prompt building, seeds, PNG metadata, text formatting, shortcuts) lives
// in logic.js, which is unit-tested under Node (tests/js).

const Logic = window.PixelFlowLogic;

// Where the graph is auto-saved so a reload (or WebView2 recovering) doesn't lose it.
const GRAPH_STORAGE_KEY = "pixelflow.graph";
const MAX_UNDO_STEPS = 100;
const MAX_GALLERY_ITEMS = 300;
const SEED_CONTROL_PROPERTY = "_seed_control"; // frontend-only: never sent (see Logic.buildPrompt)
const CUSTOM_CHOICE = "Other (type a Hugging Face Hub id)…";

let nodeDefs = {};
let registeredType = {}; // nodeName -> full LiteGraph type string e.g. "PixelFlow/image/Blur"
let nodeAliases = {}; // old class name -> current one

let graph, canvas;
let undoStack = [];
let undoIndex = -1;
let suppressHistory = false;
let historyTimer = null;

// Unsaved-changes tracking (F7): the snapshot of the last save/load.
let savedSnapshot = null;
let currentWorkflowName = "";
// What an unsaved graph was loaded from (a template title, a file name): the tab's label until
// it is saved under a name (UI10). Never used as the save name.
let currentTabHint = "";

// ---- Widgets ----

const widgetTextMeasure = document.createElement("canvas").getContext("2d");

/** Widen a node so no widget's label overprints its value (UI2); never narrows it. */
function fitNodeWidth(node) {
  if (!node.widgets || !node.widgets.length) return;
  widgetTextMeasure.font = "normal " + LiteGraph.NODE_SUBTEXT_SIZE + "px Arial"; // LiteGraph's widget font
  const need = Logic.widgetFitWidth(node.widgets, (text) => widgetTextMeasure.measureText(text).width);
  if (need > node.size[0]) node.setSize([need, node.size[1]]);
}

function addPropertyWidget(node, name, spec) {
  const type = spec[0];
  const config = spec[1] || {};
  const kind = Logic.widgetKind(spec);

  if (kind === "combo") {
    const values = type.slice();
    if (config.allow_custom) values.push(CUSTOM_CHOICE);
    node.properties[name] = config.default ?? type[0] ?? "";
    const widget = node.addWidget("combo", name, node.properties[name], async (v) => {
      if (v === CUSTOM_CHOICE) {
        // A value outside the list (e.g. a Hub repo id): the backend accepts it (allow_custom).
        const typed = await promptModal(`${name}: Hugging Face Hub repo id`, "", { confirmText: "Use" });
        widget.value = typed || node.properties[name];
        if (typed) node.properties[name] = typed;
      } else {
        node.properties[name] = v;
      }
      if (config.image_upload) node.previewImage = null; // redraw the thumbnail for the new file
      scheduleHistoryPush();
    }, { values });
    if (config.image_upload) addUploadWidget(node, name, widget);
    return;
  }

  // Anything that isn't a primitive (IMAGE, MODEL, CONDITIONING, MASK, or a
  // type a future node introduces) is a socket - same rule as the backend.
  if (kind === "socket") {
    node.addInput(name, type);
    return;
  }

  if (kind === "number") {
    node.properties[name] = Logic.normalizeNumber(config.default ?? 0, type, config);
    const widget = node.addWidget("number", name, node.properties[name], (v) => {
      const value = Logic.normalizeNumber(v, type, config);
      widget.value = value;
      node.properties[name] = value;
      scheduleHistoryPush();
    }, Logic.numberWidgetOptions(type, config));
    if (config.control_after_generate) addSeedControlWidget(node, name);
    return;
  }

  if (kind === "toggle") {
    node.properties[name] = !!(config.default ?? false);
    node.addWidget("toggle", name, node.properties[name], (v) => {
      node.properties[name] = !!v;
      scheduleHistoryPush();
    });
    return;
  }

  // STRING
  node.properties[name] = config.default ?? "";
  if (config.multiline) {
    addPromptWidget(node, name);
    return;
  }
  node.addWidget("text", name, node.properties[name], (v) => {
    node.properties[name] = v;
    scheduleHistoryPush();
  }, { multiline: false });
}

// ---- Multi-line prompt boxes drawn on the node (UI11) ----
// A STRING marked multiline is a box on the node showing its wrapped text: it grows with the text
// (Logic.PROMPT_BOX: 3 to 10 lines), then scrolls inside with the mouse wheel. A click opens a
// textarea right over it; leaving it (or Ctrl+Enter) keeps the text, Esc puts the old text back.
const PROMPT_FONT = "13px sans-serif";
const promptMeasure = document.createElement("canvas").getContext("2d");
promptMeasure.font = PROMPT_FONT;
let promptEditor = null; // { textarea, node, widget, before }

function addPromptWidget(node, name) {
  const box = Logic.PROMPT_BOX;
  const widget = {
    type: "prompt", name, value: node.properties[name], options: { multiline: true }, scroll: 0,
    callback(v) {
      node.properties[name] = v;
      scheduleHistoryPush();
    },
    wrapped(width) {
      const key = `${width}\u0000${this.value}`;
      if (this._wrapKey !== key) {
        this._wrapKey = key;
        this._lines = Logic.wrapPromptLines(this.value, width - 12 - 2 * box.padding, (t) => promptMeasure.measureText(t).width);
      }
      return this._lines;
    },
    computeSize(width) {
      return [width, Logic.promptBoxHeight(this.wrapped(width).length)];
    },
    draw(ctx, drawnNode, width, y, _height) {
      const lines = this.wrapped(width);
      const height = Logic.promptBoxHeight(lines.length);
      const { max, scroll } = Logic.promptScroll(lines.length, this.scroll);
      this.scroll = scroll;
      ctx.save();
      ctx.font = "11px sans-serif";
      ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
      ctx.textAlign = "left";
      ctx.fillText(this.label || this.name, 8, y + 10);
      const top = y + box.labelHeight, boxHeight = height - box.labelHeight;
      ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
      ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
      ctx.beginPath();
      ctx.roundRect(6, top, width - 12, boxHeight, 4);
      ctx.fill();
      ctx.stroke();
      if (promptEditor && promptEditor.widget === this) { ctx.restore(); return; } // the textarea covers it
      ctx.beginPath();
      ctx.rect(6, top + 1, width - 12, boxHeight - 2);
      ctx.clip();
      ctx.font = PROMPT_FONT;
      ctx.textBaseline = "top";
      const empty = !String(this.value || "").length;
      ctx.fillStyle = empty ? LiteGraph.WIDGET_SECONDARY_TEXT_COLOR : LiteGraph.WIDGET_TEXT_COLOR;
      const shown = empty ? ["Click to type..."] : lines;
      shown.forEach((line, i) => ctx.fillText(line, 6 + box.padding, top + box.padding + i * box.lineHeight - scroll));
      if (max > 0) { // a scrollbar thumb on the right edge
        const track = boxHeight - 4, thumb = Math.max(14, track * (track / (track + max)));
        ctx.fillStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
        ctx.fillRect(width - 10, top + 2 + (track - thumb) * (scroll / max), 3, thumb);
      }
      ctx.restore();
    },
    mouse(event, pos, clickedNode) {
      if (pos[1] - this.last_y < box.labelHeight) return false; // the label row
      // On release: litegraph refocuses the canvas after a press, which would blur the textarea.
      if (String(event.type).endsWith("up")) setTimeout(() => openPromptEditor(clickedNode, this), 0);
      return true;
    },
  };
  node.addCustomWidget(widget);
  return widget;
}

function openPromptEditor(node, widget) {
  closePromptEditor(true);
  const textarea = document.createElement("textarea");
  textarea.className = "prompt-editor";
  textarea.setAttribute("aria-label", `${widget.name} (${node.title})`);
  textarea.value = widget.value ?? "";
  document.body.appendChild(textarea);
  promptEditor = { textarea, node, widget, before: widget.value ?? "" };
  placePromptEditor();
  textarea.addEventListener("keydown", (e) => {
    e.stopPropagation(); // typing here never fires canvas shortcuts
    if (e.key === "Escape") { e.preventDefault(); closePromptEditor(false); }
    else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); closePromptEditor(true); }
  });
  textarea.addEventListener("input", () => { widget.value = textarea.value; canvas.setDirty(true, true); });
  textarea.addEventListener("blur", () => closePromptEditor(true));
  textarea.focus();
  canvas.setDirty(true, true);
}

/** Keeps the textarea on its box while the canvas pans and zooms. */
function placePromptEditor() {
  if (!promptEditor) return;
  const { textarea, node, widget } = promptEditor;
  const box = Logic.PROMPT_BOX;
  const width = node.size[0];
  const height = Logic.promptBoxHeight(widget.wrapped(width).length) - box.labelHeight;
  const rect = canvas.canvas.getBoundingClientRect();
  const [x, y] = canvas.ds.convertOffsetToCanvas([node.pos[0] + 6, node.pos[1] + widget.last_y + box.labelHeight]);
  const scale = canvas.ds.scale;
  Object.assign(textarea.style, {
    left: `${rect.left + x}px`, top: `${rect.top + y}px`,
    width: `${(width - 12) * scale}px`, height: `${height * scale}px`,
    fontSize: `${13 * scale}px`, lineHeight: `${box.lineHeight * scale}px`, padding: `${box.padding * scale}px`,
    display: graph._nodes.includes(node) ? "block" : "none",
  });
}

function closePromptEditor(keep) {
  if (!promptEditor) return;
  const { textarea, node, widget, before } = promptEditor;
  promptEditor = null;
  textarea.remove();
  const value = keep ? textarea.value : before;
  widget.value = value;
  if (value !== before) widget.callback(value);
  node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
  fitPromptNode(node);
  canvas.setDirty(true, true);
}

/** A node grows (or shrinks back) to fit its prompt boxes' heights. */
function fitPromptNode(node) {
  if ((node.widgets || []).some((w) => w.type === "prompt")) node.setSize([node.size[0], node.computeSize()[1]]);
}

/** The mouse wheel over a prompt box with more text than fits scrolls the box, not the canvas. */
function promptWheel(e) {
  if (e.ctrlKey) return;
  const rect = canvas.canvas.getBoundingClientRect();
  const [gx, gy] = canvas.ds.convertCanvasToOffset([e.clientX - rect.left, e.clientY - rect.top]);
  const node = graph.getNodeOnPos(gx, gy);
  if (!node || !node.widgets) return;
  const localY = gy - node.pos[1];
  for (const widget of node.widgets) {
    if (widget.type !== "prompt" || widget.last_y === undefined) continue;
    const lines = widget.wrapped(node.size[0]);
    if (localY < widget.last_y || localY > widget.last_y + Logic.promptBoxHeight(lines.length)) continue;
    const { max } = Logic.promptScroll(lines.length, 0);
    if (!max) return;
    widget.scroll = Logic.promptScroll(lines.length, widget.scroll + Math.sign(e.deltaY) * Logic.PROMPT_BOX.lineHeight * 3).scroll;
    e.preventDefault();
    e.stopPropagation();
    canvas.setDirty(true, true);
    return;
  }
}

// Seed "control after generate" (F11): what happens to the seed after each queue.
function addSeedControlWidget(node, seedName) {
  node.seedInputName = seedName;
  node.properties[SEED_CONTROL_PROPERTY] = "randomize";
  node.addWidget("combo", "control after generate", "randomize", (v) => {
    node.properties[SEED_CONTROL_PROPERTY] = v;
    scheduleHistoryPush();
  }, { values: Logic.SEED_CONTROLS.slice() });
}

// ---- Image upload widget (Load Image / Load Mask: specs with image_upload) ----

async function uploadImageFile(file) {
  const form = new FormData();
  form.append("file", file, file.name || "pasted.png");
  const res = await fetch("/api/upload", { method: "POST", body: form });
  if (!res.ok) {
    const { message } = await errorFromResponse(res);
    throw new Error(message);
  }
  const { filename } = await res.json();
  await refreshUploadChoices();
  return filename;
}

// Every image_upload combo on the canvas offers every uploaded file.
async function refreshUploadChoices() {
  let names;
  try {
    names = await (await fetch("/api/uploads")).json();
  } catch (_) {
    return;
  }
  for (const def of Object.values(nodeDefs)) {
    for (const group of [def.input_types.required || {}, def.input_types.optional || {}]) {
      for (const spec of Object.values(group)) {
        if (Array.isArray(spec[0]) && spec[1] && spec[1].image_upload) spec[0] = names.slice();
      }
    }
  }
  for (const node of graph._nodes) {
    for (const widget of node.widgets || []) {
      if (widget.uploadCombo) widget.options.values = names.slice();
    }
  }
}

function setUploadedImage(node, comboName, filename) {
  setNodeValue(node, comboName, filename);
  node.previewImage = null;
  scheduleHistoryPush();
  canvas.setDirty(true, true);
}

function addUploadWidget(node, comboName, comboWidget) {
  comboWidget.uploadCombo = true;
  node.uploadComboName = comboName;
  node.addWidget("button", "Upload image…", null, () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = async () => {
      const file = input.files[0];
      if (!file) return;
      try {
        setUploadedImage(node, comboName, await uploadImageFile(file));
      } catch (err) {
        showToast(`Upload failed: ${err.message}`, "error");
      }
    };
    input.click();
  });
}

// Draws `img` scaled into the space under the node's widgets, growing the
// node once to make room. Used by the upload thumbnail and output previews.
function drawImageBelowWidgets(node, ctx, img) {
  if (!img || !img.complete || !img.naturalWidth) return;
  const top = node.computeSize()[1] + 4;
  const maxW = node.size[0] - 16;
  const maxH = Math.max(0, node.size[1] - top - 8);
  if (maxH < 24) {
    node.size[1] = top + 160;
    return;
  }
  const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight);
  const w = img.naturalWidth * scale;
  const h = img.naturalHeight * scale;
  ctx.drawImage(img, (node.size[0] - w) / 2, top, w, h);
}

function loadImageForCanvas(src) {
  const img = new Image();
  img.onload = () => canvas.setDirty(true, false);
  img.src = src;
  return img;
}

// Thumbnail of the chosen upload under the node's widgets.
function drawUploadPreview(node, ctx) {
  const filename = node.properties[node.uploadComboName];
  if (!filename) return;
  if (!node.previewImage || node.previewImage.dataset.name !== filename) {
    node.previewImage = loadImageForCanvas(`/uploads/${encodeURIComponent(filename)}`);
    node.previewImage.dataset.name = filename;
  }
  drawImageBelowWidgets(node, ctx, node.previewImage);
}

// ---- Node decorations: errors, the running node, mute/bypass, previews.
// All drawn, never stored in node.color, so nothing leaks into a saved
// workflow (F16). ----

const nodeErrors = new Map(); // node id -> {message, input}
let runningNodeId = null;
const livePreviews = new Map(); // node id -> Image (latent preview while sampling, F10)
const outputPreviews = new Map(); // node id -> Image (last output of a Save/Preview node, F13)
const comparePreviews = new Map(); // node id -> [Image A, Image B] (Image Compare, F28)

// Image Compare (F28): A on the left of the split, B on the right; the split
// follows the mouse over the node (node.compareSplit, 0..1).
function drawComparePreview(node, ctx, pair) {
  const [a, b] = pair;
  if (!a.complete || !a.naturalWidth || !b.complete || !b.naturalWidth) return;
  const top = node.computeSize()[1] + 4;
  const maxW = node.size[0] - 16;
  const maxH = Math.max(0, node.size[1] - top - 8);
  if (maxH < 24) {
    node.size[1] = top + 160;
    return;
  }
  const scale = Math.min(maxW / a.naturalWidth, maxH / a.naturalHeight);
  const w = a.naturalWidth * scale;
  const h = a.naturalHeight * scale;
  const x = (node.size[0] - w) / 2;
  const split = Math.min(1, Math.max(0, node.compareSplit ?? 0.5));
  node.compareBox = [x, top, w, h];
  ctx.drawImage(a, x, top, w, h);
  ctx.save();
  ctx.beginPath();
  ctx.rect(x + w * split, top, w * (1 - split), h);
  ctx.clip();
  ctx.drawImage(b, x, top, w, h);
  ctx.restore();
  ctx.save();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x + w * split, top);
  ctx.lineTo(x + w * split, top + h);
  ctx.stroke();
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(x + 4, top + 4, 16, 16);
  ctx.fillRect(x + w - 20, top + 4, 16, 16);
  ctx.fillStyle = "#fff";
  ctx.fillText("A", x + 8, top + 16);
  ctx.fillText("B", x + w - 16, top + 16);
  ctx.restore();
}

function drawNodeDecorations(node, ctx) {
  const titleH = LiteGraph.NODE_TITLE_HEIGHT;
  if (node.mode === Logic.MODE_BYPASS || node.mode === Logic.MODE_MUTE) {
    ctx.save();
    ctx.fillStyle = node.mode === Logic.MODE_BYPASS ? "rgba(160, 60, 200, 0.28)" : "rgba(20, 20, 20, 0.55)";
    ctx.fillRect(0, -titleH, node.size[0], node.size[1] + titleH);
    ctx.fillStyle = "#ddd";
    ctx.font = "11px sans-serif";
    ctx.fillText(node.mode === Logic.MODE_BYPASS ? "bypassed" : "muted", node.size[0] - 60, -6);
    ctx.restore();
  }
  if (String(node.id) === String(runningNodeId)) {
    ctx.save();
    ctx.strokeStyle = "#5b8def";
    ctx.lineWidth = 3;
    ctx.strokeRect(-3, -titleH - 3, node.size[0] + 6, node.size[1] + titleH + 6);
    ctx.restore();
  }
  const pair = comparePreviews.get(node.id);
  const preview = livePreviews.get(node.id) || outputPreviews.get(node.id);
  if (pair) drawComparePreview(node, ctx, pair);
  else if (preview && !node.uploadComboName) drawImageBelowWidgets(node, ctx, preview);
  drawNodeError(node, ctx);
}

function drawNodeError(node, ctx) {
  const err = nodeErrors.get(node.id);
  if (!err) return;
  const titleH = LiteGraph.NODE_TITLE_HEIGHT;
  ctx.save();
  ctx.strokeStyle = "#ff4d4f";
  ctx.lineWidth = 3;
  ctx.strokeRect(-4, -titleH - 4, node.size[0] + 8, node.size[1] + titleH + 8);
  if (err.input) {
    const slot = (node.inputs || []).findIndex((i) => i.name === err.input);
    if (slot >= 0) {
      const pos = node.getConnectionPos(true, slot);
      ctx.fillStyle = "#ff4d4f";
      ctx.beginPath();
      ctx.arc(pos[0] - node.pos[0], pos[1] - node.pos[1], 6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.font = "12px sans-serif";
  const text = err.message.length > 90 ? `${err.message.slice(0, 87)}…` : err.message;
  const width = Math.min(ctx.measureText(text).width + 12, 480);
  ctx.fillStyle = "#ff4d4f";
  ctx.fillRect(0, node.size[1] + 8, width, 20);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, 6, node.size[1] + 22, width - 12);
  ctx.restore();
}

function markNodeError(nodeId, message, input) {
  const node = nodeId != null ? promptNode(nodeId, false) : null;
  if (!node) return false;
  nodeErrors.set(node.id, { message, input });
  canvas.centerOnNode(node);
  canvas.setDirty(true, true);
  return true;
}

function clearNodeErrors() {
  if (!nodeErrors.size) return;
  nodeErrors.clear();
  canvas.setDirty(true, true);
}

function setRunningNode(nodeId) {
  runningNodeId = nodeId;
  canvas.setDirty(true, true);
}

function makeNodeClass(nodeName, def) {
  function NodeCtor() {
    this.comfyClass = nodeName;
    this.properties = {};

    const required = def.input_types.required || {};
    const optional = def.input_types.optional || {};
    const all = Object.assign({}, required, optional);
    this.inputSpecs = all;
    for (const [pname, pspec] of Object.entries(all)) {
      addPropertyWidget(this, pname, pspec);
    }

    (def.return_types || []).forEach((rt, idx) => {
      const label = (def.return_names && def.return_names[idx]) || rt;
      this.addOutput(label, rt);
    });

    this.size = this.computeSize();
    if (this.uploadComboName || def.output_node) this.size[1] += 140;
    fitNodeWidth(this);
  }
  NodeCtor.prototype.onWidgetChanged = function () {
    fitNodeWidth(this);
  };
  NodeCtor.title = def.display_name;
  NodeCtor.desc = def.description || "";
  NodeCtor.prototype.onDrawForeground = function (ctx) {
    if (this.flags.collapsed) return;
    if (this.uploadComboName) drawUploadPreview(this, ctx);
    drawNodeDecorations(this, ctx);
  };
  NodeCtor.prototype.onMouseMove = function (event, localPos) {
    // Image Compare (F28): move the split with the mouse over the images.
    const box = this.compareBox;
    if (!box || !comparePreviews.has(this.id)) return;
    const [x, top, w, h] = box;
    if (localPos[1] < top || localPos[1] > top + h) return;
    this.compareSplit = Math.min(1, Math.max(0, (localPos[0] - x) / w));
    this.setDirtyCanvas(true, false);
  };
  NodeCtor.prototype.getExtraMenuOptions = function () {
    return [
      null,
      { content: this.mode === Logic.MODE_BYPASS ? "Un-bypass (Ctrl+B)" : "Bypass (Ctrl+B)", callback: () => toggleMode([this], Logic.MODE_BYPASS) },
      { content: this.mode === Logic.MODE_MUTE ? "Unmute (Ctrl+M)" : "Mute (Ctrl+M)", callback: () => toggleMode([this], Logic.MODE_MUTE) },
      ...(this.comfyClass === "LoadImage" ? [null, { content: "Open in mask editor", callback: () => openMaskEditor(this) }] : []),
      ...widgetConversionOptions(this),
      ...appModeOptions(this),
      null,
      { content: "Convert selection to subgraph", callback: () => convertToSubgraph(selectedNodes().length ? selectedNodes() : [this]) },
    ];
  };
  NodeCtor.prototype.onConfigure = function () {
    // A loaded/restored graph: its converted widgets are inputs again (their
    // sockets came back with the saved inputs; the widgets came from the constructor).
    for (const name of this.properties[CONVERTED_PROPERTY] || []) hideWidget(this, name);
    fitNodeWidth(this);
    fitPromptNode(this);
  };
  return NodeCtor;
}

// ---- Widget <-> input conversion, Reroute and Primitive nodes (F18) ----

const CONVERTED_PROPERTY = "_converted"; // frontend-only: widget names shown as inputs
const PRIMITIVE_WIDGET_TYPES = ["INT", "FLOAT", "STRING", "BOOLEAN"];

function hideWidget(node, name) {
  const index = (node.widgets || []).findIndex((w) => w.name === name);
  if (index >= 0) node.widgets.splice(index, 1);
  if (name === node.seedInputName) {
    const control = node.widgets.findIndex((w) => w.name === "control after generate");
    if (control >= 0) node.widgets.splice(control, 1);
  }
  node.setSize(node.computeSize());
}

function convertWidgetToInput(node, name) {
  const type = node.inputSpecs[name][0];
  hideWidget(node, name);
  node.addInput(name, type);
  node.properties[CONVERTED_PROPERTY] = [...(node.properties[CONVERTED_PROPERTY] || []), name];
  node.setSize(node.computeSize());
  canvas.setDirty(true, true);
  scheduleHistoryPush();
}

function convertInputToWidget(node, name) {
  const slot = findInputSlot(node, name);
  if (slot < 0 || node.inputs[slot].link != null) return;
  node.removeInput(slot);
  const value = node.properties[name];
  addPropertyWidget(node, name, node.inputSpecs[name]);
  setNodeValue(node, name, value); // addPropertyWidget resets the property to the default
  node.properties[CONVERTED_PROPERTY] = (node.properties[CONVERTED_PROPERTY] || []).filter((n) => n !== name);
  node.setSize(node.computeSize());
  canvas.setDirty(true, true);
  scheduleHistoryPush();
}

function widgetConversionOptions(node) {
  const converted = new Set(node.properties[CONVERTED_PROPERTY] || []);
  const options = [];
  for (const [name, spec] of Object.entries(node.inputSpecs || {})) {
    if (!PRIMITIVE_WIDGET_TYPES.includes(spec[0])) continue;
    if (converted.has(name)) {
      const linked = node.inputs[findInputSlot(node, name)]?.link != null;
      options.push({ content: `Convert ${name} to widget`, disabled: linked, callback: () => convertInputToWidget(node, name) });
    } else {
      options.push({ content: `Convert ${name} to input`, callback: () => convertWidgetToInput(node, name) });
    }
  }
  return options.length ? [null, ...options] : [];
}

function RerouteNode() {
  this.comfyClass = Logic.REROUTE;
  this.properties = {};
  this.addInput("", "*");
  this.addOutput("", "*");
  this.size = [90, 26];
}
RerouteNode.title = "Reroute";
RerouteNode.desc = "Passes one connection through: tidy long wires, or split one output to many inputs.";
RerouteNode.prototype.onConnectionsChange = function () {
  // Show the type it carries once wired (any type goes through).
  const link = this.inputs[0].link != null ? graph.links[this.inputs[0].link] : null;
  this.outputs[0].label = link ? link.type : "";
};

function PrimitiveNode() {
  this.comfyClass = Logic.PRIMITIVE;
  this.properties = { value: "" };
  this.addOutput("value", "INT,FLOAT,STRING,BOOLEAN");
  this.addWidget("text", "value", "", (v) => {
    this.properties.value = v;
    scheduleHistoryPush();
  });
  this.size = this.computeSize();
}
PrimitiveNode.title = "Primitive";
PrimitiveNode.desc = "One value (a seed, a prompt, a size) for several nodes: right-click a node -> Convert <widget> to input, then wire this into it.";
PrimitiveNode.prototype.onConnectOutput = function (_slot, type) {
  return PRIMITIVE_WIDGET_TYPES.includes(type);
};
PrimitiveNode.prototype.onConnectionsChange = function (kind, _slot, connected, linkInfo) {
  // First wire: start from the driven widget's current value and say which kind it is.
  if (kind !== LiteGraph.OUTPUT || !connected || !linkInfo) return;
  const target = graph.getNodeById(linkInfo.target_id);
  const input = target && target.inputs[linkInfo.target_slot];
  if (!input) return;
  this.title = `Primitive (${input.type})`;
  if (this.properties.value === "" && input.name in target.properties) {
    setNodeValue(this, "value", String(target.properties[input.name]));
  }
};

const FRONTEND_NODE_DEFS = {
  [Logic.REROUTE]: { display_name: "Reroute", category: "utils", description: RerouteNode.desc, ctor: RerouteNode },
  [Logic.PRIMITIVE]: { display_name: "Primitive", category: "utils", description: PrimitiveNode.desc, ctor: PrimitiveNode },
};

function toggleMode(nodes, mode) {
  if (!nodes.length) return;
  const allSet = nodes.every((n) => n.mode === mode);
  for (const node of nodes) node.mode = allSet ? LiteGraph.ALWAYS : mode;
  canvas.setDirty(true, true);
  scheduleHistoryPush();
}

function selectedNodes() {
  return Object.values(canvas.selected_nodes || {});
}

// ---- Groups (F18; Ctrl+G is F14's last shortcut) ----

function groupSelected() {
  const nodes = selectedNodes();
  if (!nodes.length) {
    showToast("Select the nodes to group first (Ctrl+click or drag a selection box).");
    return null;
  }
  const bounds = Logic.groupBounds(nodes.map((node) => Array.from(node.getBounding())));
  const group = new LGraphGroup("Group");
  group.pos = [bounds[0], bounds[1]];
  group.size = [bounds[2], bounds[3]];
  graph.add(group);
  canvas.setDirty(true, true);
  scheduleHistoryPush();
  return group;
}

function nodesInGroup(group) {
  group.recomputeInsideNodes();
  return group._nodes.slice();
}

// The group's context menu ("Edit Group") gains mute/bypass for everything in it.
const baseGroupMenuOptions = LGraphCanvas.prototype.getGroupMenuOptions;
LGraphCanvas.prototype.getGroupMenuOptions = function (group) {
  return [
    { content: "Mute group", callback: () => toggleMode(nodesInGroup(group), Logic.MODE_MUTE) },
    { content: "Bypass group", callback: () => toggleMode(nodesInGroup(group), Logic.MODE_BYPASS) },
    {
      content: "Select nodes",
      callback: () => {
        canvas.selectNodes(nodesInGroup(group));
        canvas.setDirty(true, true);
      },
    },
    null,
    ...baseGroupMenuOptions.call(this, group),
  ];
};

async function loadNodeDefs() {
  const res = await fetch("/api/nodes");
  nodeDefs = await res.json();
  try {
    nodeAliases = await (await fetch("/api/node_aliases")).json();
  } catch (_) {
    nodeAliases = {};
  }
  for (const [name, def] of Object.entries(nodeDefs)) {
    const cls = makeNodeClass(name, def);
    const fullType = `PixelFlow/${def.category}/${def.display_name}`;
    registeredType[name] = fullType;
    LiteGraph.registerNodeType(fullType, cls);
  }
  for (const [oldName, newName] of Object.entries(nodeAliases)) {
    if (registeredType[newName]) registeredType[oldName] = registeredType[newName];
  }
  for (const [name, def] of Object.entries(FRONTEND_NODE_DEFS)) {
    const fullType = `PixelFlow/${def.category}/${def.display_name}`;
    registeredType[name] = fullType;
    LiteGraph.registerNodeType(fullType, def.ctor);
  }
  buildSidebar();
}

// ---- Node palette: focusable buttons, click to add or drag onto the canvas (F15) ----

function buildSidebar() {
  const container = document.getElementById("node-list");
  container.innerHTML = "";
  const byCategory = {};
  const allDefs = { ...nodeDefs, ...FRONTEND_NODE_DEFS };
  for (const [name, def] of Object.entries(allDefs)) {
    (byCategory[def.category] = byCategory[def.category] || []).push(name);
  }
  for (const category of Object.keys(byCategory).sort()) {
    const group = document.createElement("div");
    group.className = "category-group";
    const title = document.createElement("h3");  // a heading, so screen readers can jump between categories
    title.className = "category-title";
    title.textContent = category;
    group.appendChild(title);

    for (const name of byCategory[category]) {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "node-item";
      item.textContent = allDefs[name].display_name;
      item.title = allDefs[name].description || `Add ${allDefs[name].display_name}`;
      item.dataset.name = allDefs[name].display_name.toLowerCase();
      item.draggable = true;
      item.onclick = () => addNodeToGraph(name);
      item.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("application/x-pixelflow-node", name);
        e.dataTransfer.effectAllowed = "copy";
      });
      group.appendChild(item);
    }
    container.appendChild(group);
  }
  filterNodeList(document.getElementById("node-search").value);
}

function filterNodeList(query) {
  const q = query.trim().toLowerCase();
  const container = document.getElementById("node-list");
  let anyVisible = false;
  for (const group of container.querySelectorAll(".category-group")) {
    let groupHasMatch = false;
    for (const item of group.querySelectorAll(".node-item")) {
      const match = !q || item.dataset.name.includes(q);
      item.classList.toggle("hidden", !match);
      if (match) groupHasMatch = true;
    }
    group.classList.toggle("hidden", !groupHasMatch);
    if (groupHasMatch) anyVisible = true;
  }
  document.getElementById("node-search-empty").classList.toggle("hidden", anyVisible);
}

function addNodeToGraph(nodeName, pos) {
  const node = LiteGraph.createNode(registeredType[nodeName]);
  const view = canvas.ds;
  const center = pos || [
    (-view.offset[0] + canvas.canvas.width / 2) / view.scale,
    (-view.offset[1] + canvas.canvas.height / 2) / view.scale,
  ];
  node.pos = center;
  canvas.graph.add(node); // the open subgraph (UI12), or the graph
  scheduleHistoryPush();
  updateEmptyState();
  return node;
}

// ---- Model Library panel (UI13) ----

const MODEL_MIME = "application/x-pixelflow-model";
let modelLibrary = null; // GET /api/model-library, loaded when the Models tab first opens

function showSidebarTab(tab) {
  for (const [name, panel] of [["nodes", "sidebar-nodes"], ["models", "sidebar-models"]]) {
    const button = document.getElementById(`tab-${name}`);
    button.setAttribute("aria-selected", String(name === tab));
    button.tabIndex = name === tab ? 0 : -1;
    document.getElementById(panel).hidden = name !== tab;
  }
  if (tab === "models" && !modelLibrary) loadModelLibrary();
}

async function loadModelLibrary() {
  const empty = document.getElementById("model-list-empty");
  try {
    const res = await fetch("/api/model-library");
    if (!res.ok) throw new Error((await errorFromResponse(res)).message);
    modelLibrary = await res.json();
  } catch (err) {
    modelLibrary = null;
    document.getElementById("model-list").innerHTML = "";
    empty.textContent = `Couldn't list the models: ${err.message}`;
    empty.classList.remove("hidden");
    return;
  }
  renderModelLibrary();
}

/** The loader for a library model, placed at `pos` (or the view's centre), with the model chosen. */
function addModelLoader(model, pos) {
  if (!registeredType[model.node]) return null;
  const node = addNodeToGraph(model.node, pos);
  setNodeValue(node, model.input, model.name);
  scheduleHistoryPush();
  canvas.setDirty(true, true);
  return node;
}

function renderModelLibrary() {
  const list = document.getElementById("model-list");
  const empty = document.getElementById("model-list-empty");
  list.innerHTML = "";
  const query = document.getElementById("model-search").value;
  const groups = Logic.filterModelLibrary(modelLibrary, query);
  for (const group of groups) {
    const section = document.createElement("div");
    section.className = "category-group";
    const title = document.createElement("h3");
    title.className = "category-title";
    title.textContent = `${group.label} (${group.items.length})`;
    section.appendChild(title);
    for (const item of group.items) {
      const model = { node: group.node, input: group.input, name: item.name };
      const card = document.createElement("button");
      card.type = "button";
      card.className = "model-item";
      card.draggable = true;
      card.title = `${item.name}\nClick or drag onto the canvas: a ${nodeDefs[group.node]?.display_name || group.node} with it`;
      const thumb = document.createElement(item.preview ? "img" : "span");
      thumb.className = "model-thumb";
      if (item.preview) {
        thumb.src = `/api/model-library/preview?kind=${encodeURIComponent(group.kind)}&name=${encodeURIComponent(item.name)}`;
        thumb.alt = "";
        thumb.loading = "lazy";
      } else {
        thumb.textContent = group.label.slice(0, 2);
        thumb.setAttribute("aria-hidden", "true");
      }
      const text = document.createElement("span");
      text.className = "model-text";
      const name = document.createElement("span");
      name.className = "model-name";
      name.textContent = Logic.modelDisplayName(item.name);
      const size = document.createElement("span");
      size.className = "model-size";
      size.textContent = item.size == null ? "Hugging Face" : Logic.formatBytes(item.size);
      text.append(name, size);
      card.append(thumb, text);
      card.onclick = () => addModelLoader(model);
      card.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData(MODEL_MIME, JSON.stringify(model));
        e.dataTransfer.effectAllowed = "copy";
      });
      section.appendChild(card);
    }
    list.appendChild(section);
  }
  const total = Object.values(modelLibrary || {}).reduce((n, g) => n + g.items.length, 0);
  empty.textContent = total ? "No models match your search."
    : "No models installed yet. Put them in models/checkpoints, loras, vae or controlnet, or download one (View > System).";
  empty.classList.toggle("hidden", groups.length > 0);
}

function updateEmptyState() {
  document.getElementById("empty-state").classList.toggle("hidden", graph._nodes.length > 0);
}

// ---- Topbar menus (UI4) ----

function closeMenus(except = null) {
  for (const button of document.querySelectorAll(".menu-button[aria-expanded='true']")) {
    if (button === except) continue;
    button.setAttribute("aria-expanded", "false");
    document.getElementById(button.getAttribute("aria-controls")).hidden = true;
  }
}

function menuItems(list) {
  return [...list.querySelectorAll("[role='menuitem']")].filter((item) => !item.disabled);
}

function initMenus() {
  for (const button of document.querySelectorAll(".menu-button")) {
    const list = document.getElementById(button.getAttribute("aria-controls"));
    const open = (focusFirst) => {
      closeMenus(button);
      button.setAttribute("aria-expanded", "true");
      list.hidden = false;
      if (focusFirst) menuItems(list)[0]?.focus();
    };
    button.onclick = () => (button.getAttribute("aria-expanded") === "true" ? closeMenus() : open(false));
    button.onkeydown = (e) => {
      if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        open(true);
      }
    };
    list.addEventListener("click", (e) => {
      if (e.target.closest("[role='menuitem']")) closeMenus();  // the item's own handler has run
    });
    list.addEventListener("keydown", (e) => {
      const items = menuItems(list);
      const index = items.indexOf(document.activeElement);
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        items[(index + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length]?.focus();
      } else if (e.key === "Escape") {
        e.stopPropagation();
        closeMenus();
        button.focus();
      } else if (e.key === "Tab") {
        closeMenus();
      }
    });
  }
  document.addEventListener("pointerdown", (e) => {
    if (!e.target.closest(".menu")) closeMenus();
  });
}

// ---- App mode (UI8) ----

/** The node menu's "Show in App mode" toggles, one per value widget. */
function appModeOptions(node) {
  const shown = new Set(node.properties[Logic.APP_INPUTS_PROPERTY] || []);
  const options = (node.widgets || [])
    .filter((w) => ["number", "combo", "toggle", "text"].includes(w.type) && w.name !== "control after generate")
    .map((w) => ({
      content: `${shown.has(w.name) ? "✓ " : ""}Show in App mode: ${w.name}`,
      callback: () => {
        node.properties[Logic.APP_INPUTS_PROPERTY] = Logic.toggledAppInputs(node.properties[Logic.APP_INPUTS_PROPERTY], w.name);
        scheduleHistoryPush();
        renderAppMode();
      },
    }));
  return options.length ? [null, ...options] : [];
}

function appModeOpen() {
  return !document.getElementById("app-mode").hidden;
}

function setAppMode(open) {
  document.getElementById("app-mode").hidden = !open;
  document.getElementById("btn-app-mode").setAttribute("aria-pressed", String(open));
  if (open) renderAppMode();
  else canvas.setDirty(true, true);
}

function appField(node, name) {
  const widget = node.widgets.find((w) => w.name === name);
  const write = (value) => {
    widget.value = value;
    if (widget.callback) widget.callback(value, canvas, node);  // the widget's own normalising and history
    canvas.setDirty(true, true);
  };
  let control;
  if (widget.type === "combo") {
    control = document.createElement("select");
    for (const value of widget.options.values || []) control.add(new Option(value, value));
    control.value = widget.value;
    control.onchange = () => write(control.value);
  } else if (widget.type === "toggle") {
    control = document.createElement("input");
    control.type = "checkbox";
    control.checked = !!widget.value;
    control.onchange = () => write(control.checked);
  } else if (widget.type === "number") {
    control = document.createElement("input");
    control.type = "number";
    Object.assign(control, { value: widget.value, min: widget.options.min ?? "", max: widget.options.max ?? "" });
    control.step = "any"; // the widget's callback rounds and clamps (Logic.normalizeNumber)
    control.onchange = () => {
      write(Number(control.value));
      control.value = widget.value; // normalised (clamped, rounded) by the widget
    };
  } else {
    control = document.createElement(widget.options && widget.options.multiline ? "textarea" : "input");
    control.value = widget.value ?? "";
    control.oninput = () => write(control.value);
  }
  control.dataset.node = node.id;
  control.dataset.input = name;
  return control;
}

/** The App mode form (the exposed inputs, or a way to pick the usual ones) and the latest result. */
function renderAppMode() {
  if (!appModeOpen()) return;
  const form = document.getElementById("app-mode-form");
  const fields = Logic.appFields(graph._nodes);
  const items = [];
  if (!fields.length) {
    const empty = document.createElement("p");
    empty.className = "app-mode-empty";
    empty.textContent = "No inputs are shown yet. Right-click a node and choose Show in App mode for each input "
      + "you want here, or start with the usual ones (prompts, seed, steps, image size).";
    const usual = document.createElement("button");
    usual.type = "button";
    usual.textContent = "Show the usual inputs";
    usual.onclick = () => {
      for (const [id, names] of Object.entries(Logic.usualAppInputs(graph._nodes))) {
        graph.getNodeById(Number(id)).properties[Logic.APP_INPUTS_PROPERTY] = names;
      }
      scheduleHistoryPush();
      renderAppMode();
    };
    items.push(empty, usual);
  }
  for (const field of fields) {
    const control = appField(field.node, field.name);
    const label = document.createElement("label");
    if (control.type === "checkbox") label.className = "inline";
    label.append(field.label, control);
    items.push(label);
  }
  const run = document.createElement("button");
  run.type = "button";
  run.className = "primary app-run";
  run.textContent = "Run";
  run.onclick = () => document.getElementById("btn-queue").click();
  items.push(run);
  form.replaceChildren(...items);
  renderAppModeResult();
}

function renderAppModeResult() {
  if (!appModeOpen()) return;
  const images = Logic.latestPromptItems(galleryItems).map((item) => {
    const img = document.createElement("img");
    img.src = Logic.imageUrl(item.filename, item.type);
    img.alt = `Result: ${item.filename}`;
    return img;
  });
  document.getElementById("app-mode-images").replaceChildren(...images);
}

function initAppMode() {
  document.getElementById("btn-app-mode").onclick = () => setAppMode(!appModeOpen());
  document.getElementById("app-mode-exit").onclick = () => setAppMode(false);
}

// ---- Canvas view toolbar and minimap (UI5) ----

const CANVAS_VIEW_KEY = "pixelflow.canvasView";
const SPLINE_LINKS = 2; // LiteGraph.SPLINE_LINK, the default
const HIDDEN_LINKS = -1; // not one of LiteGraph's link modes: drawLink() then draws nothing
let minimapPending = false;

function loadCanvasView() {
  try {
    return JSON.parse(localStorage.getItem(CANVAS_VIEW_KEY) || "{}");
  } catch (_) {
    return {};
  }
}

function saveCanvasView(change) {
  try {
    localStorage.setItem(CANVAS_VIEW_KEY, JSON.stringify({ ...loadCanvasView(), ...change }));
  } catch (_) {
    // storage full or blocked: the toggles just aren't remembered
  }
}

function zoomCanvas(scale) {
  const wrap = document.getElementById("canvas-wrap");
  canvas.ds.changeScale(scale, [wrap.clientWidth / 2, wrap.clientHeight / 2]);
  canvas.setDirty(true, true);
}

function setMinimapVisible(visible) {
  document.getElementById("minimap").hidden = !visible;
  document.getElementById("btn-minimap").setAttribute("aria-pressed", String(visible));
  if (visible) scheduleMinimap();
}

function setLinksVisible(visible) {
  canvas.links_render_mode = visible ? SPLINE_LINKS : HIDDEN_LINKS;
  document.getElementById("btn-links").setAttribute("aria-pressed", String(visible));
  canvas.setDirty(true, true);
}

function scheduleMinimap() {
  if (minimapPending || document.getElementById("minimap").hidden) return;
  minimapPending = true;
  requestAnimationFrame(() => {
    minimapPending = false;
    drawMinimap();
  });
}

function minimapLayoutNow(map) {
  const nodes = graph._nodes.map((node) => ({ x: node.pos[0], y: node.pos[1] - 20, w: node.size[0], h: node.size[1] + 20 }));
  return Logic.minimapLayout(nodes, canvas.visible_area, map.width, map.height);
}

function drawMinimap() {
  const map = document.getElementById("minimap");
  const ctx = map.getContext("2d");
  const layout = minimapLayoutNow(map);
  const at = (x, y) => [x * layout.scale + layout.ox, y * layout.scale + layout.oy];
  ctx.clearRect(0, 0, map.width, map.height);
  for (const node of graph._nodes) {
    const [x, y] = at(node.pos[0], node.pos[1] - 20);
    ctx.fillStyle = node.mode === 0 ? node.color || "#4a4b55" : "#2c2d33";
    ctx.fillRect(x, y, Math.max(2, node.size[0] * layout.scale), Math.max(2, (node.size[1] + 20) * layout.scale));
  }
  const view = canvas.visible_area;
  const [vx, vy] = at(view[0], view[1]);
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#5b8def";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(vx, vy, view[2] * layout.scale, view[3] * layout.scale);
}

function centerViewOnMinimap(event) {
  const map = document.getElementById("minimap");
  const rect = map.getBoundingClientRect();
  const [gx, gy] = Logic.minimapToGraph(minimapLayoutNow(map), event.clientX - rect.left, event.clientY - rect.top);
  const wrap = document.getElementById("canvas-wrap");
  canvas.ds.offset[0] = wrap.clientWidth / 2 / canvas.ds.scale - gx;
  canvas.ds.offset[1] = wrap.clientHeight / 2 / canvas.ds.scale - gy;
  canvas.setDirty(true, true);
}

function initCanvasToolbar() {
  const reset = document.getElementById("btn-zoom-reset");
  document.getElementById("btn-zoom-in").onclick = () => zoomCanvas(Logic.zoomStep(canvas.ds.scale, 1));
  document.getElementById("btn-zoom-out").onclick = () => zoomCanvas(Logic.zoomStep(canvas.ds.scale, -1));
  reset.onclick = () => zoomCanvas(1);
  document.getElementById("btn-canvas-fit").onclick = fitView;
  document.getElementById("btn-minimap").onclick = (e) => {
    const visible = e.currentTarget.getAttribute("aria-pressed") !== "true";
    setMinimapVisible(visible);
    saveCanvasView({ minimap: visible });
  };
  document.getElementById("btn-links").onclick = (e) => {
    const visible = e.currentTarget.getAttribute("aria-pressed") !== "true";
    setLinksVisible(visible);
    saveCanvasView({ links: visible });
  };
  const map = document.getElementById("minimap");
  map.onpointerdown = (e) => {
    map.setPointerCapture(e.pointerId);
    centerViewOnMinimap(e);
  };
  map.onpointermove = (e) => {
    if (map.hasPointerCapture(e.pointerId)) centerViewOnMinimap(e);
  };
  // Every redraw (wheel zoom, drag, edits) refreshes the zoom label and the minimap.
  canvas.onDrawOverlay = () => {
    const label = Logic.zoomPercent(canvas.ds.scale);
    if (reset.textContent !== label) {
      reset.textContent = label;
      reset.setAttribute("aria-label", `Zoom ${label}; click for 100%`);
    }
    scheduleMinimap();
    placePromptEditor(); // the prompt textarea (UI11) follows its box as the view moves
  };
  const saved = loadCanvasView();
  setMinimapVisible(saved.minimap === true);
  setLinksVisible(saved.links !== false);
}

function fitView() {
  const wrap = document.getElementById("canvas-wrap");
  const viewW = wrap.clientWidth;
  const viewH = wrap.clientHeight;

  if (!graph._nodes.length) {
    canvas.ds.scale = 1;
    canvas.ds.offset[0] = 0;
    canvas.ds.offset[1] = 0;
    canvas.setDirty(true, true);
    return;
  }

  const TITLE_HEIGHT = 20; // litegraph draws a node's title bar above node.pos
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const node of graph._nodes) {
    minX = Math.min(minX, node.pos[0]);
    minY = Math.min(minY, node.pos[1] - TITLE_HEIGHT);
    maxX = Math.max(maxX, node.pos[0] + node.size[0]);
    maxY = Math.max(maxY, node.pos[1] + node.size[1]);
  }

  const padding = 60;
  const bboxW = Math.max(maxX - minX, 1);
  const bboxH = Math.max(maxY - minY, 1);
  let scale = Math.min((viewW - padding * 2) / bboxW, (viewH - padding * 2) / bboxH);
  scale = Math.max(0.1, Math.min(scale, 1.5));

  // LiteGraph draws at (graph + offset) * scale, so the offset is in graph units.
  canvas.ds.scale = scale;
  canvas.ds.offset[0] = viewW / 2 / scale - (minX + maxX) / 2;
  canvas.ds.offset[1] = viewH / 2 / scale - (minY + maxY) / 2;
  canvas.setDirty(true, true);
}

// ---- Prompt building (graph -> ComfyUI-style execution prompt; F12 mute/bypass) ----

function buildPrompt() {
  const flat = Logic.flattenSubgraphs(plainGraph(graph)); // subgraphs (UI12) are inlined first
  return Logic.buildPrompt(flat.nodes, flat.links);
}

/** A live LGraph (and its subgraphs) as the plain data Logic.flattenSubgraphs takes. */
function plainGraph(g) {
  const kinds = { "graph/subgraph": "subgraph", "graph/input": "input", "graph/output": "output" };
  return {
    nodes: g._nodes.map((node) => ({
      id: node.id,
      mode: node.mode,
      kind: kinds[node.type] || "node",
      name: node.properties && node.properties.name,
      comfyClass: nodeAliases[node.comfyClass] || node.comfyClass,
      properties: node.properties,
      inputs: (node.inputs || []).map((inp) => ({ name: inp.name, type: inp.type, link: inp.link })),
      outputs: (node.outputs || []).map((out) => ({ name: out.name, type: out.type })),
      subgraph: node.subgraph ? plainGraph(node.subgraph) : undefined,
    })),
    links: g.links,
  };
}

/** Every node, those inside subgraphs too. */
function allNodes(g = graph) {
  return g._nodes.flatMap((node) => [node, ...(node.subgraph ? allNodes(node.subgraph) : [])]);
}

/** The canvas node a /prompt id names: "7:3" is node 3 inside subgraph node 7. `inner` false
 *  gives the top-level node (the subgraph node), for errors and progress the user can see. */
function promptNode(id, inner = true) {
  const path = String(id).split(":").map(Number);
  let g = graph, node = null;
  for (const part of inner ? path : path.slice(0, 1)) {
    node = g ? g.getNodeById(part) : null;
    g = node && node.subgraph;
  }
  return node;
}

// ---- Subgraphs (UI12): a selection becomes one node; open it in place, Esc back out ----

function convertToSubgraph(selection) {
  const g = canvas.graph;
  const picked = selection.filter((n) => n.graph === g && !["graph/input", "graph/output"].includes(n.type));
  if (!picked.length) return null;
  const ids = new Set(picked.map((n) => n.id));
  const wires = Object.values(g.links || {}).filter(Boolean);
  const inside = wires.filter((l) => ids.has(l.origin_id) && ids.has(l.target_id));
  const incoming = wires.filter((l) => !ids.has(l.origin_id) && ids.has(l.target_id));
  const outgoing = wires.filter((l) => ids.has(l.origin_id) && !ids.has(l.target_id));
  const key = (l) => `${l.origin_id}/${l.origin_slot}`;
  const sources = [...new Map(incoming.map((l) => [key(l), l])).values()];
  const results = [...new Map(outgoing.map((l) => [key(l), l])).values()];
  const inNames = Logic.uniqueNames(sources.map((l) => g.getNodeById(l.target_id).inputs[l.target_slot].name));
  const outNames = Logic.uniqueNames(results.map((l) => {
    const out = g.getNodeById(l.origin_id).outputs[l.origin_slot];
    return out.label || out.name || String(out.type).toLowerCase();
  }));

  const left = Math.min(...picked.map((n) => n.pos[0])), top = Math.min(...picked.map((n) => n.pos[1]));
  const right = Math.max(...picked.map((n) => n.pos[0] + n.size[0]));
  const sub = LiteGraph.createNode("graph/subgraph");
  sub.title = "Subgraph";
  // Where the most downstream picked node was: the space the selection leaves, near its wires.
  sub.pos = picked.reduce((a, b) => (b.pos[0] > a.pos[0] ? b : a)).pos.slice();
  g.add(sub);
  const inner = sub.subgraph;
  const copies = new Map();
  for (const node of picked) {
    const data = node.serialize();
    for (const input of data.inputs || []) input.link = null;
    for (const output of data.outputs || []) output.links = null;
    const copy = LiteGraph.createNode(node.type);
    copy.configure(data);
    inner.add(copy);
    copies.set(node.id, copy);
  }
  for (const l of inside) copies.get(l.origin_id).connect(l.origin_slot, copies.get(l.target_id), l.target_slot);
  // A graph input/output node inside; naming it adds the subgraph node's input/output (litegraph).
  const port = (nodeType, name, valueType, index, x) => {
    const io = LiteGraph.createNode(nodeType);
    io.pos = [x, top + index * 120];
    inner.add(io);
    io.setProperty("type", valueType);
    io.setProperty("name", name);
    return io;
  };
  sources.forEach((l, i) => {
    const io = port("graph/input", inNames[i], l.type, i, left - 220);
    for (const target of incoming.filter((t) => key(t) === key(l))) io.connect(0, copies.get(target.target_id), target.target_slot);
  });
  results.forEach((l, i) => {
    const io = port("graph/output", outNames[i], l.type, i, right + 60);
    copies.get(l.origin_id).connect(l.origin_slot, io, 0);
  });
  // Outside: the same wires, now to and from the subgraph node.
  const outerIn = sources.map((l) => [g.getNodeById(l.origin_id), l.origin_slot]);
  const outerOut = results.map((l) => outgoing.filter((t) => key(t) === key(l)).map((t) => [g.getNodeById(t.target_id), t.target_slot]));
  for (const node of picked) g.remove(node);
  outerIn.forEach(([origin, slot], i) => origin && origin.connect(slot, sub, sub.findInputSlot(inNames[i])));
  outerOut.forEach((targets, i) => {
    for (const [target, slot] of targets) if (target) sub.connect(sub.findOutputSlot(outNames[i]), target, slot);
  });
  sub.setSize(sub.computeSize());
  canvas.deselectAllNodes();
  canvas.selectNode(sub);
  canvas.setDirty(true, true);
  clearTimeout(historyTimer);
  pushHistory(); // one undo step now: a Ctrl+Z right after must not go past the conversion
  return sub;
}

function openSubgraph(node) {
  if (!node || !node.subgraph) return;
  canvas.openSubgraph(node.subgraph);
}

/** Back out of every open subgraph: before the graph is replaced (undo, load, a new tab). */
function leaveSubgraphs() {
  while (canvas && canvas._graph_stack && canvas._graph_stack.length) canvas.closeSubgraph();
}

/** The bar over the canvas while a subgraph is open: where you are and the way back. */
function updateSubgraphBar() {
  const bar = document.getElementById("subgraph-bar");
  const depth = (canvas._graph_stack || []).length;
  bar.classList.toggle("hidden", depth === 0);
  if (!depth) return;
  const titles = [];
  for (let g = canvas.graph; g && g._subgraph_node; g = g._subgraph_node.graph) titles.unshift(g._subgraph_node.title);
  document.getElementById("subgraph-path").textContent = ["Graph", ...titles].join(" › ");
}

// ---- Queueing (F8, F11) ----

function seedNodes() {
  return allNodes().filter((n) => n.seedInputName && n.mode !== Logic.MODE_MUTE && n.mode !== Logic.MODE_BYPASS);
}

async function queuePrompt({ front = false, count = 1 } = {}) {
  clearNodeErrors();
  for (let i = 0; i < count; i++) {
    // Never send -1: resolve random seeds here, show them in the widget, and
    // apply each node's "control after generate" once the prompt is queued.
    const used = new Map();
    for (const node of seedNodes()) {
      const seed = Logic.seedForQueue(node.properties[node.seedInputName]);
      used.set(node, seed);
      setNodeValue(node, node.seedInputName, seed);
    }
    const ok = await submitPrompt(front);
    if (!ok) return;
    for (const [node, seed] of used) {
      setNodeValue(node, node.seedInputName, Logic.seedAfterQueue(node.properties[SEED_CONTROL_PROPERTY], seed));
    }
  }
  scheduleHistoryPush();
}

async function submitPrompt(front) {
  const execPrompt = buildPrompt();
  let res;
  try {
    res = await fetch("/api/queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // The litegraph workflow rides along so Save/Preview Image can embed it
      // in the PNG (ComfyUI's extra_data.extra_pnginfo shape).
      body: JSON.stringify({ prompt: execPrompt, front, client_id: CLIENT_ID, extra_data: { extra_pnginfo: { workflow: graph.serialize() } } }),
    });
  } catch (err) {
    showToast(`Could not reach the server: ${err.message}`, "error");
    return false;
  }
  if (!res.ok) {
    const { message, nodeId, input } = await errorFromResponse(res);
    markNodeError(nodeId, message, input);
    showToast(`Failed to queue prompt: ${message}`, "error");
    return false;
  }
  const data = await res.json();
  // A quick run's WebSocket messages can arrive before this response: don't
  // turn its "running"/"completed" back into "queued".
  if (!queueState.entries.has(data.prompt_id)) {
    queueState.entries.set(data.prompt_id, { status: "queued", queued_at: Date.now() / 1000, client_id: CLIENT_ID });
    if (front) queueState.pending.unshift(data.prompt_id);
    else if (!queueState.pending.includes(data.prompt_id)) queueState.pending.push(data.prompt_id);
  }
  renderQueueRow(data.prompt_id);
  return data.prompt_id;
}

// ---- Queue panel (F8) ----

// Every window (tab) has its own client id, kept across reloads in
// sessionStorage: it rides along with each queued prompt, so a window shows
// its own prompts, progress and images -- other windows' only with "All
// windows" -- and never marks its canvas with another window's node ids.
const CLIENT_ID_KEY = "pixelflow-client-id";
const SHOW_ALL_WINDOWS_KEY = "pixelflow-queue-all-windows";
const CLIENT_ID = (() => {
  const fresh = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`).replace(/[^0-9a-z]/gi, "").slice(0, 32);
  try {
    let id = sessionStorage.getItem(CLIENT_ID_KEY);
    if (!id) {
      id = fresh();
      sessionStorage.setItem(CLIENT_ID_KEY, id);
    }
    return id;
  } catch (_) {
    return fresh();
  }
})();
let showAllWindows = false;
try {
  showAllWindows = localStorage.getItem(SHOW_ALL_WINDOWS_KEY) === "1";
} catch (_) { /* storage blocked: default view */ }

function isMine(promptId) {
  return Logic.isOwnPrompt(queueState.entries.get(promptId) || historyCache.get(promptId), CLIENT_ID);
}

function isShown(promptId) {
  return showAllWindows || isMine(promptId);
}

function setShowAllWindows(value) {
  showAllWindows = value;
  try {
    localStorage.setItem(SHOW_ALL_WINDOWS_KEY, value ? "1" : "0");
  } catch (_) { /* not remembered */ }
  renderAllQueueRows();
  setGallery(Logic.galleryItems(Logic.visibleHistory(Object.fromEntries(historyCache), CLIENT_ID, showAllWindows)));
}

const queueState = {
  entries: new Map(), // prompt id -> {status, queued_at, started_at, finished_at, error, class_type, outputs}
  pending: [], // ids in run order
  rows: new Map(), // prompt id -> element
};

function renderQueueRow(promptId) {
  const entry = queueState.entries.get(promptId) || { status: "queued" };
  let el = queueState.rows.get(promptId);
  if (!el) {
    el = document.createElement("div");
    el.className = "queue-entry";
    document.getElementById("queue-list").prepend(el);
    queueState.rows.set(promptId, el);
  }
  el.hidden = !isShown(promptId);
  el.classList.toggle("foreign", !isMine(promptId));
  // Server-provided text (error messages, filenames, prompt ids) is only
  // ever inserted via textContent, never innerHTML.
  const [statusClass, text] = Logic.statusRow(promptId, entry, entry.status === "running");
  el.replaceChildren();
  const label = document.createElement("span");
  label.className = statusClass;
  label.textContent = entry.runningText || text;
  const detail = document.createElement("small");
  detail.className = "queue-detail";
  detail.textContent = Logic.queueDetail(entry, queueState.pending.indexOf(promptId), Date.now() / 1000)
    + (isMine(promptId) ? "" : " · another window");
  el.append(label, detail);
  if (entry.status === "queued") {
    el.appendChild(queueActionButton("Cancel", "Remove from the queue", () => cancelQueued(promptId)));
  } else if (entry.status === "running") {
    el.appendChild(queueActionButton("Cancel", "Stop this run", () => interruptRunning(promptId)));
  }
}

function renderAllQueueRows() {
  for (const promptId of queueState.rows.keys()) renderQueueRow(promptId);
}

// X/Y plot runs waiting for their prompts to finish: prompt id -> resolve(outputs | null).
const xyWaiters = new Map();

function updateQueueEntry(promptId, fields) {
  if (xyWaiters.has(promptId) && ["completed", "error", "cancelled", "interrupted"].includes(fields.status)) {
    xyWaiters.get(promptId)(fields.status === "completed" ? fields.outputs || {} : null);
    xyWaiters.delete(promptId);
  }
  const entry = { ...(queueState.entries.get(promptId) || {}), ...fields };
  queueState.entries.set(promptId, entry);
  if (entry.status !== "queued") queueState.pending = queueState.pending.filter((id) => id !== promptId);
  renderQueueRow(promptId);
  if (fields.status && fields.status !== "running") renderAllQueueRows(); // positions moved
}

function queueActionButton(text, title, onClick) {
  const btn = document.createElement("button");
  btn.className = "queue-action";
  btn.textContent = text;
  btn.title = title;
  btn.onclick = async () => {
    btn.disabled = true;
    await onClick();
  };
  return btn;
}

async function cancelQueued(promptId) {
  const res = await fetch(`/api/queue/${encodeURIComponent(promptId)}`, { method: "DELETE" });
  // 404 means it already started running (or finished) -- the websocket
  // messages for that will update this entry, so there's nothing to do here.
  if (res.ok) updateQueueEntry(promptId, { status: "cancelled" });
}

async function interruptRunning(promptId) {
  // With the row's id, the server only stops *that* prompt (B11): a stale
  // button can't cancel a different prompt that started in the meantime.
  const res = await fetch("/api/interrupt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt_id: promptId }),
  });
  if (res.status === 409) showToast("That prompt is no longer running.");
}

// Without "All windows", Cancel all / Clear finished leave other windows' prompts alone.
function scopeQuery() {
  return showAllWindows ? "" : `?client_id=${encodeURIComponent(CLIENT_ID)}`;
}

async function cancelAllPending() {
  const res = await fetch(`/api/queue${scopeQuery()}`, { method: "DELETE" });
  if (!res.ok) return;
  for (const promptId of (await res.json()).cancelled) updateQueueEntry(promptId, { status: "cancelled" });
}

async function clearFinished() {
  const res = await fetch(`/api/history${scopeQuery()}`, { method: "DELETE" });
  if (!res.ok) return;
  for (const promptId of (await res.json()).deleted) {
    const row = queueState.rows.get(promptId);
    if (row) row.remove();
    queueState.rows.delete(promptId);
    queueState.entries.delete(promptId);
  }
}

// Running rows show elapsed time; refresh them once a second.
setInterval(() => {
  for (const [promptId, entry] of queueState.entries) if (entry.status === "running") renderQueueRow(promptId);
}, 1000);

// ---- Gallery and image viewer (F9) ----

let galleryItems = []; // oldest first: {filename, type, promptId, meta}
let lightboxIndex = -1;
const historyCache = new Map(); // prompt id -> history entry (for workflows / metadata)

function setGallery(items) {
  galleryItems = Logic.capItems(items, MAX_GALLERY_ITEMS);
  renderGallery();
  renderAppModeResult();
}

function addGalleryItems(items) {
  setGallery(galleryItems.concat(items));
}

function renderGallery() {
  const gallery = document.getElementById("gallery");
  gallery.replaceChildren();
  let lastPrompt = null;
  // Newest first, grouped by the prompt that made them.
  for (let i = galleryItems.length - 1; i >= 0; i--) {
    const item = galleryItems[i];
    if (item.promptId !== lastPrompt) {
      const header = document.createElement("div");
      header.className = "gallery-group";
      header.textContent = item.meta && item.meta.seed !== undefined ? `${item.promptId} · seed ${item.meta.seed}` : item.promptId;
      gallery.appendChild(header);
      lastPrompt = item.promptId;
    }
    const cell = document.createElement("div");
    cell.className = `gallery-item ${item.type}`;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "thumb";
    button.title = item.type === "temp" ? `${item.filename} (preview, not saved)` : item.filename;
    const img = document.createElement("img");
    img.src = Logic.imageUrl(item.filename, item.type);
    img.alt = item.filename;
    img.loading = "lazy";
    // Previews of an earlier server run are gone (temp/ is emptied at start).
    img.onerror = () => cell.remove();
    const index = i;
    button.onclick = () => openLightbox(index);
    button.appendChild(img);
    const caption = document.createElement("div");
    caption.className = "caption";
    caption.textContent = item.filename;
    cell.append(button, caption);
    gallery.appendChild(cell);
  }
}

function openLightbox(index) {
  if (!galleryItems.length) return;
  lightboxIndex = Logic.stepIndex(index, 0, galleryItems.length);
  const item = galleryItems[lightboxIndex];
  document.getElementById("lightbox-img").src = Logic.imageUrl(item.filename, item.type);
  document.getElementById("lightbox-img").alt = item.filename;
  document.getElementById("lightbox-name").textContent = item.filename;
  const details = document.getElementById("lightbox-details");
  details.replaceChildren();
  const rows = Object.entries(item.meta || {});
  if (item.type === "temp") rows.unshift(["kind", "preview (not saved)"]);
  for (const [key, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    details.append(dt, dd);
  }
  document.getElementById("lightbox-reuse-seed").disabled = !(item.meta && item.meta.seed !== undefined);
  const overlay = document.getElementById("lightbox-overlay");
  if (overlay.classList.contains("hidden")) {
    rememberFocus();
    overlay.classList.remove("hidden");
    document.getElementById("lightbox-close").focus();
  }
}

function closeLightbox() {
  document.getElementById("lightbox-overlay").classList.add("hidden");
  document.getElementById("lightbox-img").src = "";
  lightboxIndex = -1;
  restoreFocus();
}

function currentGalleryItem() {
  return lightboxIndex >= 0 ? galleryItems[lightboxIndex] : null;
}

async function historyEntry(promptId) {
  if (historyCache.has(promptId)) return historyCache.get(promptId);
  try {
    const res = await fetch(`/api/history/${encodeURIComponent(promptId)}`);
    if (!res.ok) return null;
    const entry = await res.json();
    historyCache.set(promptId, entry);
    return entry;
  } catch (_) {
    return null;
  }
}

async function lightboxLoadWorkflow() {
  const item = currentGalleryItem();
  if (!item) return;
  const entry = await historyEntry(item.promptId);
  let workflow = entry && entry.workflow;
  if (!workflow) {
    // Fall back to the workflow embedded in the PNG itself.
    try {
      const buf = await (await fetch(Logic.imageUrl(item.filename, item.type))).arrayBuffer();
      workflow = Logic.workflowFromPngChunks(Logic.readPngTextChunks(buf));
    } catch (_) { /* reported below */ }
  }
  if (!workflow) {
    showToast("No workflow is stored with this image.", "error");
    return;
  }
  if (await replaceGraph(workflow, "Load the workflow that made this image?")) closeLightbox();
}

function lightboxReuseSeed() {
  const item = currentGalleryItem();
  if (!item || item.meta.seed === undefined) return;
  const nodes = seedNodes();
  if (!nodes.length) {
    showToast("There's no sampler in the graph to take the seed.", "error");
    return;
  }
  const target = nodes.find((n) => selectedNodes().includes(n)) || nodes[0];
  setNodeValue(target, target.seedInputName, item.meta.seed);
  setNodeValue(target, SEED_CONTROL_PROPERTY, "fixed");
  scheduleHistoryPush();
  showToast(`Seed ${item.meta.seed} set on ${target.title} (control: fixed).`);
}

async function lightboxSendToLoadImage() {
  const item = currentGalleryItem();
  if (!item) return;
  const res = await fetch("/api/outputs/to-upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: item.filename, type: item.type }),
  });
  if (!res.ok) {
    showToast(`Could not copy the image: ${(await errorFromResponse(res)).message}`, "error");
    return;
  }
  const { filename } = await res.json();
  await refreshUploadChoices();
  const node = addNodeToGraph("LoadImage");
  setUploadedImage(node, node.uploadComboName, filename);
  closeLightbox();
  showToast(`Added a Load Image node for ${filename}.`);
}

async function lightboxDelete() {
  const item = currentGalleryItem();
  if (!item) return;
  if (!(await confirmModal("Delete image?", `${item.filename} will be deleted from disk.`, "Delete"))) return;
  const url = `/api/outputs/${item.filename.split("/").map(encodeURIComponent).join("/")}?type=${item.type}`;
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok && res.status !== 404) {
    showToast(`Could not delete: ${(await errorFromResponse(res)).message}`, "error");
    return;
  }
  const index = lightboxIndex;
  galleryItems.splice(index, 1);
  renderGallery();
  if (galleryItems.length) openLightbox(Math.min(index, galleryItems.length - 1));
  else closeLightbox();
}

// ---- Progress (F10) ----

const progress = { nodeIndex: 0, nodeCount: 0, title: "" };

function setProgressBar(id, fraction) {
  const bar = document.getElementById(id);
  const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  bar.querySelector(".progress-fill").style.width = `${pct}%`;
  bar.setAttribute("aria-valuenow", String(pct));
}

function showNodeStart(msg) {
  const node = promptNode(msg.node_id, false);
  Object.assign(progress, {
    nodeIndex: msg.node_index || 0,
    nodeCount: msg.node_count || 0,
    title: node ? node.title || node.comfyClass : msg.class_type,
  });
  document.getElementById("progress-wrap").classList.remove("hidden");
  document.getElementById("progress-preview").classList.add("hidden");
  document.getElementById("progress-label").textContent = Logic.progressText(progress);
  setProgressBar("progress-overall", Logic.overallProgress(progress.nodeIndex, progress.nodeCount, 0, 0));
  setProgressBar("progress-track", 0);
}

function showProgress(msg) {
  const text = Logic.progressText({
    ...progress,
    step: msg.step,
    total: msg.total,
    itPerS: msg.it_per_s,
    etaS: msg.eta_s,
    unit: msg.unit,
  });
  document.getElementById("progress-wrap").classList.remove("hidden");
  document.getElementById("progress-label").textContent = text;
  setProgressBar("progress-track", msg.total ? msg.step / msg.total : 0);
  setProgressBar("progress-overall", Logic.overallProgress(progress.nodeIndex, progress.nodeCount, msg.step, msg.total));
}

function showPreview(msg) {
  const img = document.getElementById("progress-preview");
  img.src = msg.image;
  img.classList.remove("hidden");
  const node = promptNode(msg.node_id, false);
  if (node) {
    livePreviews.set(node.id, loadImageForCanvas(msg.image));
    canvas.setDirty(true, false);
  }
}

function hideProgress() {
  document.getElementById("progress-wrap").classList.add("hidden");
  setProgressBar("progress-track", 0);
  setProgressBar("progress-overall", 0);
  livePreviews.clear();
  setRunningNode(null);
}

// ---- Toasts (replace alert(): non-blocking, dismissable) ----

function showToast(message, kind = "info", timeoutMs = kind === "error" ? 12000 : 4000) {
  const area = document.getElementById("toast-area");
  const toast = document.createElement("div");
  toast.className = `toast toast-${kind}`;
  toast.setAttribute("role", kind === "error" ? "alert" : "status");
  const text = document.createElement("span");
  text.textContent = message;
  const close = document.createElement("button");
  close.className = "toast-close";
  close.textContent = "×";
  close.title = "Dismiss";
  close.setAttribute("aria-label", "Dismiss");
  const dismiss = () => toast.remove();
  close.onclick = dismiss;
  toast.append(text, close);
  area.appendChild(toast);
  if (timeoutMs) setTimeout(dismiss, timeoutMs);
  return toast;
}

async function errorFromResponse(res) {
  const body = await res.text();
  let detail = body || `${res.status} ${res.statusText}`;
  try {
    detail = JSON.parse(body).detail ?? detail;
  } catch (_) { /* not JSON -- show it as-is */ }
  return Logic.parseErrorDetail(detail);
}

// ---- Server state re-sync (page load and every WebSocket reconnect) ----

async function resyncState() {
  try {
    const [queueRes, historyRes] = await Promise.all([fetch("/api/queue"), fetch("/api/history?limit=100")]);
    if (!queueRes.ok || !historyRes.ok) return;
    const serverQueue = await queueRes.json();
    const historyEntries = await historyRes.json();
    document.getElementById("queue-list").replaceChildren();
    queueState.rows.clear();
    queueState.entries.clear();
    queueState.pending = serverQueue.pending.slice();
    for (const [promptId, entry] of Object.entries(historyEntries)) {
      historyCache.set(promptId, entry);
      queueState.entries.set(promptId, { ...entry, status: promptId === serverQueue.running ? "running" : entry.status });
      renderQueueRow(promptId);
    }
    setGallery(Logic.galleryItems(Logic.visibleHistory(historyEntries, CLIENT_ID, showAllWindows)));
    if (!serverQueue.running || !isMine(serverQueue.running)) hideProgress();
  } catch (err) {
    console.warn("could not re-sync queue/history", err);
  }
}

function setConnectionStatus(state) {
  const dot = document.getElementById("conn-status");
  dot.dataset.state = state;
  const label = { connected: "Connected to the PixelFlow server", reconnecting: "Reconnecting to the server…" }[state];
  dot.title = label;
  dot.setAttribute("aria-label", label);
}

let reconnectAttempt = 0;

async function onExecutionSuccess(msg) {
  updateQueueEntry(msg.prompt_id, { status: "completed", outputs: msg.outputs, finished_at: Date.now() / 1000 });
  const mine = isMine(msg.prompt_id);
  if (mine) hideProgress();
  const entry = await historyEntry(msg.prompt_id);
  if (entry) {
    historyCache.set(msg.prompt_id, entry);
    queueState.entries.set(msg.prompt_id, entry);
    renderQueueRow(msg.prompt_id);
  }
  if (!mine && !showAllWindows) return;
  const items = Logic.galleryItems({ [msg.prompt_id]: entry || { status: "completed", outputs: msg.outputs } });
  addGalleryItems(items);
  // Another window's node ids mean nothing in this graph.
  if (!mine) return;
  // F13: show each output node's newest image inside the node.
  for (const [nodeId, output] of Object.entries(msg.outputs || {})) {
    const node = promptNode(nodeId);
    if (node && output && output.compare && Array.isArray(output.images) && output.images.length === 2) {
      comparePreviews.set(node.id, output.images.map((name) => loadImageForCanvas(Logic.imageUrl(name, output.type))));
    } else if (node && output && Array.isArray(output.images) && output.images.length) {
      outputPreviews.set(node.id, loadImageForCanvas(Logic.imageUrl(output.images[output.images.length - 1], output.type)));
    }
  }
  canvas.setDirty(true, true);
}

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    reconnectAttempt = 0;
    setConnectionStatus("connected");
    // Anything that happened while the socket was down (a prompt finishing,
    // a server restart) is picked up from the server's own state.
    resyncState();
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    // Progress, previews and node marks of another window's prompt stay
    // out of this window (its queue row still follows along).
    const mine = !msg.prompt_id || isMine(msg.prompt_id);
    switch (msg.type) {
      case "queued":
        if (!queueState.entries.has(msg.prompt_id)) {
          queueState.entries.set(msg.prompt_id, { status: "queued", queued_at: Date.now() / 1000, client_id: msg.client_id || null });
          queueState.pending.push(msg.prompt_id);
        }
        renderQueueRow(msg.prompt_id);
        break;
      case "execution_start":
        updateQueueEntry(msg.prompt_id, { status: "running", started_at: Date.now() / 1000, runningText: "" });
        if (mine) hideProgress();
        break;
      case "executing":
        updateQueueEntry(msg.prompt_id, { status: "running", runningText: `Running: ${msg.prompt_id} — ${msg.class_type}` });
        if (!mine) break;
        setRunningNode(msg.node_id);
        livePreviews.clear();
        showNodeStart(msg);
        break;
      case "execution_cached":
        if (mine && msg.node_index) {
          progress.nodeIndex = msg.node_index;
          progress.nodeCount = msg.node_count;
          setProgressBar("progress-overall", Logic.overallProgress(progress.nodeIndex, progress.nodeCount, 1, 1));
        }
        break;
      case "progress":
        if (mine) showProgress(msg);
        break;
      case "preview":
        if (mine) showPreview(msg);
        break;
      case "execution_success":
        onExecutionSuccess(msg);
        break;
      case "execution_cancelled":
        updateQueueEntry(msg.prompt_id, { status: "cancelled" });
        break;
      case "execution_interrupted":
        updateQueueEntry(msg.prompt_id, { status: "interrupted", finished_at: Date.now() / 1000 });
        if (mine) hideProgress();
        break;
      case "execution_error":
        updateQueueEntry(msg.prompt_id, { status: "error", error: msg.error, class_type: msg.class_type, finished_at: Date.now() / 1000 });
        if (!mine) break;
        hideProgress();
        markNodeError(msg.node_id, msg.error);
        showToast(msg.class_type ? `${msg.class_type} failed: ${msg.error}` : `Run failed: ${msg.error}`, "error");
        break;
      default:
        break;
    }
  };
  ws.onclose = () => {
    setConnectionStatus("reconnecting");
    setTimeout(connectWebSocket, Logic.reconnectDelay(reconnectAttempt++));
  };
}

// ---- Undo / redo (F6) ----

function scheduleHistoryPush() {
  if (suppressHistory) return;
  clearTimeout(historyTimer);
  historyTimer = setTimeout(pushHistory, 400);
}

function pushHistory() {
  if (suppressHistory) return;
  const snap = JSON.stringify(graph.serialize());
  saveGraphLocally(snap);
  [undoStack, undoIndex] = Logic.pushUndo(undoStack, undoIndex, snap, MAX_UNDO_STEPS);
}

function isDirty() {
  if (!graph._nodes.length) return false;
  return savedSnapshot === null || JSON.stringify(graph.serialize()) !== savedSnapshot;
}

function markSaved(name, hint = "") {
  savedSnapshot = JSON.stringify(graph.serialize());
  if (name !== undefined) {
    currentWorkflowName = name;
    currentTabHint = hint;
  }
  persistTabs();
  renderTabs();
}

// Auto-save: every undo snapshot is also the "last graph" restored on reload, and the open
// tab's graph in the stored tab list (UI6).
function saveGraphLocally(snap) {
  try {
    localStorage.setItem(GRAPH_STORAGE_KEY, snap);
  } catch (_) { /* storage full or disabled -- auto-save is best effort */ }
  const tab = activeTab();
  if (tab) tab.graph = snap;
  persistTabs();
  renderTabs();
}

// ---- Workflow tabs (UI6) ----

const TABS_STORAGE_KEY = "pixelflow.tabs";
let tabs = []; // {id, name, graph, saved} (+ undoStack, undoIndex, view while the page is open)
let activeTabId = null;

function activeTab() {
  return tabs.find((tab) => tab.id === activeTabId);
}

/** The open tab's live state (globals) copied into its record. */
function stashActiveTab() {
  const tab = activeTab();
  if (!tab) return;
  Object.assign(tab, {
    graph: JSON.stringify(graph.serialize()),
    saved: savedSnapshot,
    name: currentWorkflowName,
    hint: currentTabHint,
    undoStack,
    undoIndex,
    view: { scale: canvas.ds.scale, offset: [...canvas.ds.offset] },
  });
}

function persistTabs() {
  if (!tabs.length) return; // not restored yet: never overwrite the stored tabs with none
  const tab = activeTab();
  if (tab) Object.assign(tab, { saved: savedSnapshot, name: currentWorkflowName, hint: currentTabHint });
  try {
    localStorage.setItem(TABS_STORAGE_KEY, Logic.storeTabs(tabs, activeTabId));
  } catch (_) { /* best effort, like the graph auto-save */ }
}

/** Put `tab`'s state in place: its graph on the canvas, its undo history, saved state and view. */
function showTab(tab) {
  activeTabId = tab.id;
  suppressHistory = true;
  try {
    leaveSubgraphs();
    graph.configure(JSON.parse(tab.graph));
  } catch (err) {
    console.warn("could not open the tab's graph", err);
    leaveSubgraphs();
    graph.clear();
  } finally {
    suppressHistory = false;
  }
  clearTimeout(historyTimer); // configure()'s node removals scheduled a push for the old graph
  syncWidgetsFromProperties();
  clearNodeErrors();
  outputPreviews.clear();
  comparePreviews.clear();
  savedSnapshot = tab.saved;
  currentWorkflowName = tab.name;
  currentTabHint = tab.hint || "";
  undoStack = tab.undoStack || [tab.graph];
  undoIndex = tab.undoIndex ?? undoStack.length - 1;
  if (tab.view) {
    canvas.ds.scale = tab.view.scale;
    canvas.ds.offset[0] = tab.view.offset[0];
    canvas.ds.offset[1] = tab.view.offset[1];
  }
  canvas.setDirty(true, true);
  updateEmptyState();
  saveGraphLocally(JSON.stringify(graph.serialize()));
  renderAppMode();
}

function switchTab(id) {
  if (id === activeTabId) return;
  const tab = tabs.find((t) => t.id === id);
  if (!tab) return;
  stashActiveTab();
  showTab(tab);
}

function newTab() {
  stashActiveTab();
  const tab = { id: Logic.newTabId(tabs), name: "", graph: JSON.stringify({ nodes: [], links: [] }), saved: null };
  tabs.push(tab);
  showTab(tab);
  return tab;
}

async function closeTab(id) {
  const tab = tabs.find((t) => t.id === id);
  if (!tab) return;
  if (id === activeTabId) stashActiveTab();
  if (Logic.tabIsDirty(tab)) {
    const label = Logic.tabLabel(tab, false);
    if (!(await confirmModal("Unsaved changes", `Close "${label}"? Its unsaved changes will be lost.`, "Close"))) return;
  }
  const next = Logic.tabAfterClose(tabs, id);
  tabs = tabs.filter((t) => t.id !== id);
  if (!next) {
    activeTabId = null;
    newTab();
  } else if (id === activeTabId) {
    showTab(tabs.find((t) => t.id === next));
  } else {
    persistTabs();
    renderTabs();
  }
}

function renderTabs() {
  const list = document.getElementById("workflow-tab-list");
  if (!list || !tabs.length) return;
  const items = tabs.map((tab) => {
    const active = tab.id === activeTabId;
    const named = active ? { name: currentWorkflowName, hint: currentTabHint } : tab;
    const dirty = active ? isDirty() : Logic.tabIsDirty(tab);
    const wrap = document.createElement("div");
    wrap.className = `wf-tab${active ? " active" : ""}`;
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    button.dataset.tab = tab.id;
    button.textContent = Logic.tabLabel(named, dirty);
    button.title = Logic.tabLabel(named, false) + (dirty ? " (unsaved changes)" : "");
    button.onclick = () => switchTab(tab.id);
    const close = document.createElement("button");
    close.type = "button";
    close.className = "wf-tab-close";
    close.textContent = "×";
    close.setAttribute("aria-label", `Close ${Logic.tabLabel(named, false)}`);
    close.onclick = () => closeTab(tab.id);
    wrap.append(button, close);
    return wrap;
  });
  list.replaceChildren(...items);
}

function initTabs() {
  document.getElementById("btn-new-tab").onclick = () => newTab();
  // Arrow keys move between tabs (the tablist pattern); Enter/Space switch (they're buttons).
  document.getElementById("workflow-tab-list").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const buttons = [...document.querySelectorAll("#workflow-tab-list [role='tab']")];
    const index = buttons.indexOf(document.activeElement);
    if (index < 0) return;
    e.preventDefault();
    buttons[(index + (e.key === "ArrowRight" ? 1 : buttons.length - 1)) % buttons.length].focus();
  });
}

/** Startup: the stored tabs, or (first run with tabs) the single auto-saved graph as one tab. */
function restoreTabs() {
  let stored = null;
  try {
    stored = Logic.parseStoredTabs(localStorage.getItem(TABS_STORAGE_KEY));
  } catch (_) { /* storage blocked: start with one tab */ }
  if (stored) {
    tabs = stored.tabs;
    showTab(tabs.find((tab) => tab.id === stored.active));
    return;
  }
  restoreSavedGraph();
  tabs = [{ id: "t1", name: "", graph: JSON.stringify(graph.serialize()), saved: null }];
  activeTabId = "t1";
  persistTabs();
  renderTabs();
}

function restoreSavedGraph() {
  let saved;
  try {
    saved = localStorage.getItem(GRAPH_STORAGE_KEY);
  } catch (_) {
    return false;
  }
  if (!saved) return false;
  try {
    suppressHistory = true;
    leaveSubgraphs();
    graph.configure(JSON.parse(saved));
  } catch (err) {
    console.warn("could not restore the auto-saved graph", err);
    leaveSubgraphs();
    graph.clear();
    return false;
  } finally {
    suppressHistory = false;
  }
  syncWidgetsFromProperties();
  return true;
}

function syncWidgetsFromProperties() {
  // graph.configure() restores node.properties from the snapshot but does not
  // reliably refresh each widget's displayed value from it, so widgets can
  // show stale text after undo/redo or loading a workflow. Re-sync explicitly.
  for (const node of allNodes()) {
    if (!node.widgets) continue;
    for (const widget of node.widgets) {
      if (widget.name in node.properties) widget.value = node.properties[widget.name];
      else if (widget.name === "control after generate" && node.properties[SEED_CONTROL_PROPERTY]) {
        widget.value = node.properties[SEED_CONTROL_PROPERTY];
      }
    }
    fitNodeWidth(node); // the restored values can be longer than the constructor's defaults
    fitPromptNode(node);
  }
  canvas.setDirty(true, true);
}

function restoreSnapshot(snap) {
  suppressHistory = true;
  leaveSubgraphs();
  graph.configure(JSON.parse(snap));
  suppressHistory = false;
  syncWidgetsFromProperties();
  updateEmptyState();
  saveGraphLocally(snap);
}

function undo() {
  if (undoIndex <= 0) return;
  undoIndex--;
  restoreSnapshot(undoStack[undoIndex]);
}

function redo() {
  if (undoIndex >= undoStack.length - 1) return;
  undoIndex++;
  restoreSnapshot(undoStack[undoIndex]);
}

// ---- Dialogs (themed, with focus handling: F15) ----

let focusBeforeDialog = null;

function rememberFocus() {
  focusBeforeDialog = document.activeElement;
}

function restoreFocus() {
  if (focusBeforeDialog && typeof focusBeforeDialog.focus === "function") focusBeforeDialog.focus();
  focusBeforeDialog = null;
}

// Keeps Tab inside an open dialog.
function trapFocus(overlay, e) {
  if (e.key !== "Tab") return;
  const focusable = [...overlay.querySelectorAll("button, input, select, textarea, [tabindex]")].filter(
    (el) => !el.disabled && el.offsetParent !== null,
  );
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault();
    first.focus();
  }
}

/**
 * The generic dialog. With `input`, resolves to the trimmed text (or null);
 * without, to "confirm" | "alt" | null.
 */
function openDialog({ title, message = "", input = false, defaultValue = "", confirmText = "OK", altText = "" }) {
  return new Promise((resolve) => {
    const overlay = document.getElementById("modal-overlay");
    const inputEl = document.getElementById("modal-input");
    const messageEl = document.getElementById("modal-message");
    const confirmBtn = document.getElementById("modal-confirm");
    const cancelBtn = document.getElementById("modal-cancel");
    const altBtn = document.getElementById("modal-alt");

    document.getElementById("modal-title").textContent = title;
    messageEl.textContent = message;
    messageEl.classList.toggle("hidden", !message);
    inputEl.classList.toggle("hidden", !input);
    inputEl.value = defaultValue;
    confirmBtn.textContent = confirmText;
    altBtn.textContent = altText;
    altBtn.classList.toggle("hidden", !altText);
    rememberFocus();
    overlay.classList.remove("hidden");
    (input ? inputEl : confirmBtn).focus();
    if (input) inputEl.select();

    function cleanup(result) {
      overlay.classList.add("hidden");
      confirmBtn.removeEventListener("click", onConfirm);
      cancelBtn.removeEventListener("click", onCancel);
      altBtn.removeEventListener("click", onAlt);
      overlay.removeEventListener("keydown", onKeydown);
      restoreFocus();
      resolve(result);
    }
    function onConfirm() {
      cleanup(input ? inputEl.value.trim() || null : "confirm");
    }
    function onCancel() {
      cleanup(null);
    }
    function onAlt() {
      cleanup("alt");
    }
    function onKeydown(e) {
      trapFocus(overlay, e);
      if (e.key === "Enter" && (input ? e.target === inputEl : true)) {
        e.preventDefault();
        onConfirm();
      }
      if (e.key === "Escape") {
        e.stopPropagation();
        onCancel();
      }
    }
    confirmBtn.addEventListener("click", onConfirm);
    cancelBtn.addEventListener("click", onCancel);
    altBtn.addEventListener("click", onAlt);
    overlay.addEventListener("keydown", onKeydown);
  });
}

function promptModal(title, defaultValue = "", { confirmText = "Save" } = {}) {
  return openDialog({ title, input: true, defaultValue, confirmText });
}

async function confirmModal(title, message, confirmText = "OK") {
  return (await openDialog({ title, message, confirmText })) === "confirm";
}

// Replace the graph after asking when there are unsaved changes (F7).
async function replaceGraph(data, question = "Replace the current graph?") {
  if (isDirty() && !(await confirmModal("Unsaved changes", `${question} Your unsaved changes will be lost.`, "Replace"))) {
    return false;
  }
  try {
    suppressHistory = true;
    leaveSubgraphs();
    graph.configure(Logic.withoutSavedPrompt(data));
  } catch (err) {
    showToast(`That workflow could not be loaded: ${err.message}`, "error");
    return false;
  } finally {
    suppressHistory = false;
  }
  syncWidgetsFromProperties();
  clearNodeErrors();
  outputPreviews.clear();
  comparePreviews.clear();
  pushHistory();
  updateEmptyState();
  renderAppMode();
  return true;
}

// ---- Workflows (F7, with B12's delete / rename / overwrite protection) ----

async function fetchWorkflows() {
  const res = await fetch("/api/workflows");
  if (!res.ok) throw new Error((await errorFromResponse(res)).message);
  return res.json();
}

async function saveWorkflow(nameHint) {
  const name = nameHint || (await promptModal("Save Workflow As", currentWorkflowName));
  if (!name) return false;
  // F24: with the runnable prompt, for the MCP server.
  const body = JSON.stringify(Logic.withSavedPrompt(graph.serialize(), buildPrompt()));
  const post = (overwrite) =>
    fetch(`/api/workflows/${encodeURIComponent(name)}?overwrite=${overwrite}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
  let res = await post(false);
  if (res.status === 409) {
    if (!(await confirmModal("Replace workflow?", `A workflow named "${name}" already exists.`, "Replace"))) return false;
    res = await post(true);
  }
  if (!res.ok) {
    showToast(`Workflow not saved: ${(await errorFromResponse(res)).message}`, "error");
    return false;
  }
  markSaved(name);
  showToast(`Saved workflow "${name}"`);
  if (!document.getElementById("workflow-overlay").classList.contains("hidden")) renderWorkflowList();
  return true;
}

async function loadWorkflow(name) {
  const res = await fetch(`/api/workflows/${encodeURIComponent(name)}`);
  if (!res.ok) {
    showToast(`Could not load "${name}": ${(await errorFromResponse(res)).message}`, "error");
    return;
  }
  const data = await res.json();
  // UI6: a workflow already open in another tab is switched to; one opened over a graph that
  // has nodes gets a tab of its own (nothing is replaced, so nothing to confirm).
  const open = tabs.find((tab) => tab.id !== activeTabId && tab.name === name);
  if (open) {
    switchTab(open.id);
    closeOverlay("workflow-overlay");
    return;
  }
  if (graph._nodes.length && currentWorkflowName !== name) newTab();
  if (await replaceGraph(data, `Load "${name}"?`)) {
    markSaved(name);
    closeOverlay("workflow-overlay");
  }
}

async function renameWorkflow(name) {
  const newName = await promptModal(`Rename "${name}"`, name, { confirmText: "Rename" });
  if (!newName || newName === name) return;
  const res = await fetch(`/api/workflows/${encodeURIComponent(name)}/rename`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ new_name: newName }),
  });
  if (!res.ok) {
    showToast(`Not renamed: ${(await errorFromResponse(res)).message}`, "error");
    return;
  }
  if (currentWorkflowName === name) currentWorkflowName = newName;
  for (const tab of tabs) if (tab.name === name) tab.name = newName;
  persistTabs();
  renderTabs();
  renderWorkflowList();
}

async function duplicateWorkflow(name) {
  const res = await fetch(`/api/workflows/${encodeURIComponent(name)}`);
  if (!res.ok) return;
  const data = await res.json();
  const copyName = await promptModal(`Duplicate "${name}" as`, `${name} copy`, { confirmText: "Duplicate" });
  if (!copyName) return;
  const save = await fetch(`/api/workflows/${encodeURIComponent(copyName)}?overwrite=false`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!save.ok) showToast(`Not duplicated: ${(await errorFromResponse(save)).message}`, "error");
  renderWorkflowList();
}

async function deleteWorkflow(name) {
  if (!(await confirmModal("Delete workflow?", `"${name}" will be deleted.`, "Delete"))) return;
  const res = await fetch(`/api/workflows/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) showToast(`Not deleted: ${(await errorFromResponse(res)).message}`, "error");
  renderWorkflowList();
}

async function renderWorkflowList() {
  const list = document.getElementById("workflow-list");
  list.replaceChildren();
  let items;
  try {
    items = await fetchWorkflows();
  } catch (err) {
    showToast(`Could not list workflows: ${err.message}`, "error");
    return;
  }
  if (!items.length) {
    const empty = document.createElement("p");
    empty.textContent = "No saved workflows yet — use Save (Ctrl+S).";
    list.appendChild(empty);
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "workflow-row";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = item.name;
    const when = document.createElement("span");
    when.className = "when";
    when.textContent = new Date(item.modified * 1000).toLocaleString();
    const button = (text, fn, cls = "") => {
      const b = document.createElement("button");
      b.className = `small ${cls}`.trim();
      b.textContent = text;
      b.onclick = fn;
      return b;
    };
    row.append(
      name,
      when,
      button("Load", () => loadWorkflow(item.name)),
      button("Rename", () => renameWorkflow(item.name)),
      button("Duplicate", () => duplicateWorkflow(item.name)),
      button("Delete", () => deleteWorkflow(item.name), "danger"),
    );
    list.appendChild(row);
  }
}

function openOverlay(id) {
  const overlay = document.getElementById(id);
  rememberFocus();
  overlay.classList.remove("hidden");
  const focusTarget = overlay.querySelector(".primary, button");
  if (focusTarget) focusTarget.focus();
}

function closeOverlay(id) {
  const overlay = document.getElementById(id);
  if (overlay.classList.contains("hidden")) return;
  overlay.classList.add("hidden");
  restoreFocus();
}

function openWorkflowManager() {
  openOverlay("workflow-overlay");
  renderWorkflowList();
}

// Import: a workflow .json, or a PNG with an embedded workflow.
async function importFile(file) {
  if (file.type === "image/png" || /\.png$/i.test(file.name)) {
    const workflow = Logic.workflowFromPngChunks(Logic.readPngTextChunks(await file.arrayBuffer()));
    if (!workflow) {
      showToast(`${file.name} has no embedded workflow.`, "error");
      return;
    }
    if (await replaceGraph(workflow, `Load the workflow embedded in ${file.name}?`)) markSaved("", fileTitle(file.name));
    return;
  }
  let data;
  try {
    data = JSON.parse(await file.text());
  } catch (err) {
    showToast(`${file.name} is not valid JSON: ${err.message}`, "error");
    return;
  }
  if (!data || !Array.isArray(data.nodes)) {
    showToast(`${file.name} is not a PixelFlow / litegraph workflow.`, "error");
    return;
  }
  if (await replaceGraph(data, `Import ${file.name}?`)) markSaved("", fileTitle(file.name));
}

/** A file's name without its extension, as a tab label ("portrait.png" -> "portrait"). */
function fileTitle(name) {
  return name.replace(/\.[^.]+$/, "");
}

function findInputSlot(node, name) {
  return node.inputs.findIndex((i) => i.name === name);
}

function setNodeValue(node, name, value) {
  node.properties[name] = value;
  const widgetName = name === SEED_CONTROL_PROPERTY ? "control after generate" : name;
  const widget = (node.widgets || []).find((w) => w.name === widgetName);
  if (widget) widget.value = value;
  fitNodeWidth(node);
}

// ---- Mask editor (F18): paint a mask over a Load Image's picture ----

const MASK_PAINT = "rgba(255, 48, 48, 1)"; // shown at the canvas's CSS opacity
const MASK_NODE_PROPERTY = "_mask_node"; // Load Image -> the Load Mask node it feeds
let maskTarget = null; // the Load Image node being edited

function maskCanvasPoint(e) {
  const el = document.getElementById("mask-canvas");
  const rect = el.getBoundingClientRect();
  return [(e.clientX - rect.left) * (el.width / rect.width), (e.clientY - rect.top) * (el.height / rect.height)];
}

function linkedMaskNode(imageNode) {
  const node = graph.getNodeById(imageNode.properties[MASK_NODE_PROPERTY]);
  return node && node.comfyClass === "LoadMask" ? node : null;
}

async function openMaskEditor(imageNode) {
  const filename = imageNode.properties[imageNode.uploadComboName];
  if (!filename) {
    showToast("Upload or choose an image in Load Image first.");
    return;
  }
  maskTarget = imageNode;
  const img = document.getElementById("mask-image");
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error("the image couldn't be loaded"));
    img.src = `/uploads/${encodeURIComponent(filename)}`;
  }).catch((err) => showToast(`Mask editor: ${err.message}`, "error"));
  if (!img.naturalWidth) return;
  openOverlay("mask-overlay");
  const el = document.getElementById("mask-canvas");
  el.width = img.naturalWidth;
  el.height = img.naturalHeight;
  layoutMaskCanvas();
  const ctx = el.getContext("2d");
  ctx.clearRect(0, 0, el.width, el.height);
  // Re-opening shows the mask saved last time (white = painted).
  const maskNode = linkedMaskNode(imageNode);
  const maskFile = maskNode && maskNode.properties.filename;
  if (maskFile) {
    const previous = new Image();
    await new Promise((resolve) => {
      previous.onload = resolve;
      previous.onerror = resolve;
      previous.src = `/uploads/${encodeURIComponent(maskFile)}`;
    });
    if (previous.naturalWidth) {
      ctx.drawImage(previous, 0, 0, el.width, el.height);
      const data = ctx.getImageData(0, 0, el.width, el.height);
      for (let i = 0; i < data.data.length; i += 4) {
        const luminance = data.data[i];
        data.data[i] = 255;
        data.data[i + 1] = 48;
        data.data[i + 2] = 48;
        data.data[i + 3] = luminance;
      }
      ctx.putImageData(data, 0, 0);
    }
  }
}

function layoutMaskCanvas() {
  const img = document.getElementById("mask-image");
  const el = document.getElementById("mask-canvas");
  el.style.left = `${img.offsetLeft}px`;
  el.style.top = `${img.offsetTop}px`;
  el.style.width = `${img.clientWidth}px`;
  el.style.height = `${img.clientHeight}px`;
}

function initMaskEditor() {
  const el = document.getElementById("mask-canvas");
  const brush = document.getElementById("mask-brush");
  let last = null;
  const stroke = (from, to) => {
    const ctx = el.getContext("2d");
    const scale = el.width / el.getBoundingClientRect().width;
    const erase = document.querySelector('input[name="mask-mode"]:checked').value === "erase";
    ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
    ctx.strokeStyle = MASK_PAINT;
    ctx.lineWidth = Number(brush.value) * scale;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(from[0], from[1]);
    ctx.lineTo(to[0], to[1]);
    ctx.stroke();
  };
  el.addEventListener("pointerdown", (e) => {
    el.setPointerCapture(e.pointerId);
    last = maskCanvasPoint(e);
    stroke(last, last);
  });
  el.addEventListener("pointermove", (e) => {
    if (!last) return;
    const point = maskCanvasPoint(e);
    stroke(last, point);
    last = point;
  });
  const end = () => {
    last = null;
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  document.getElementById("mask-overlay").addEventListener("keydown", (e) => {
    if (e.key === "[" || e.key === "]") {
      brush.value = String(Number(brush.value) + (e.key === "]" ? 1 : -1) * (e.shiftKey ? 20 : 5));
      e.preventDefault();
    }
  });
  window.addEventListener("resize", () => {
    if (!document.getElementById("mask-overlay").classList.contains("hidden")) layoutMaskCanvas();
  });
  document.getElementById("mask-clear").onclick = () => el.getContext("2d").clearRect(0, 0, el.width, el.height);
  document.getElementById("mask-cancel").onclick = () => closeOverlay("mask-overlay");
  document.getElementById("mask-save").onclick = saveMask;
}

/** The painted strokes as a white-on-black PNG at the image's own size. */
function maskPngBlob() {
  const el = document.getElementById("mask-canvas");
  const out = document.createElement("canvas");
  out.width = el.width;
  out.height = el.height;
  const ctx = out.getContext("2d");
  ctx.drawImage(el, 0, 0);
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.globalCompositeOperation = "destination-over";
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, out.width, out.height);
  return new Promise((resolve) => out.toBlob(resolve, "image/png"));
}

async function saveMask() {
  const imageNode = maskTarget;
  if (!imageNode) return;
  let filename;
  try {
    const stem = String(imageNode.properties[imageNode.uploadComboName]).replace(/\.[^.]+$/, "");
    filename = await uploadImageFile(new File([await maskPngBlob()], `${stem}_mask.png`, { type: "image/png" }));
  } catch (err) {
    showToast(`Saving the mask failed: ${err.message}`, "error");
    return;
  }
  let maskNode = linkedMaskNode(imageNode);
  if (!maskNode) {
    maskNode = addNodeToGraph("LoadMask", [imageNode.pos[0], imageNode.pos[1] + imageNode.size[1] + 40]);
    imageNode.properties[MASK_NODE_PROPERTY] = maskNode.id;
  }
  setNodeValue(maskNode, "filename", filename);
  setNodeValue(maskNode, "channel", "luminance");
  setNodeValue(maskNode, "invert", false);
  maskNode.previewImage = null;
  // Wire it into the inpaint sampler(s) this image feeds, unless one already takes another mask.
  for (const link of Object.values(graph.links)) {
    if (link.origin_id !== imageNode.id || link.origin_slot !== 0) continue;
    const sampler = graph.getNodeById(link.target_id);
    if (!sampler || sampler.comfyClass !== "KSamplerInpaint") continue;
    const slot = findInputSlot(sampler, "mask");
    const current = sampler.inputs[slot].link != null ? graph.links[sampler.inputs[slot].link] : null;
    if (!current || current.origin_id === imageNode.id || current.origin_id === maskNode.id) maskNode.connect(0, sampler, slot);
  }
  closeOverlay("mask-overlay");
  canvas.setDirty(true, true);
  scheduleHistoryPush();
  showToast(`Mask saved as ${filename}`);
}

// ---- X/Y plot (F18): a grid over one or two widget values -> a contact sheet ----

/** Resolves with a prompt's outputs once it finishes (null if it fails) -- also
 * when it already finished before this was called: a fast run's WebSocket
 * message can beat the queue request's own response. */
function waitForPrompt(promptId) {
  const entry = queueState.entries.get(promptId);
  if (entry && entry.status === "completed") return Promise.resolve(entry.outputs || {});
  if (entry && ["error", "cancelled", "interrupted"].includes(entry.status)) return Promise.resolve(null);
  return new Promise((resolve) => xyWaiters.set(promptId, resolve));
}

function xyCandidateNodes() {
  return graph._nodes.filter((n) => n.inputSpecs && n.mode === LiteGraph.ALWAYS && xyWidgets(n).length);
}

function xyWidgets(node) {
  const converted = new Set(node.properties[CONVERTED_PROPERTY] || []);
  return Object.entries(node.inputSpecs || {})
    .filter(([name, spec]) => !converted.has(name) && (PRIMITIVE_WIDGET_TYPES.includes(spec[0]) || Array.isArray(spec[0])))
    .map(([name]) => name);
}

function fillXyWidgetSelect(axis) {
  const node = graph.getNodeById(Number(document.getElementById(`xy-${axis}-node`).value));
  const select = document.getElementById(`xy-${axis}-widget`);
  select.textContent = "";
  for (const name of node ? xyWidgets(node) : []) select.add(new Option(name, name));
  select.disabled = !node;
  document.getElementById(`xy-${axis}-values`).disabled = !node;
  if (node) {
    const preferred = { x: ["cfg", "steps", "denoise"], y: ["steps", "seed", "sampler"] }[axis].find((n) => xyWidgets(node).includes(n));
    if (preferred) select.value = preferred;
  }
}

function openXyPlot() {
  const nodes = xyCandidateNodes();
  if (!nodes.length) {
    showToast("Build a graph first: the X/Y plot varies widgets of its nodes.");
    return;
  }
  for (const axis of ["x", "y"]) {
    const select = document.getElementById(`xy-${axis}-node`);
    select.textContent = "";
    if (axis === "y") select.add(new Option("(none)", ""));
    for (const node of nodes) select.add(new Option(`${node.title} #${node.id}`, String(node.id)));
    const sampler = nodes.find((n) => /^KSampler/.test(n.comfyClass));
    if (sampler) select.value = String(sampler.id);
    if (axis === "y") select.value = "";
    select.onchange = () => fillXyWidgetSelect(axis);
    fillXyWidgetSelect(axis);
  }
  document.getElementById("xy-status").textContent = "";
  openOverlay("xy-overlay");
}

function xyAxis(axis) {
  const node = graph.getNodeById(Number(document.getElementById(`xy-${axis}-node`).value));
  if (!node) return null;
  const name = document.getElementById(`xy-${axis}-widget`).value;
  const spec = node.inputSpecs[name];
  let values;
  try {
    values = Logic.parseAxisValues(document.getElementById(`xy-${axis}-values`).value, Array.isArray(spec[0]) ? "COMBO" : spec[0]);
  } catch (err) {
    throw new Error(`${axis.toUpperCase()} values: ${err.message}`, { cause: err });
  }
  if (Array.isArray(spec[0])) {
    const unknown = values.filter((v) => !spec[0].includes(v));
    if (unknown.length) throw new Error(`${axis.toUpperCase()}: ${name} has no option ${unknown.join(", ")}`);
  }
  return { node, name, values };
}

async function runXyPlot() {
  const status = document.getElementById("xy-status");
  let x, y;
  try {
    x = xyAxis("x");
    y = xyAxis("y");
    if (!x) throw new Error("choose the X axis node");
  } catch (err) {
    status.textContent = err.message;
    return;
  }
  const yValues = y ? y.values : [null];
  const total = x.values.length * yValues.length;
  if (total > Logic.XY_MAX_CELLS) {
    status.textContent = `${total} cells is too many (at most ${Logic.XY_MAX_CELLS}).`;
    return;
  }
  const button = document.getElementById("xy-run");
  button.disabled = true;
  // Every cell uses the same seeds, so only the X and Y values differ.
  const saved = new Map();
  const remember = (node, name) => {
    const key = `${node.id}:${name}`;
    if (!saved.has(key)) saved.set(key, [node, name, node.properties[name]]);
  };
  for (const node of seedNodes()) {
    remember(node, node.seedInputName);
    setNodeValue(node, node.seedInputName, Logic.seedForQueue(node.properties[node.seedInputName]));
  }
  remember(x.node, x.name);
  if (y) remember(y.node, y.name);
  const waits = [];
  try {
    for (const yValue of yValues) {
      for (const xValue of x.values) {
        setNodeValue(x.node, x.name, xValue);
        if (y) setNodeValue(y.node, y.name, yValue);
        status.textContent = `Queueing ${waits.length + 1} of ${total}…`;
        const promptId = await submitPrompt(false);
        if (!promptId) throw new Error("a prompt couldn't be queued (see the message)");
        waits.push(waitForPrompt(promptId));
      }
    }
  } catch (err) {
    status.textContent = err.message;
    button.disabled = false;
    return;
  } finally {
    for (const [node, name, value] of saved.values()) setNodeValue(node, name, value);
  }
  let done = 0;
  const outputs = await Promise.all(waits.map((wait) => wait.then((result) => {
    done += 1;
    status.textContent = `Running: ${done} of ${total} done…`;
    return result;
  })));
  const cells = outputs.map((result) => Logic.firstImage(result));
  if (!cells.some(Boolean)) {
    status.textContent = "No run produced an image (add a Save Image or Preview Image).";
    button.disabled = false;
    return;
  }
  status.textContent = "Composing the contact sheet…";
  try {
    const res = await fetch("/api/contact_sheet", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cells,
        columns: x.values.length,
        x_values: x.values.map(String),
        y_values: yValues.map((v) => (v === null ? "" : String(v))),
        x_label: `${x.node.title}: ${x.name}`,
        y_label: y ? `${y.node.title}: ${y.name}` : "",
      }),
    });
    if (!res.ok) throw new Error((await errorFromResponse(res)).message);
    const sheet = await res.json();
    addGalleryItems([{ filename: sheet.filename, type: sheet.type, promptId: null, meta: {} }]);
    status.textContent = `Saved ${sheet.filename}`;
    closeOverlay("xy-overlay");
    openLightbox(galleryItems.findIndex((item) => item.filename === sheet.filename));
  } catch (err) {
    status.textContent = `Couldn't compose the sheet: ${err.message}`;
  } finally {
    button.disabled = false;
  }
}

// ---- Templates (F18): read-only starter graphs under frontend/templates/ ----

// A template is {nodes: [{id, type, pos, values}], links: [[from id, output slot,
// to id, input name]]} -- node ids are the template's own names, so the files stay
// readable and independent of litegraph's serialization.
// With `insert` (F28), the template is added beside the current graph -- a
// reusable block -- and selected, instead of replacing the graph.
function buildTemplateGraph(spec, insert = false, title = "") {
  let offset = [0, 0];
  if (insert && graph._nodes.length) {
    const right = Math.max(...graph._nodes.map((n) => n.pos[0] + n.size[0]));
    const top = Math.min(...graph._nodes.map((n) => n.pos[1]));
    const left = Math.min(...spec.nodes.map((entry) => entry.pos[0]));
    const specTop = Math.min(...spec.nodes.map((entry) => entry.pos[1]));
    offset = [right + 80 - left, top - specTop];
  } else {
    leaveSubgraphs();
    graph.clear();
    outputPreviews.clear();
    comparePreviews.clear();
  }
  const nodes = new Map();
  for (const entry of spec.nodes) {
    const node = addNodeToGraph(entry.type, [entry.pos[0] + offset[0], entry.pos[1] + offset[1]]);
    if (!node) throw new Error(`unknown node type "${entry.type}"`);
    for (const [name, value] of Object.entries(entry.values || {})) setNodeValue(node, name, value);
    nodes.set(entry.id, node);
  }
  for (const [fromId, slot, toId, inputName] of spec.links) {
    const target = nodes.get(toId);
    const input = findInputSlot(target, inputName);
    if (input < 0) throw new Error(`${toId} has no input "${inputName}"`);
    nodes.get(fromId).connect(slot, target, input);
  }
  canvas.setDirty(true, true);
  pushHistory();
  if (insert) {
    canvas.selectNodes([...nodes.values()]);
  } else {
    markSaved("", title);
  }
  updateEmptyState();
  fitView();
}

async function loadTemplate(file, title = file, insert = false) {
  if (!insert && isDirty() && !(await confirmModal("Unsaved changes", `Load the "${title}" template? Your unsaved changes will be lost.`, "Load template"))) {
    return false;
  }
  try {
    const res = await fetch(`templates/${file}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    buildTemplateGraph(await res.json(), insert, title);
  } catch (err) {
    showToast(`Couldn't load the template: ${err.message}`, "error");
    return false;
  }
  return true;
}

// The default checkpoint (stabilityai/sd-turbo) is a distilled few-step model
// trained without classifier-free guidance, so the starter graph uses 4 steps at
// cfg 0 (KSampler's 20-step / cfg 7 defaults are for regular SD checkpoints).
function loadExampleGraph() {
  return loadTemplate("txt2img.json", "Text to image");
}

/** A template's graph as a small diagram (UI7): boxes coloured by node category, links as lines. */
async function drawTemplateThumb(thumb, file) {
  let spec;
  try {
    const res = await fetch(`templates/${file}`);
    if (!res.ok) return;
    spec = await res.json();
  } catch (_) {
    return;
  }
  const ctx = thumb.getContext("2d");
  const { boxes, lines } = Logic.templateDiagram(spec, thumb.width, thumb.height);
  const style = getComputedStyle(document.documentElement);
  const colours = { loaders: style.getPropertyValue("--accent").trim() || "#5b8def", sampling: "#ffa931" };
  ctx.strokeStyle = "#8a8b93";
  ctx.lineWidth = 1;
  for (const [x1, y1, x2, y2] of lines) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo((x1 + x2) / 2, y1, (x1 + x2) / 2, y2, x2, y2);
    ctx.stroke();
  }
  for (const box of boxes) {
    const category = (nodeDefs[box.type] || {}).category;
    ctx.fillStyle = colours[category] || "#4a4b55";
    ctx.fillRect(box.x, box.y, Math.max(2, box.w), Math.max(2, box.h));
  }
}

async function openTemplates() {
  openOverlay("template-overlay");
  const list = document.getElementById("template-list");
  list.textContent = "Loading…";
  try {
    const res = await fetch("templates/index.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { templates } = await res.json();
    list.textContent = "";
    for (const template of templates) {
      const button = document.createElement("button");
      button.className = "template-item";
      button.dataset.file = template.file;
      const title = document.createElement("strong");
      title.textContent = template.title;
      const description = document.createElement("span");
      description.textContent = template.description;
      const text = document.createElement("div");
      text.className = "template-text";
      text.append(title, description);
      const thumb = document.createElement("canvas");
      thumb.className = "template-thumb";
      thumb.width = 120;
      thumb.height = 64;
      thumb.setAttribute("aria-hidden", "true");
      button.append(thumb, text);
      drawTemplateThumb(thumb, template.file);
      button.onclick = async () => {
        closeOverlay("template-overlay");
        await loadTemplate(template.file, template.title, document.getElementById("template-insert").checked);
      };
      list.append(button);
    }
    const first = list.querySelector("button");
    if (first) first.focus();
  } catch (err) {
    list.textContent = `Couldn't list the templates: ${err.message}`;
  }
}

// ---- System panel (B18; R2's "Open log folder") ----

async function refreshSystemStatus() {
  if (document.hidden) return;
  try {
    const res = await fetch("/api/system");
    if (!res.ok) return;
    const info = await res.json();
    const button = document.getElementById("system-status");
    if (info.device === "cuda") {
      const used = info.vram_total - info.vram_free;
      button.textContent = `${info.device_name} · VRAM ${Logic.formatBytes(used)} / ${Logic.formatBytes(info.vram_total)}`;
    } else {
      button.textContent = Logic.cpuDeviceLabel(info);
    }
    button.title = "System: GPU, memory, logs";
    return info;
  } catch (_) {
    return null;
  }
}

async function openSystemPanel() {
  openOverlay("system-overlay");
  const info = await refreshSystemStatus();
  if (!info) return;
  const dl = document.getElementById("system-info");
  dl.replaceChildren();
  const rows = [
    ["PixelFlow", info.version],
    ["Device", info.device === "cuda" ? info.device_name : "CPU"],
    ["VRAM free", info.vram_total ? `${Logic.formatBytes(info.vram_free)} of ${Logic.formatBytes(info.vram_total)}` : "—"],
    ["torch", `${info.torch} (CUDA ${info.cuda || "none"})`],
    ["diffusers", info.diffusers || "?"],
    ["Python", info.python],
    ["Data folder", info.data_dir],
    ["Models folder", info.models_dir],
    ["Logs", info.log_dir],
    ["Custom nodes", Logic.describeCustomNodes(info.custom_nodes)],
  ];
  for (const [key, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    dl.append(dt, dd);
  }
  const list = document.getElementById("system-models");
  list.replaceChildren();
  for (const model of info.loaded_models || []) {
    const li = document.createElement("li");
    li.textContent = `${model.name} — ${model.arch}, ${model.dtype}, ${model.placement} (${Logic.formatBytes(model.bytes)})`;
    list.appendChild(li);
  }
  if (!list.children.length) {
    const li = document.createElement("li");
    li.textContent = "None loaded";
    list.appendChild(li);
  }
  refreshDownloads();
  refreshMcpConfig();
}

// ---- Connect an assistant (F27) ----

let mcpConfig = null;

async function refreshMcpConfig() {
  try {
    const res = await fetch("/api/mcp");
    if (!res.ok) return;
    mcpConfig = await res.json();
  } catch (_) {
    return;
  }
  document.getElementById("mcp-config").textContent = JSON.stringify(mcpConfig.json, null, 2);
  document.getElementById("mcp-claude-code").textContent = mcpConfig.claude_code;
}

async function copyText(text, what) {
  try {
    await navigator.clipboard.writeText(text);
    showToast(`${what} copied`);
  } catch (_) {
    showToast(`Could not copy the ${what.toLowerCase()}: select it and press Ctrl+C`, "error");
  }
}

// ---- Model downloads (F23) ----

let downloadPoll = null;

async function refreshDownloads() {
  clearTimeout(downloadPoll);
  let jobs = [];
  try {
    const res = await fetch("/api/downloads");
    if (res.ok) jobs = await res.json();
  } catch (_) {
    return;
  }
  const list = document.getElementById("download-list");
  list.replaceChildren();
  for (const job of jobs.slice().reverse()) {
    const li = document.createElement("li");
    li.className = "download-list-item";
    li.textContent = Logic.describeDownload(job);
    if (job.help_url) {
      // F26: a gated model -- its page, to log in and accept the licence.
      const link = document.createElement("a");
      link.href = job.help_url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "Open the model page";
      li.append(" ", link);
    }
    if (job.status === "resolving" || job.status === "downloading") {
      const cancel = document.createElement("button");
      cancel.textContent = "Cancel";
      cancel.onclick = async () => {
        await fetch(`/api/downloads/${encodeURIComponent(job.id)}`, { method: "DELETE" });
        refreshDownloads();
      };
      li.append(cancel);
    }
    list.append(li);
  }
  const active = jobs.some((job) => job.status === "resolving" || job.status === "downloading");
  const open = !document.getElementById("system-overlay").classList.contains("hidden");
  if (active && open) downloadPoll = setTimeout(refreshDownloads, 1000);
}

async function startDownload(e) {
  e.preventDefault();
  const url = document.getElementById("download-url");
  const res = await fetch("/api/downloads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: url.value.trim(), kind: document.getElementById("download-kind").value }),
  });
  if (!res.ok) {
    showToast(`Download not started: ${(await errorFromResponse(res)).message}`, "error");
    return;
  }
  url.value = "";
  refreshDownloads();
}

// ---- Wiring up UI ----

function handleShortcut(e) {
  for (const id of ["modal-overlay", "workflow-overlay", "template-overlay", "mask-overlay", "xy-overlay", "system-overlay", "help-overlay"]) {
    const overlay = document.getElementById(id);
    if (!overlay.classList.contains("hidden")) {
      if (id !== "modal-overlay") trapFocus(overlay, e);
      if (e.key === "Escape" && id !== "modal-overlay") closeOverlay(id);
      return; // dialogs handle their own keys
    }
  }
  const lightboxOpen = !document.getElementById("lightbox-overlay").classList.contains("hidden");
  if (lightboxOpen) {
    if (e.key === "Escape") closeLightbox();
    else if (e.key === "ArrowLeft") openLightbox(Logic.stepIndex(lightboxIndex, -1, galleryItems.length));
    else if (e.key === "ArrowRight") openLightbox(Logic.stepIndex(lightboxIndex, 1, galleryItems.length));
    else trapFocus(document.getElementById("lightbox-overlay"), e);
    return;
  }
  // Typing in a text field (node search, a dialog, litegraph's own prompt
  // box) keeps the browser's own Ctrl+Z etc. (F6).
  if (Logic.isTypingTarget(e.target)) return;
  const action = Logic.shortcutFor(e, loadShortcutOverrides());
  if (!action) return;
  const actions = {
    queue: () => queuePrompt({ count: queueCount() }),
    "queue-front": () => queuePrompt({ front: true, count: 1 }),
    undo,
    redo,
    save: () => saveWorkflow(currentWorkflowName || undefined),
    open: openWorkflowManager,
    mute: () => toggleMode(selectedNodes(), Logic.MODE_MUTE),
    bypass: () => toggleMode(selectedNodes(), Logic.MODE_BYPASS),
    group: groupSelected,
    pin: () => toggleNodeFlag(selectedNodes(), "pinned"),
    collapse: () => toggleNodeFlag(selectedNodes(), "collapsed"),
    fit: fitView,
    "toggle-library": () => togglePanel("sidebar"),
    "toggle-queue": () => togglePanel("rightpanel"),
    settings: openSystemPanel,
    help: openShortcutHelp,
    escape: () => { if ((canvas._graph_stack || []).length) canvas.closeSubgraph(); },
  };
  e.preventDefault();
  actions[action]();
}

// Pin (P) and collapse (Alt+C), UI14: all on when any selected node is off, else all off, as
// toggleMode does for mute and bypass.
function toggleNodeFlag(nodes, flag) {
  if (!nodes.length) return;
  const on = !nodes.every((n) => n.flags && n.flags[flag]);
  for (const node of nodes) {
    if (!!(node.flags && node.flags[flag]) === on) continue;
    if (flag === "pinned") node.pin(on);
    else node.collapse(true);
  }
  canvas.setDirty(true, true);
  scheduleHistoryPush();
}

// ---- Rebindable shortcuts (UI14): Logic.SHORTCUTS plus the user's keys, kept per browser ----

const SHORTCUT_STORAGE_KEY = "pixelflow.shortcuts";

function loadShortcutOverrides() {
  try {
    const stored = JSON.parse(localStorage.getItem(SHORTCUT_STORAGE_KEY) || "{}");
    return stored && typeof stored === "object" ? stored : {};
  } catch {
    return {};
  }
}

function saveShortcutOverrides(overrides) {
  try {
    if (Object.keys(overrides).length) localStorage.setItem(SHORTCUT_STORAGE_KEY, JSON.stringify(overrides));
    else localStorage.removeItem(SHORTCUT_STORAGE_KEY);
  } catch {
    // Storage blocked (private window): the new key works until the page is reloaded.
  }
}

// A table cell showing combos as keys: "Ctrl+Shift+Enter" -> Ctrl + Shift + Enter, each a <kbd>.
function comboCell(combos) {
  const cell = document.createElement("td");
  if (!combos.length) {
    cell.append(Object.assign(document.createElement("span"), { className: "hint", textContent: "none" }));
  }
  combos.forEach((combo, i) => {
    if (i) cell.append(" / ");
    combo.split(/\+(?=.)/).forEach((part, j) => {
      if (j) cell.append("+");
      cell.append(Object.assign(document.createElement("kbd"), { textContent: part }));
    });
  });
  return cell;
}

function renderShortcutTable() {
  const keys = Logic.shortcutKeys(loadShortcutOverrides());
  const table = document.getElementById("shortcut-table");
  table.replaceChildren();
  const addRow = (combos, label, action) => {
    const row = table.insertRow();
    row.append(comboCell(combos), Object.assign(document.createElement("td"), { textContent: label }));
    const last = row.insertCell();
    if (!action) return;
    const button = Object.assign(document.createElement("button"), { className: "shortcut-change", textContent: "Change" });
    button.dataset.action = action;
    button.setAttribute("aria-label", `Change the key for: ${label}`);
    button.onclick = () => captureShortcut(button);
    last.append(button);
  };
  for (const { action, label } of Logic.SHORTCUTS) addRow(keys[action], label, action);
  addRow(["←", "→"], "Previous / next image in the viewer");
  addRow(["Esc"], "Close a dialog or the image viewer");
}

// The next key pressed becomes the action's only key (taken from any action that had it);
// Esc cancels, Backspace or Delete leaves the action without a key.
function captureShortcut(button) {
  const overlay = document.getElementById("help-overlay");
  const message = document.getElementById("shortcut-message");
  const action = button.dataset.action;
  button.textContent = "Press a key…";
  message.textContent = "Press the new key (Esc cancels, Backspace removes the key).";
  function onKey(e) {
    const combo = Logic.keyCombo(e);
    if (!combo) return; // a modifier on its own: wait for the rest
    e.preventDefault();
    e.stopPropagation();
    overlay.removeEventListener("keydown", onKey, true);
    if (combo === "Escape") {
      message.textContent = "";
    } else {
      const cleared = combo === "Backspace" || combo === "Delete";
      const { overrides, takenFrom } = Logic.rebindShortcut(loadShortcutOverrides(), action, cleared ? null : combo);
      saveShortcutOverrides(overrides);
      const labels = Object.fromEntries(Logic.SHORTCUTS.map((s) => [s.action, s.label]));
      message.textContent = takenFrom.length
        ? `${combo} no longer does: ${takenFrom.map((a) => labels[a]).join("; ")}.`
        : cleared ? "Key removed." : `${combo} set.`;
    }
    renderShortcutTable();
    const again = overlay.querySelector(`.shortcut-change[data-action="${action}"]`);
    if (again) again.focus();
  }
  overlay.addEventListener("keydown", onKey, true);
}

function openShortcutHelp() {
  document.getElementById("shortcut-message").textContent = "";
  renderShortcutTable();
  openOverlay("help-overlay");
}

function queueCount() {
  const value = parseInt(document.getElementById("queue-count").value, 10);
  return Number.isFinite(value) ? Math.max(1, Math.min(100, value)) : 1;
}

// ---- Side panels (F17): resizable and collapsible, remembered per browser ----

const PANEL_STORAGE_KEY = "pixelflow.panels";

function loadPanelState() {
  try {
    return JSON.parse(localStorage.getItem(PANEL_STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function savePanelState(state) {
  try {
    localStorage.setItem(PANEL_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Storage blocked (private window): the layout just isn't remembered.
  }
}

function applyPanel(id, { width, collapsed }) {
  const panel = document.getElementById(id);
  const splitter = document.querySelector(`.splitter[data-panel="${id}"]`);
  if (width) panel.style.width = `${Logic.clampPanelWidth(width, window.innerWidth)}px`;
  panel.classList.toggle("collapsed", !!collapsed);
  const toggle = splitter.querySelector(".splitter-toggle");
  const opensLeft = id === "sidebar";
  toggle.textContent = collapsed === opensLeft ? "›" : "‹";
  splitter.setAttribute("aria-valuenow", collapsed ? 0 : Math.round(panel.getBoundingClientRect().width));
  resizeCanvas();
}

function setPanel(id, changes) {
  const state = loadPanelState();
  state[id] = { ...(state[id] || {}), ...changes };
  applyPanel(id, state[id]);
  savePanelState(state);
}

function togglePanel(id) {
  setPanel(id, { collapsed: !document.getElementById(id).classList.contains("collapsed") });
}

function initPanels() {
  const state = loadPanelState();
  for (const splitter of document.querySelectorAll(".splitter")) {
    const id = splitter.dataset.panel;
    const panel = document.getElementById(id);
    const sign = id === "sidebar" ? 1 : -1; // dragging right widens the left panel, narrows the right one
    if (state[id]) applyPanel(id, state[id]);
    splitter.querySelector(".splitter-toggle").addEventListener("click", (e) => {
      e.stopPropagation();
      togglePanel(id);
    });
    splitter.addEventListener("dblclick", () => togglePanel(id));
    splitter.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        togglePanel(id);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        const step = (e.key === "ArrowRight" ? 20 : -20) * sign;
        setPanel(id, { collapsed: false, width: panel.getBoundingClientRect().width + step });
      }
    });
    splitter.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".splitter-toggle") || panel.classList.contains("collapsed")) return;
      const startX = e.clientX;
      const startWidth = panel.getBoundingClientRect().width;
      splitter.setPointerCapture(e.pointerId);
      splitter.classList.add("dragging");
      const move = (ev) => applyPanel(id, { width: startWidth + (ev.clientX - startX) * sign });
      const up = (ev) => {
        splitter.removeEventListener("pointermove", move);
        splitter.removeEventListener("pointerup", up);
        splitter.classList.remove("dragging");
        setPanel(id, { width: startWidth + (ev.clientX - startX) * sign });
      };
      splitter.addEventListener("pointermove", move);
      splitter.addEventListener("pointerup", up);
    });
  }
}

function initUI() {
  initPanels();
  initMenus();
  initAppMode();
  initMaskEditor();
  document.getElementById("btn-xy").onclick = openXyPlot;
  document.getElementById("xy-run").onclick = runXyPlot;
  document.getElementById("xy-close").onclick = () => closeOverlay("xy-overlay");
  document.getElementById("btn-queue").onclick = (e) => queuePrompt({ front: e.shiftKey, count: queueCount() });
  document.getElementById("btn-example").onclick = loadExampleGraph;
  document.getElementById("empty-load-example").onclick = loadExampleGraph;
  document.getElementById("btn-templates").onclick = openTemplates;
  document.getElementById("template-close").onclick = () => closeOverlay("template-overlay");
  document.getElementById("btn-clear").onclick = async () => {
    if (isDirty() && !(await confirmModal("Clear the graph?", "Your unsaved changes will be lost.", "Clear"))) return;
    leaveSubgraphs();
    graph.clear();
    outputPreviews.clear();
    comparePreviews.clear();
    pushHistory();
    markSaved("");
    updateEmptyState();
  };
  document.getElementById("btn-undo").onclick = undo;
  document.getElementById("btn-redo").onclick = redo;
  document.getElementById("btn-fit-view").onclick = fitView;
  document.getElementById("btn-help").onclick = openShortcutHelp;
  document.getElementById("shortcut-reset").onclick = () => {
    saveShortcutOverrides({});
    renderShortcutTable();
    document.getElementById("shortcut-message").textContent = "All keys are back to their defaults.";
  };
  document.getElementById("help-close").onclick = () => closeOverlay("help-overlay");
  const allWindows = document.getElementById("queue-all-windows");
  allWindows.checked = showAllWindows;
  allWindows.onchange = () => setShowAllWindows(allWindows.checked);
  document.getElementById("btn-cancel-all").onclick = cancelAllPending;
  document.getElementById("btn-clear-finished").onclick = clearFinished;

  document.getElementById("node-search").addEventListener("input", (e) => filterNodeList(e.target.value));

  // Image viewer
  document.getElementById("lightbox-overlay").addEventListener("click", (e) => {
    if (e.target.id === "lightbox-overlay") closeLightbox();
  });
  document.getElementById("lightbox-close").onclick = closeLightbox;
  document.getElementById("lightbox-prev").onclick = () => openLightbox(Logic.stepIndex(lightboxIndex, -1, galleryItems.length));
  document.getElementById("lightbox-next").onclick = () => openLightbox(Logic.stepIndex(lightboxIndex, 1, galleryItems.length));
  document.getElementById("lightbox-load-workflow").onclick = lightboxLoadWorkflow;
  document.getElementById("lightbox-reuse-seed").onclick = lightboxReuseSeed;
  document.getElementById("lightbox-to-upload").onclick = lightboxSendToLoadImage;
  document.getElementById("lightbox-delete").onclick = lightboxDelete;

  // Workflows
  document.getElementById("btn-save-workflow").onclick = () => saveWorkflow();
  document.getElementById("btn-workflows").onclick = openWorkflowManager;
  document.getElementById("workflow-close").onclick = () => closeOverlay("workflow-overlay");
  document.getElementById("btn-export").onclick = () => {
    const blob = new Blob([JSON.stringify(graph.serialize(), null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${currentWorkflowName || "workflow"}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  // One accessible control: the button opens the (hidden) file input's dialog.
  document.getElementById("btn-import").onclick = () => document.getElementById("file-import").click();
  document.getElementById("file-import").onchange = async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (file) await importFile(file);
  };

  // System
  document.getElementById("system-status").onclick = openSystemPanel;
  document.getElementById("system-close").onclick = () => closeOverlay("system-overlay");
  document.getElementById("download-form").onsubmit = startDownload;
  document.getElementById("btn-copy-mcp").onclick = () =>
    mcpConfig && copyText(JSON.stringify(mcpConfig.json, null, 2), "MCP configuration");
  document.getElementById("btn-copy-mcp-cli").onclick = () =>
    mcpConfig && copyText(mcpConfig.claude_code, "Claude Code command");
  document.getElementById("btn-free-memory").onclick = async () => {
    const res = await fetch("/api/free", { method: "POST" });
    if (res.ok) showToast(`Unloaded ${(await res.json()).unloaded} model(s).`);
    openSystemPanel();
  };
  document.getElementById("btn-open-logs").onclick = async () => {
    const res = await fetch("/api/logs/open", { method: "POST" });
    if (!res.ok) showToast(`Could not open the log folder: ${(await errorFromResponse(res)).message}`, "error");
  };

  document.addEventListener("keydown", handleShortcut);
  document.getElementById("tab-nodes").onclick = () => showSidebarTab("nodes");
  document.getElementById("tab-models").onclick = () => showSidebarTab("models");
  for (const id of ["tab-nodes", "tab-models"]) {
    document.getElementById(id).addEventListener("keydown", (e) => { // arrow keys move between tabs
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      const next = id === "tab-nodes" ? "models" : "nodes";
      showSidebarTab(next);
      document.getElementById(`tab-${next}`).focus();
    });
  }
  document.getElementById("model-search").addEventListener("input", renderModelLibrary);
  document.getElementById("model-refresh").onclick = loadModelLibrary;

  // Drops on the canvas: a palette node, a PNG with an embedded workflow,
  // or an image (uploaded and given a Load Image node). Capture phase +
  // stopPropagation keeps litegraph's own canvas drop handler out of it.
  const wrap = document.getElementById("canvas-wrap");
  wrap.addEventListener("dragover", (e) => {
    const types = [...(e.dataTransfer?.types || [])];
    if (types.includes("Files") || types.includes("application/x-pixelflow-node") || types.includes(MODEL_MIME)) e.preventDefault();
  }, true);
  wrap.addEventListener("drop", async (e) => {
    const nodeName = e.dataTransfer?.getData("application/x-pixelflow-node");
    const pos = canvas.convertEventToCanvasOffset(e);
    const modelData = e.dataTransfer?.getData(MODEL_MIME);
    if (modelData) {
      e.preventDefault();
      e.stopPropagation();
      try { addModelLoader(JSON.parse(modelData), pos); } catch (_) { /* not ours */ }
      return;
    }
    if (nodeName) {
      e.preventDefault();
      e.stopPropagation();
      addNodeToGraph(nodeName, pos);
      return;
    }
    const files = [...(e.dataTransfer?.files || [])];
    if (!files.length) return;
    e.preventDefault();
    e.stopPropagation();
    const json = files.find((f) => /\.json$/i.test(f.name));
    if (json) {
      await importFile(json);
      return;
    }
    const images = files.filter(Logic.isImageFile);
    for (const [i, file] of images.entries()) {
      if (file.type === "image/png") {
        const workflow = Logic.workflowFromPngChunks(Logic.readPngTextChunks(await file.arrayBuffer()));
        if (workflow) {
          const choice = await openDialog({
            title: "Workflow found",
            message: `${file.name} contains the workflow that made it.`,
            confirmText: "Load workflow",
            altText: "Use as image",
          });
          if (choice === "confirm") {
            if (await replaceGraph(workflow, "Load it?")) markSaved("");
            continue;
          }
          if (choice !== "alt") continue;
        }
      }
      addLoadImageNodeFor(file, [pos[0] + i * 30, pos[1] + i * 30]);
    }
  }, true);
  document.addEventListener("paste", (e) => {
    if (Logic.isTypingTarget(e.target)) return;
    const files = [...(e.clipboardData?.files || [])].filter(Logic.isImageFile);
    if (!files.length) return;
    e.preventDefault();
    files.forEach((file) => addLoadImageNodeFor(file));
  });

  // Litegraph's own right-click "Add Node" context menu only closes itself on
  // a click back on the canvas -- clicking the sidebar or the queue/gallery
  // panel instead leaves it stuck floating on screen. Force-close it on any
  // pointerdown outside the canvas area.
  document.addEventListener("pointerdown", (e) => {
    if (!document.getElementById("canvas-wrap").contains(e.target)) {
      LiteGraph.closeAllContextMenus(window);
    }
  });
}

async function addLoadImageNodeFor(file, pos) {
  if (!registeredType.LoadImage) return;
  try {
    const filename = await uploadImageFile(file);
    const node = addNodeToGraph("LoadImage", pos);
    setUploadedImage(node, node.uploadComboName, filename);
  } catch (err) {
    showToast(`Upload failed: ${err.message}`, "error");
  }
}

function resizeCanvas() {
  const wrap = document.getElementById("canvas-wrap");
  const el = document.getElementById("graph-canvas");
  el.width = wrap.clientWidth;
  el.height = wrap.clientHeight;
  if (canvas) canvas.resize();
}

async function main() {
  // Type-coloured links (F17).
  LGraphCanvas.link_type_colors = Object.assign(LGraphCanvas.link_type_colors || {}, {
    IMAGE: "#64b5f6",
    MODEL: "#b39ddb",
    CONDITIONING: "#ffa931",
    MASK: "#81c784",
  });

  graph = new LGraph();
  const el = document.getElementById("graph-canvas");
  canvas = new LGraphCanvas(el, graph);
  canvas.show_info = false; // LiteGraph's debug corner (T/I/N/V/FPS) is for its developers (UI1)
  for (const method of ["openSubgraph", "closeSubgraph"]) {
    const native = canvas[method].bind(canvas);
    canvas[method] = (...args) => { closePromptEditor(true); native(...args); updateSubgraphBar(); };
  }
  document.getElementById("subgraph-back").onclick = () => canvas.closeSubgraph();
  // litegraph's own corner panels (Graph Inputs/Outputs) would cover the tab bar; the Input and
  // Output nodes inside, and the bar above, do their job.
  canvas.drawSubgraphPanel = () => {};
  const subgraphMenu = LiteGraph.Subgraph.prototype.getExtraMenuOptions;
  LiteGraph.Subgraph.prototype.getExtraMenuOptions = function (graphCanvas, options) {
    const extra = [{ content: "Open subgraph (double-click)", callback: () => openSubgraph(this) }, null];
    return extra.concat(subgraphMenu ? subgraphMenu.call(this, graphCanvas, options) || [] : []);
  };
  LiteGraph.Subgraph.prototype.onDblClick = function () { openSubgraph(this); };
  el.addEventListener("wheel", promptWheel, { capture: true, passive: false });
  initCanvasToolbar();
  window.addEventListener("resize", resizeCanvas);
  new ResizeObserver(resizeCanvas).observe(document.getElementById("canvas-wrap"));
  resizeCanvas();

  await loadNodeDefs();
  initUI();
  // Bring back the graph from before the reload, if there is one. (The
  // queue panel and gallery are rebuilt from the server by resyncState()
  // once the WebSocket connects.)
  initTabs();
  restoreTabs();
  setConnectionStatus("reconnecting");
  connectWebSocket();

  // Undo covers every kind of edit (F6): nodes added/removed, links made or
  // broken, nodes moved, and widget values (pushed from their callbacks).
  graph.onNodeAdded = () => {
    scheduleHistoryPush();
    updateEmptyState();
  };
  graph.onNodeRemoved = () => {
    scheduleHistoryPush();
    updateEmptyState();
  };
  graph.onConnectionChange = scheduleHistoryPush;
  canvas.onNodeMoved = scheduleHistoryPush;

  graph.start();
  pushHistory();
  updateEmptyState();
  refreshSystemStatus();
  setInterval(refreshSystemStatus, 10000);
  // The server checks for a newer release once, in the background (backend/update_check.py);
  // with no feed configured /api/update is always {}.
  setTimeout(showAvailableUpdate, 8000);
}

async function showAvailableUpdate() {
  let info = {};
  try {
    const res = await fetch("/api/update");
    if (res.ok) info = await res.json();
  } catch (_) { /* offline or server gone: say nothing */ }
  if (!info.version) return;
  const toast = showToast(`PixelFlow ${info.version} is available.${info.notes ? ` ${info.notes}` : ""}`, "info", 0);
  const link = document.createElement("a");
  link.href = info.url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = "Download";
  toast.insertBefore(link, toast.lastChild);
}

main();
