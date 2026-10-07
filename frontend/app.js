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

// ---- Widgets ----

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
  node.addWidget("text", name, node.properties[name], (v) => {
    node.properties[name] = v;
    scheduleHistoryPush();
  }, { multiline: !!config.multiline });
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
  const preview = livePreviews.get(node.id) || outputPreviews.get(node.id);
  if (preview && !node.uploadComboName) drawImageBelowWidgets(node, ctx, preview);
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
  const node = nodeId != null ? graph.getNodeById(Number(nodeId)) : null;
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
    for (const [pname, pspec] of Object.entries(all)) {
      addPropertyWidget(this, pname, pspec);
    }

    (def.return_types || []).forEach((rt, idx) => {
      const label = (def.return_names && def.return_names[idx]) || rt;
      this.addOutput(label, rt);
    });

    this.size = this.computeSize();
    if (this.uploadComboName || def.output_node) this.size[1] += 140;
  }
  NodeCtor.title = def.display_name;
  NodeCtor.desc = def.description || "";
  NodeCtor.prototype.onDrawForeground = function (ctx) {
    if (this.flags.collapsed) return;
    if (this.uploadComboName) drawUploadPreview(this, ctx);
    drawNodeDecorations(this, ctx);
  };
  NodeCtor.prototype.getExtraMenuOptions = function () {
    return [
      null,
      { content: this.mode === Logic.MODE_BYPASS ? "Un-bypass (Ctrl+B)" : "Bypass (Ctrl+B)", callback: () => toggleMode([this], Logic.MODE_BYPASS) },
      { content: this.mode === Logic.MODE_MUTE ? "Unmute (Ctrl+M)" : "Mute (Ctrl+M)", callback: () => toggleMode([this], Logic.MODE_MUTE) },
    ];
  };
  return NodeCtor;
}

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
  buildSidebar();
}

// ---- Node palette: focusable buttons, click to add or drag onto the canvas (F15) ----

function buildSidebar() {
  const container = document.getElementById("node-list");
  container.innerHTML = "";
  const byCategory = {};
  for (const [name, def] of Object.entries(nodeDefs)) {
    (byCategory[def.category] = byCategory[def.category] || []).push(name);
  }
  for (const category of Object.keys(byCategory).sort()) {
    const group = document.createElement("div");
    group.className = "category-group";
    const title = document.createElement("div");
    title.className = "category-title";
    title.textContent = category;
    group.appendChild(title);

    for (const name of byCategory[category]) {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "node-item";
      item.textContent = nodeDefs[name].display_name;
      item.title = nodeDefs[name].description || `Add ${nodeDefs[name].display_name}`;
      item.dataset.name = nodeDefs[name].display_name.toLowerCase();
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
  graph.add(node);
  scheduleHistoryPush();
  updateEmptyState();
  return node;
}

function updateEmptyState() {
  document.getElementById("empty-state").classList.toggle("hidden", graph._nodes.length > 0);
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

  canvas.ds.scale = scale;
  canvas.ds.offset[0] = viewW / 2 - ((minX + maxX) / 2) * scale;
  canvas.ds.offset[1] = viewH / 2 - ((minY + maxY) / 2) * scale;
  canvas.setDirty(true, true);
}

// ---- Prompt building (graph -> ComfyUI-style execution prompt; F12 mute/bypass) ----

function buildPrompt() {
  const nodes = graph._nodes.map((node) => ({
    id: node.id,
    mode: node.mode,
    comfyClass: nodeAliases[node.comfyClass] || node.comfyClass,
    properties: node.properties,
    inputs: (node.inputs || []).map((inp) => ({ name: inp.name, type: inp.type, link: inp.link })),
    outputs: (node.outputs || []).map((out) => ({ type: out.type })),
  }));
  return Logic.buildPrompt(nodes, graph.links);
}

// ---- Queueing (F8, F11) ----

function seedNodes() {
  return graph._nodes.filter((n) => n.seedInputName && n.mode !== Logic.MODE_MUTE && n.mode !== Logic.MODE_BYPASS);
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
      body: JSON.stringify({ prompt: execPrompt, front, extra_data: { extra_pnginfo: { workflow: graph.serialize() } } }),
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
  queueState.entries.set(data.prompt_id, { status: "queued", queued_at: Date.now() / 1000 });
  if (front) queueState.pending.unshift(data.prompt_id);
  else if (!queueState.pending.includes(data.prompt_id)) queueState.pending.push(data.prompt_id);
  renderQueueRow(data.prompt_id);
  return true;
}

// ---- Queue panel (F8) ----

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
  // Server-provided text (error messages, filenames, prompt ids) is only
  // ever inserted via textContent, never innerHTML.
  const [statusClass, text] = Logic.statusRow(promptId, entry, entry.status === "running");
  el.replaceChildren();
  const label = document.createElement("span");
  label.className = statusClass;
  label.textContent = entry.runningText || text;
  const detail = document.createElement("small");
  detail.className = "queue-detail";
  detail.textContent = Logic.queueDetail(entry, queueState.pending.indexOf(promptId), Date.now() / 1000);
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

function updateQueueEntry(promptId, fields) {
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

async function cancelAllPending() {
  const res = await fetch("/api/queue", { method: "DELETE" });
  if (!res.ok) return;
  for (const promptId of (await res.json()).cancelled) updateQueueEntry(promptId, { status: "cancelled" });
}

async function clearFinished() {
  const res = await fetch("/api/history", { method: "DELETE" });
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
  const node = graph.getNodeById(Number(msg.node_id));
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
  const node = graph.getNodeById(Number(msg.node_id));
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
    setGallery(Logic.galleryItems(historyEntries));
    if (!serverQueue.running) hideProgress();
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
  hideProgress();
  const entry = await historyEntry(msg.prompt_id);
  if (entry) {
    historyCache.set(msg.prompt_id, entry);
    queueState.entries.set(msg.prompt_id, entry);
    renderQueueRow(msg.prompt_id);
  }
  const items = Logic.galleryItems({ [msg.prompt_id]: entry || { status: "completed", outputs: msg.outputs } });
  addGalleryItems(items);
  // F13: show each output node's newest image inside the node.
  for (const [nodeId, output] of Object.entries(msg.outputs || {})) {
    const node = graph.getNodeById(Number(nodeId));
    if (node && output && Array.isArray(output.images) && output.images.length) {
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
    switch (msg.type) {
      case "queued":
        if (!queueState.entries.has(msg.prompt_id)) {
          queueState.entries.set(msg.prompt_id, { status: "queued", queued_at: Date.now() / 1000 });
          queueState.pending.push(msg.prompt_id);
        }
        renderQueueRow(msg.prompt_id);
        break;
      case "execution_start":
        updateQueueEntry(msg.prompt_id, { status: "running", started_at: Date.now() / 1000, runningText: "" });
        hideProgress();
        break;
      case "executing":
        updateQueueEntry(msg.prompt_id, { status: "running", runningText: `Running: ${msg.prompt_id} — ${msg.class_type}` });
        setRunningNode(msg.node_id);
        livePreviews.clear();
        showNodeStart(msg);
        break;
      case "execution_cached":
        if (msg.node_index) {
          progress.nodeIndex = msg.node_index;
          progress.nodeCount = msg.node_count;
          setProgressBar("progress-overall", Logic.overallProgress(progress.nodeIndex, progress.nodeCount, 1, 1));
        }
        break;
      case "progress":
        showProgress(msg);
        break;
      case "preview":
        showPreview(msg);
        break;
      case "execution_success":
        onExecutionSuccess(msg);
        break;
      case "execution_cancelled":
        updateQueueEntry(msg.prompt_id, { status: "cancelled" });
        break;
      case "execution_interrupted":
        updateQueueEntry(msg.prompt_id, { status: "interrupted", finished_at: Date.now() / 1000 });
        hideProgress();
        break;
      case "execution_error":
        updateQueueEntry(msg.prompt_id, { status: "error", error: msg.error, class_type: msg.class_type, finished_at: Date.now() / 1000 });
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

function markSaved(name) {
  savedSnapshot = JSON.stringify(graph.serialize());
  if (name !== undefined) currentWorkflowName = name;
}

// Auto-save: every undo snapshot is also the "last graph" restored on reload.
function saveGraphLocally(snap) {
  try {
    localStorage.setItem(GRAPH_STORAGE_KEY, snap);
  } catch (_) { /* storage full or disabled -- auto-save is best effort */ }
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
    graph.configure(JSON.parse(saved));
  } catch (err) {
    console.warn("could not restore the auto-saved graph", err);
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
  for (const node of graph._nodes) {
    if (!node.widgets) continue;
    for (const widget of node.widgets) {
      if (widget.name in node.properties) widget.value = node.properties[widget.name];
      else if (widget.name === "control after generate" && node.properties[SEED_CONTROL_PROPERTY]) {
        widget.value = node.properties[SEED_CONTROL_PROPERTY];
      }
    }
  }
  canvas.setDirty(true, true);
}

function restoreSnapshot(snap) {
  suppressHistory = true;
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
    graph.configure(data);
  } catch (err) {
    showToast(`That workflow could not be loaded: ${err.message}`, "error");
    return false;
  } finally {
    suppressHistory = false;
  }
  syncWidgetsFromProperties();
  clearNodeErrors();
  outputPreviews.clear();
  pushHistory();
  updateEmptyState();
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
  const body = JSON.stringify(graph.serialize());
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
  if (await replaceGraph(await res.json(), `Load "${name}"?`)) {
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
    if (await replaceGraph(workflow, `Load the workflow embedded in ${file.name}?`)) markSaved("");
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
  if (await replaceGraph(data, `Import ${file.name}?`)) markSaved("");
}

function findInputSlot(node, name) {
  return node.inputs.findIndex((i) => i.name === name);
}

function setNodeValue(node, name, value) {
  node.properties[name] = value;
  const widgetName = name === SEED_CONTROL_PROPERTY ? "control after generate" : name;
  const widget = (node.widgets || []).find((w) => w.name === widgetName);
  if (widget) widget.value = value;
}

// ---- Templates (F18): read-only starter graphs under frontend/templates/ ----

// A template is {nodes: [{id, type, pos, values}], links: [[from id, output slot,
// to id, input name]]} -- node ids are the template's own names, so the files stay
// readable and independent of litegraph's serialization.
function buildTemplateGraph(spec) {
  graph.clear();
  outputPreviews.clear();
  const nodes = new Map();
  for (const entry of spec.nodes) {
    const node = addNodeToGraph(entry.type, entry.pos);
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
  markSaved("");
  updateEmptyState();
  fitView();
}

async function loadTemplate(file, title = file) {
  if (isDirty() && !(await confirmModal("Unsaved changes", `Load the "${title}" template? Your unsaved changes will be lost.`, "Load template"))) {
    return false;
  }
  try {
    const res = await fetch(`templates/${file}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    buildTemplateGraph(await res.json());
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
      button.append(title, description);
      button.onclick = async () => {
        closeOverlay("template-overlay");
        await loadTemplate(template.file, template.title);
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
      button.textContent = `CPU · torch ${info.torch}${info.cuda ? "" : " (no CUDA)"}`;
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
}

// ---- Wiring up UI ----

function handleShortcut(e) {
  for (const id of ["modal-overlay", "workflow-overlay", "template-overlay", "system-overlay", "help-overlay"]) {
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
  const action = Logic.shortcutFor(e);
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
    help: () => openOverlay("help-overlay"),
    escape: () => {},
  };
  e.preventDefault();
  actions[action]();
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
  document.getElementById("btn-queue").onclick = (e) => queuePrompt({ front: e.shiftKey, count: queueCount() });
  document.getElementById("btn-example").onclick = loadExampleGraph;
  document.getElementById("btn-templates").onclick = openTemplates;
  document.getElementById("template-close").onclick = () => closeOverlay("template-overlay");
  document.getElementById("btn-clear").onclick = async () => {
    if (isDirty() && !(await confirmModal("Clear the graph?", "Your unsaved changes will be lost.", "Clear"))) return;
    graph.clear();
    outputPreviews.clear();
    pushHistory();
    markSaved("");
    updateEmptyState();
  };
  document.getElementById("btn-undo").onclick = undo;
  document.getElementById("btn-redo").onclick = redo;
  document.getElementById("btn-fit-view").onclick = fitView;
  document.getElementById("btn-help").onclick = () => openOverlay("help-overlay");
  document.getElementById("help-close").onclick = () => closeOverlay("help-overlay");
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
  document.getElementById("file-import").onchange = async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (file) await importFile(file);
  };

  // System
  document.getElementById("system-status").onclick = openSystemPanel;
  document.getElementById("system-close").onclick = () => closeOverlay("system-overlay");
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

  // Drops on the canvas: a palette node, a PNG with an embedded workflow,
  // or an image (uploaded and given a Load Image node). Capture phase +
  // stopPropagation keeps litegraph's own canvas drop handler out of it.
  const wrap = document.getElementById("canvas-wrap");
  wrap.addEventListener("dragover", (e) => {
    const types = [...(e.dataTransfer?.types || [])];
    if (types.includes("Files") || types.includes("application/x-pixelflow-node")) e.preventDefault();
  }, true);
  wrap.addEventListener("drop", async (e) => {
    const nodeName = e.dataTransfer?.getData("application/x-pixelflow-node");
    const pos = canvas.convertEventToCanvasOffset(e);
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
  window.addEventListener("resize", resizeCanvas);
  new ResizeObserver(resizeCanvas).observe(document.getElementById("canvas-wrap"));
  resizeCanvas();

  await loadNodeDefs();
  initUI();
  // Bring back the graph from before the reload, if there is one. (The
  // queue panel and gallery are rebuilt from the server by resyncState()
  // once the WebSocket connects.)
  restoreSavedGraph();
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
}

main();
