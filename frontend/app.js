// PixelFlow — a small ComfyUI-style node graph editor.
// Node types are NOT hard-coded here: they are fetched from /api/nodes and
// used to dynamically build LiteGraph node classes, the same way ComfyUI's
// own web client builds nodes from its backend's object_info.

const LINK_TYPES = new Set(["IMAGE", "MODEL", "CONDITIONING"]);

let nodeDefs = {};
let registeredType = {}; // nodeName -> full LiteGraph type string e.g. "PixelFlow/image/Blur"

let graph, canvas;
let history = [];
let historyIndex = -1;
let suppressHistory = false;
let historyTimer = null;

function addPropertyWidget(node, name, spec) {
  const type = spec[0];
  const config = spec[1] || {};

  if (Array.isArray(type)) {
    node.properties[name] = config.default ?? type[0];
    node.addWidget("combo", name, node.properties[name], (v) => {
      node.properties[name] = v;
      scheduleHistoryPush();
    }, { values: type });
    return;
  }

  if (LINK_TYPES.has(type)) {
    node.addInput(name, type);
    return;
  }

  if (type === "INT" || type === "FLOAT") {
    node.properties[name] = config.default ?? 0;
    node.addWidget("number", name, node.properties[name], (v) => {
      node.properties[name] = v;
      scheduleHistoryPush();
    }, {
      min: config.min, max: config.max,
      step: type === "INT" ? 1 : 0.1,
      precision: type === "INT" ? 0 : 3,
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
  }
  NodeCtor.title = def.display_name;
  return NodeCtor;
}

async function loadNodeDefs() {
  const res = await fetch("/api/nodes");
  nodeDefs = await res.json();
  for (const [name, def] of Object.entries(nodeDefs)) {
    const cls = makeNodeClass(name, def);
    const fullType = `PixelFlow/${def.category}/${def.display_name}`;
    registeredType[name] = fullType;
    LiteGraph.registerNodeType(fullType, cls);
  }
  buildSidebar();
}

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
      const item = document.createElement("div");
      item.className = "node-item";
      item.textContent = nodeDefs[name].display_name;
      item.dataset.name = nodeDefs[name].display_name.toLowerCase();
      item.onclick = () => addNodeToGraph(name);
      group.appendChild(item);
    }
    container.appendChild(group);
  }
  filterNodeList(document.getElementById("node-search").value);
}

// ---- Node palette search (filters buildSidebar()'s items live as-typed) ----

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

// ---- Empty-state hint ----

function updateEmptyState() {
  document.getElementById("empty-state").classList.toggle("hidden", graph._nodes.length > 0);
}

// ---- Fit graph to view ----

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

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  canvas.ds.scale = scale;
  canvas.ds.offset[0] = viewW / 2 - centerX * scale;
  canvas.ds.offset[1] = viewH / 2 - centerY * scale;
  canvas.setDirty(true, true);
}

// ---- Prompt building (graph -> ComfyUI-style execution prompt) ----

function buildPrompt() {
  const execPrompt = {};
  for (const node of graph._nodes) {
    const inputs = Object.assign({}, node.properties);
    if (node.inputs) {
      node.inputs.forEach((inp, slot) => {
        if (inp.link != null) {
          const link = graph.links[inp.link];
          if (link) {
            inputs[inp.name] = [String(link.origin_id), link.origin_slot];
          }
        }
      });
    }
    execPrompt[String(node.id)] = { class_type: node.comfyClass, inputs };
  }
  return execPrompt;
}

// ---- Queue / websocket ----

const queueEntries = new Map();

function renderQueueEntry(promptId) {
  let el = queueEntries.get(promptId);
  if (!el) {
    el = document.createElement("div");
    el.className = "queue-entry";
    document.getElementById("queue-list").prepend(el);
    queueEntries.set(promptId, el);
  }
  return el;
}

function setQueueStatus(promptId, statusClass, text) {
  const el = renderQueueEntry(promptId);
  el.innerHTML = `<span class="${statusClass}">${text}</span>`;
}

function addGalleryImage(filename, promptId) {
  const gallery = document.getElementById("gallery");
  const item = document.createElement("div");
  item.className = "gallery-item";
  item.innerHTML = `<img src="/outputs/${filename}" alt="${filename}" /><div class="caption">${filename}</div>`;
  item.querySelector("img").onclick = () => openLightbox(`/outputs/${filename}`);
  gallery.prepend(item);
}

// ---- Lightbox (click a gallery thumbnail to view it full-size) ----

function openLightbox(src) {
  const overlay = document.getElementById("lightbox-overlay");
  document.getElementById("lightbox-img").src = src;
  overlay.classList.remove("hidden");
}

function closeLightbox() {
  document.getElementById("lightbox-overlay").classList.add("hidden");
  document.getElementById("lightbox-img").src = "";
}

// ---- Canvas execution highlighting ----

let executingNode = null;
let executingNodeOriginalColor;

function highlightExecutingNode(nodeId) {
  clearExecutingHighlight();
  const node = graph.getNodeById(Number(nodeId));
  if (!node) return;
  executingNode = node;
  executingNodeOriginalColor = node.color;
  node.color = "#5b8def";
  canvas.setDirty(true, true);
}

function clearExecutingHighlight() {
  if (!executingNode) return;
  executingNode.color = executingNodeOriginalColor;
  executingNode = null;
  canvas.setDirty(true, true);
}

// ---- Step progress bar (fed by "progress" websocket messages, e.g.
// KSampler's diffusion steps) ----

function showProgress(nodeId, step, total) {
  const wrap = document.getElementById("progress-wrap");
  const fill = document.getElementById("progress-fill");
  const label = document.getElementById("progress-label");
  const node = graph.getNodeById(Number(nodeId));
  const title = node ? node.title || node.comfyClass : nodeId;
  const pct = total > 0 ? Math.min(100, Math.round((step / total) * 100)) : 0;
  wrap.classList.remove("hidden");
  fill.style.width = `${pct}%`;
  label.textContent = `${title}: step ${step} / ${total}`;
}

function hideProgress() {
  const wrap = document.getElementById("progress-wrap");
  wrap.classList.add("hidden");
  document.getElementById("progress-fill").style.width = "0%";
}

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "queued") {
      setQueueStatus(msg.prompt_id, "status-queued", `Queued: ${msg.prompt_id}`);
    } else if (msg.type === "execution_start") {
      setQueueStatus(msg.prompt_id, "status-running", `Running: ${msg.prompt_id}`);
      clearExecutingHighlight();
      hideProgress();
    } else if (msg.type === "executing") {
      setQueueStatus(msg.prompt_id, "status-running", `Running: ${msg.prompt_id} — ${msg.class_type}`);
      highlightExecutingNode(msg.node_id);
      hideProgress(); // a new node started; it may not report step progress at all
    } else if (msg.type === "progress") {
      showProgress(msg.node_id, msg.step, msg.total);
    } else if (msg.type === "execution_success") {
      setQueueStatus(msg.prompt_id, "status-completed", `Done: ${msg.prompt_id}`);
      clearExecutingHighlight();
      hideProgress();
      for (const nodeOutput of Object.values(msg.outputs || {})) {
        for (const filename of nodeOutput.images || []) {
          addGalleryImage(filename, msg.prompt_id);
        }
      }
    } else if (msg.type === "execution_error") {
      setQueueStatus(msg.prompt_id, "status-error", `Error: ${msg.prompt_id} — ${msg.error}`);
      clearExecutingHighlight();
      hideProgress();
    }
  };
  ws.onclose = () => setTimeout(connectWebSocket, 2000);
}

// ---- Undo / redo ----

function scheduleHistoryPush() {
  if (suppressHistory) return;
  clearTimeout(historyTimer);
  historyTimer = setTimeout(pushHistory, 400);
}

function pushHistory() {
  if (suppressHistory) return;
  const snap = JSON.stringify(graph.serialize());
  if (history[historyIndex] === snap) return;
  history = history.slice(0, historyIndex + 1);
  history.push(snap);
  historyIndex = history.length - 1;
}

function syncWidgetsFromProperties() {
  // graph.configure() restores node.properties from the snapshot but does not
  // reliably refresh each widget's displayed value from it, so widgets can
  // show stale text after undo/redo or loading a workflow. Re-sync explicitly.
  for (const node of graph._nodes) {
    if (!node.widgets) continue;
    for (const widget of node.widgets) {
      if (widget.name in node.properties) widget.value = node.properties[widget.name];
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
}

function undo() {
  if (historyIndex <= 0) return;
  historyIndex--;
  restoreSnapshot(history[historyIndex]);
}

function redo() {
  if (historyIndex >= history.length - 1) return;
  historyIndex++;
  restoreSnapshot(history[historyIndex]);
}

// ---- Small themed modal (replaces window.prompt(), which doesn't render in
// every embedded browser context and looks out of place in a themed app) ----

function promptModal(title, defaultValue = "") {
  return new Promise((resolve) => {
    const overlay = document.getElementById("modal-overlay");
    const input = document.getElementById("modal-input");
    const confirmBtn = document.getElementById("modal-confirm");
    const cancelBtn = document.getElementById("modal-cancel");

    overlay.querySelector("h3").textContent = title;
    input.value = defaultValue;
    overlay.classList.remove("hidden");
    input.focus();
    input.select();

    function cleanup(result) {
      overlay.classList.add("hidden");
      confirmBtn.removeEventListener("click", onConfirm);
      cancelBtn.removeEventListener("click", onCancel);
      input.removeEventListener("keydown", onKeydown);
      resolve(result);
    }
    function onConfirm() {
      cleanup(input.value.trim() || null);
    }
    function onCancel() {
      cleanup(null);
    }
    function onKeydown(e) {
      if (e.key === "Enter") onConfirm();
      if (e.key === "Escape") onCancel();
    }
    confirmBtn.addEventListener("click", onConfirm);
    cancelBtn.addEventListener("click", onCancel);
    input.addEventListener("keydown", onKeydown);
  });
}

// ---- Workflow save / load / export / import ----

async function refreshWorkflowList() {
  const res = await fetch("/api/workflows");
  const names = await res.json();
  const sel = document.getElementById("workflow-select");
  sel.innerHTML = '<option value="">Load workflow...</option>' +
    names.map((n) => `<option value="${n}">${n}</option>`).join("");
}

function findInputSlot(node, name) {
  return node.inputs.findIndex((i) => i.name === name);
}

function loadExampleGraph() {
  graph.clear();

  const ckpt = addNodeToGraph("LoadCheckpoint", [50, 120]);

  const clip = addNodeToGraph("CLIPTextEncode", [320, 40]);
  clip.properties.text = "a scenic mountain lake at sunrise, cinematic lighting";
  if (clip.widgets && clip.widgets[0]) clip.widgets[0].value = clip.properties.text;

  const empty = addNodeToGraph("EmptyImage", [320, 260]);

  const sampler = addNodeToGraph("KSamplerStub", [640, 120]);

  const save = addNodeToGraph("SaveImage", [960, 120]);

  ckpt.connect(0, sampler, findInputSlot(sampler, "model"));
  clip.connect(0, sampler, findInputSlot(sampler, "positive"));
  empty.connect(0, sampler, findInputSlot(sampler, "latent_image"));
  sampler.connect(0, save, findInputSlot(save, "image"));

  canvas.setDirty(true, true);
  pushHistory();
  updateEmptyState();
}

// ---- Wiring up UI ----

function initUI() {
  document.getElementById("btn-queue").onclick = async () => {
    const execPrompt = buildPrompt();
    const res = await fetch("/api/queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: execPrompt }),
    });
    if (!res.ok) {
      alert("Failed to queue prompt: " + (await res.text()));
      return;
    }
    const data = await res.json();
    setQueueStatus(data.prompt_id, "status-queued", `Queued: ${data.prompt_id}`);
  };

  document.getElementById("btn-example").onclick = loadExampleGraph;

  document.getElementById("btn-clear").onclick = () => {
    graph.clear();
    pushHistory();
    updateEmptyState();
  };

  document.getElementById("btn-undo").onclick = undo;
  document.getElementById("btn-redo").onclick = redo;
  document.getElementById("btn-fit-view").onclick = fitView;

  document.getElementById("node-search").addEventListener("input", (e) => {
    filterNodeList(e.target.value);
  });

  document.getElementById("lightbox-overlay").addEventListener("click", closeLightbox);

  document.getElementById("btn-save-workflow").onclick = async () => {
    const name = await promptModal("Save Workflow As");
    if (!name) return;
    await fetch(`/api/workflows/${encodeURIComponent(name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(graph.serialize()),
    });
    refreshWorkflowList();
  };

  document.getElementById("workflow-select").onchange = async (e) => {
    const name = e.target.value;
    if (!name) return;
    const res = await fetch(`/api/workflows/${encodeURIComponent(name)}`);
    const data = await res.json();
    graph.configure(data);
    syncWidgetsFromProperties();
    pushHistory();
    updateEmptyState();
  };

  document.getElementById("btn-export").onclick = () => {
    const blob = new Blob([JSON.stringify(graph.serialize(), null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "workflow.json";
    a.click();
  };

  document.getElementById("file-import").onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      graph.configure(JSON.parse(reader.result));
      syncWidgetsFromProperties();
      pushHistory();
      updateEmptyState();
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !document.getElementById("lightbox-overlay").classList.contains("hidden")) {
      closeLightbox();
      return;
    }
    const mod = e.ctrlKey || e.metaKey;
    if (!mod) return;
    if (e.key.toLowerCase() === "z") {
      e.preventDefault();
      undo();
    } else if (e.key.toLowerCase() === "y") {
      e.preventDefault();
      redo();
    }
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

function resizeCanvas() {
  const wrap = document.getElementById("canvas-wrap");
  const el = document.getElementById("graph-canvas");
  el.width = wrap.clientWidth;
  el.height = wrap.clientHeight;
  if (canvas) canvas.resize();
}

async function main() {
  graph = new LGraph();
  const el = document.getElementById("graph-canvas");
  canvas = new LGraphCanvas(el, graph);
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  await loadNodeDefs();
  await refreshWorkflowList();
  initUI();
  connectWebSocket();

  graph.onNodeAdded = () => {
    scheduleHistoryPush();
    updateEmptyState();
  };
  graph.onNodeRemoved = () => {
    scheduleHistoryPush();
    updateEmptyState();
  };

  graph.start();
  pushHistory();
  updateEmptyState();
}

main();
