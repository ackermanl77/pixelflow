// PixelFlow — pure (DOM- and litegraph-free) frontend logic, split out of
// app.js so it can be unit-tested with plain Node (`node --test tests/js`).
// Loaded as a classic <script> before app.js, where it's `window.PixelFlowLogic`;
// under Node it's a CommonJS module.

(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PixelFlowLogic = api;
})(typeof self !== "undefined" ? self : this, function () {
  // Mirrors backend/engine/graph.py's PRIMITIVE_TYPES: these are widgets;
  // any other string type (IMAGE, MODEL, CONDITIONING, MASK, ...) is a socket.
  const PRIMITIVE_TYPES = new Set(["INT", "FLOAT", "STRING", "BOOLEAN"]);

  /** How an INPUT_TYPES spec ([type, config]) is rendered on a node. */
  function widgetKind(spec) {
    const type = spec[0];
    if (Array.isArray(type)) return "combo";
    if (!PRIMITIVE_TYPES.has(type)) return "socket";
    if (type === "INT" || type === "FLOAT") return "number";
    if (type === "BOOLEAN") return "toggle";
    return "text";
  }

  /** The real per-click step of a number input (backend `step`, else 1 / 0.1). */
  function realStep(type, config) {
    const cfg = config || {};
    if (typeof cfg.step === "number" && cfg.step > 0) return type === "INT" ? Math.max(1, Math.round(cfg.step)) : cfg.step;
    return type === "INT" ? 1 : 0.1;
  }

  function decimalsOf(step) {
    const text = String(step);
    const dot = text.indexOf(".");
    return dot < 0 ? 0 : text.length - dot - 1;
  }

  /**
   * Options for litegraph's "number" widget. The vendored litegraph moves
   * the value by 0.1 * options.step per arrow click / drag unit, so the
   * step handed to it is 10x the real step (the same convention ComfyUI
   * uses). Without this an INT widget moved by 0.1 and produced values
   * like 4.1 that the backend rejected.
   */
  function numberWidgetOptions(type, config) {
    const cfg = config || {};
    const step = realStep(type, cfg);
    return {
      min: cfg.min,
      max: cfg.max,
      step: step * 10,
      precision: type === "INT" ? 0 : Math.max(2, Math.min(4, decimalsOf(step))),
    };
  }

  /**
   * Normalize a number widget's value before it goes into node.properties:
   * INT is rounded to an integer, FLOAT is snapped to the backend's `round`
   * (when it declares one) or else rounded to 4 decimals to drop binary
   * noise like 0.30000000000000004, and both are clamped to min/max.
   * Non-finite input falls back to the default (or 0).
   */
  function normalizeNumber(value, type, config) {
    const cfg = config || {};
    let v = typeof value === "number" ? value : parseFloat(value);
    if (!Number.isFinite(v)) v = typeof cfg.default === "number" ? cfg.default : 0;
    if (type === "INT") {
      v = Math.round(v);
    } else if (typeof cfg.round === "number" && cfg.round > 0) {
      v = Math.round(v / cfg.round) * cfg.round;
      v = parseFloat(v.toFixed(decimalsOf(cfg.round)));
    } else {
      v = parseFloat(v.toFixed(4));
    }
    if (typeof cfg.min === "number" && v < cfg.min) v = cfg.min;
    if (typeof cfg.max === "number" && v > cfg.max) v = cfg.max;
    return Object.is(v, -0) ? 0 : v;
  }

  /**
   * A /api/queue 400's `detail` is {message, node_id, class_type, input}
   * (older servers / other errors send a plain string). Returns
   * {message, nodeId, input}.
   */
  function parseErrorDetail(detail) {
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      const message = typeof detail.message === "string" ? detail.message : JSON.stringify(detail);
      return { message, nodeId: detail.node_id ?? null, input: detail.input ?? null };
    }
    if (Array.isArray(detail)) {
      // FastAPI's own request-validation errors (422) are a list.
      return { message: detail.map((d) => d.msg || JSON.stringify(d)).join("; "), nodeId: null, input: null };
    }
    return { message: String(detail ?? "Unknown error"), nodeId: null, input: null };
  }

  /** WebSocket reconnect delay: 0.5s, 1s, 2s, ... capped at 10s. */
  function reconnectDelay(attempt) {
    return Math.min(10000, 500 * Math.pow(2, Math.max(0, attempt)));
  }

  /** [statusClass, label] for a history entry, matching the live WS rows. */
  function statusRow(promptId, entry, running) {
    const status = running ? "running" : (entry && entry.status) || "queued";
    switch (status) {
      case "queued":
        return ["status-queued", `Queued: ${promptId}`];
      case "running":
        return ["status-running", `Running: ${promptId}`];
      case "completed": {
        const seeds = Object.values((entry && entry.outputs) || {})
          .filter((o) => o && o.seed !== undefined)
          .map((o) => o.seed);
        return ["status-completed", `Done: ${promptId}${seeds.length ? ` (seed ${seeds.join(", ")})` : ""}`];
      }
      case "cancelled":
        return ["status-cancelled", `Cancelled: ${promptId}`];
      case "interrupted":
        return ["status-cancelled", `Interrupted: ${promptId}`];
      case "error": {
        const where = entry && entry.class_type ? ` — ${entry.class_type}` : "";
        return ["status-error", `Error: ${promptId}${where} — ${(entry && entry.error) || "failed"}`];
      }
      default:
        return ["status-queued", `${status}: ${promptId}`];
    }
  }

  /**
   * Rows to show after a reload/reconnect, oldest first (the panel
   * prepends, so the newest ends up on top): every history entry in
   * server order, with the running id forced to "running".
   */
  function queueRows(history, queueState) {
    const running = queueState && queueState.running;
    return Object.entries(history || {}).map(([promptId, entry]) => {
      const [statusClass, text] = statusRow(promptId, entry, promptId === running);
      return { promptId, statusClass, text };
    });
  }

  /** Gallery images from history, oldest first: [{filename, promptId}]. */
  function galleryImages(history) {
    const out = [];
    for (const [promptId, entry] of Object.entries(history || {})) {
      if (!entry || entry.status !== "completed") continue;
      for (const nodeOutput of Object.values(entry.outputs || {})) {
        for (const filename of (nodeOutput && nodeOutput.images) || []) out.push({ filename, promptId });
      }
    }
    return out;
  }

  /** True for a File/DataTransferItem-like object that is an image. */
  function isImageFile(file) {
    return !!file && typeof file.type === "string" && file.type.startsWith("image/");
  }

  // ---- Node modes (F12): litegraph's NEVER is "mute"; 4 is ComfyUI's "bypass" ----

  const MODE_MUTE = 2;
  const MODE_BYPASS = 4;

  /** Node properties starting with "_" are frontend-only (e.g. the seed's
   * "control after generate" choice) and never sent to the backend. */
  function isFrontendOnlyProperty(name) {
    return typeof name === "string" && name.startsWith("_");
  }

  /**
   * Build the ComfyUI-style execution prompt from a plain description of
   * the graph: nodes = [{id, mode, comfyClass, properties, inputs:
   * [{name, type, link}], outputs: [{type}]}], links = {id: {origin_id,
   * origin_slot}}. Muted nodes (mode 2) are left out, so anything that
   * needs them drops out too; a bypassed node (mode 4) is left out and each
   * consumer is wired to the bypassed node's first input of the same type
   * instead (through chains of bypassed nodes).
   */
  function buildPrompt(nodes, links) {
    const byId = new Map(nodes.map((n) => [String(n.id), n]));
    const linkOf = (id) => (links instanceof Map ? links.get(id) : links[id]);

    function resolveSource(originId, originSlot, seen) {
      const node = byId.get(String(originId));
      if (!node || node.mode === MODE_MUTE) return null;
      if (node.mode !== MODE_BYPASS) return [String(originId), originSlot];
      if (seen.has(node.id)) return null;
      seen.add(node.id);
      const outType = ((node.outputs || [])[originSlot] || {}).type;
      const input = (node.inputs || []).find((inp) => inp.type === outType && inp.link != null);
      if (!input) return null;
      const link = linkOf(input.link);
      return link ? resolveSource(link.origin_id, link.origin_slot, seen) : null;
    }

    const prompt = {};
    for (const node of nodes) {
      if (node.mode === MODE_MUTE || node.mode === MODE_BYPASS) continue;
      const inputs = {};
      for (const [name, value] of Object.entries(node.properties || {})) {
        if (!isFrontendOnlyProperty(name)) inputs[name] = value;
      }
      for (const inp of node.inputs || []) {
        if (inp.link == null) continue;
        const link = linkOf(inp.link);
        const source = link && resolveSource(link.origin_id, link.origin_slot, new Set());
        if (source) inputs[inp.name] = source;
        else delete inputs[inp.name];
      }
      prompt[String(node.id)] = { class_type: node.comfyClass, inputs };
    }
    return prompt;
  }

  // ---- Seed "control after generate" (F11) ----

  const MAX_SEED = 2147483647;
  const SEED_CONTROLS = ["randomize", "increment", "fixed"];

  function randomSeed(rand = Math.random) {
    return Math.floor(rand() * (MAX_SEED + 1));
  }

  /** The seed actually sent: never -1 (a random one is picked instead), so
   * every queued prompt is reproducible and cacheable. */
  function seedForQueue(seed, rand = Math.random) {
    const value = Number(seed);
    return Number.isFinite(value) && value >= 0 ? Math.floor(value) : randomSeed(rand);
  }

  /** The widget's seed after a queue, per the node's control choice. */
  function seedAfterQueue(control, usedSeed, rand = Math.random) {
    if (control === "increment") return (usedSeed + 1) % (MAX_SEED + 1);
    if (control === "randomize") return randomSeed(rand);
    return usedSeed;
  }

  // ---- PNG metadata (F7: load a workflow from a generated PNG) ----

  /** {keyword: text} of a PNG's tEXt (and uncompressed iTXt) chunks, or null
   * when the bytes aren't a PNG. */
  function readPngTextChunks(buffer) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const signature = [137, 80, 78, 71, 13, 10, 26, 10];
    if (bytes.length < 8 || signature.some((b, i) => bytes[i] !== b)) return null;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const latin1 = (start, end) => {
      let s = "";
      for (let i = start; i < end; i++) s += String.fromCharCode(bytes[i]);
      return s;
    };
    const utf8 = (start, end) => new TextDecoder("utf-8").decode(bytes.subarray(start, end));
    const chunks = {};
    let offset = 8;
    while (offset + 8 <= bytes.length) {
      const length = view.getUint32(offset);
      const type = latin1(offset + 4, offset + 8);
      const start = offset + 8;
      const end = start + length;
      if (end > bytes.length) break;
      if (type === "tEXt") {
        const zero = bytes.indexOf(0, start);
        if (zero > start && zero < end) chunks[latin1(start, zero)] = utf8(zero + 1, end);
      } else if (type === "iTXt") {
        const zero = bytes.indexOf(0, start);
        if (zero > start && zero + 2 < end && bytes[zero + 1] === 0) {
          // keyword \0 compressionFlag(0) method \0 language \0 translated \0 text
          let p = zero + 3;
          p = bytes.indexOf(0, p) + 1; // language tag
          p = bytes.indexOf(0, p) + 1; // translated keyword
          if (p > 0 && p <= end) chunks[latin1(start, zero)] = utf8(p, end);
        }
      } else if (type === "IEND") {
        break;
      }
      offset = end + 4; // skip CRC
    }
    return chunks;
  }

  /** The litegraph workflow embedded by Save/Preview Image, or null. */
  function workflowFromPngChunks(chunks) {
    if (!chunks || typeof chunks.workflow !== "string") return null;
    try {
      const data = JSON.parse(chunks.workflow);
      return data && typeof data === "object" && Array.isArray(data.nodes) ? data : null;
    } catch (_) {
      return null;
    }
  }

  // ---- Queue panel and progress text (F8, F10) ----

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "";
    if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${String(s).padStart(2, "0")}s`;
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return "";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit++;
    }
    return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
  }

  /** Extra text for a queue row: "#2 in queue", or how long it ran. */
  function queueDetail(entry, position, now) {
    if (!entry) return "";
    if (entry.status === "queued" && position >= 0) return `#${position + 1} in queue`;
    if (entry.status === "running" && entry.started_at) return `running ${formatDuration(now - entry.started_at)}`;
    if (entry.started_at && entry.finished_at) return `took ${formatDuration(entry.finished_at - entry.started_at)}`;
    return "";
  }

  /** "Node 2/5 · KSampler · step 7/20 · 3.1 it/s · ~4s left" */
  function progressText({ nodeIndex, nodeCount, title, step, total, itPerS, etaS, unit }) {
    const parts = [];
    if (nodeIndex && nodeCount) parts.push(`Node ${nodeIndex}/${nodeCount}`);
    if (title) parts.push(title);
    if (unit === "bytes") parts.push(`downloading ${formatBytes(step)} / ${formatBytes(total)}`);
    else if (total) parts.push(`step ${step}/${total}`);
    if (itPerS) parts.push(`${itPerS >= 10 ? Math.round(itPerS) : itPerS.toFixed(2)} it/s`);
    if (Number.isFinite(etaS) && etaS > 0) parts.push(`~${formatDuration(etaS)} left`);
    return parts.join(" · ");
  }

  /** Overall progress 0..1 from the node position and the step within it. */
  function overallProgress(nodeIndex, nodeCount, step, total) {
    if (!nodeCount) return total ? step / total : 0;
    const within = total ? Math.min(1, step / total) : 0;
    return Math.min(1, Math.max(0, (Math.max(1, nodeIndex) - 1 + within) / nodeCount));
  }

  // ---- Gallery (F9) ----

  /** Gallery items from history, oldest first, with what the lightbox
   * shows: [{filename, type, promptId, seed, meta}]. */
  function galleryItems(history) {
    const out = [];
    for (const [promptId, entry] of Object.entries(history || {})) {
      if (!entry || entry.status !== "completed") continue;
      const meta = promptMetadata(entry.prompt);
      for (const nodeOutput of Object.values(entry.outputs || {})) {
        if (!nodeOutput || !Array.isArray(nodeOutput.images)) continue;
        for (const filename of nodeOutput.images) {
          out.push({ filename, type: nodeOutput.type || "output", promptId, meta });
        }
      }
    }
    return out;
  }

  /** Seed / sampler / steps / cfg / prompt text of a prompt's first sampler. */
  function promptMetadata(prompt) {
    const meta = {};
    const nodes = Object.entries(prompt || {});
    for (const [, node] of nodes) {
      const inputs = (node && node.inputs) || {};
      if (inputs.seed !== undefined && inputs.steps !== undefined) {
        Object.assign(meta, {
          sampler: inputs.sampler,
          scheduler: inputs.scheduler,
          seed: inputs.seed,
          steps: inputs.steps,
          cfg: inputs.cfg,
          denoise: inputs.denoise,
        });
        const positive = Array.isArray(inputs.positive) ? prompt[inputs.positive[0]] : null;
        const negative = Array.isArray(inputs.negative) ? prompt[inputs.negative[0]] : null;
        if (positive && positive.inputs) meta.positive = positive.inputs.text;
        if (negative && negative.inputs) meta.negative = negative.inputs.text;
        const model = Array.isArray(inputs.model) ? prompt[inputs.model[0]] : null;
        if (model && model.inputs && model.inputs.ckpt_name) meta.model = model.inputs.ckpt_name;
        break;
      }
    }
    for (const key of Object.keys(meta)) if (meta[key] === undefined) delete meta[key];
    return meta;
  }

  /** URL of a generated image (subfolder segments encoded one by one). */
  function imageUrl(filename, type) {
    const base = type === "temp" ? "/temp/" : "/outputs/";
    return base + String(filename).split("/").map(encodeURIComponent).join("/");
  }

  /** Next lightbox index, wrapping around. */
  function stepIndex(index, delta, length) {
    if (!length) return -1;
    return (((index + delta) % length) + length) % length;
  }

  /** Keep the newest `max` gallery items (DOM cap). */
  function capItems(items, max) {
    return items.length > max ? items.slice(items.length - max) : items;
  }

  // ---- Keyboard (F6, F14) ----

  /** True when a keyboard shortcut must not fire because the user is
   * typing (inputs, text areas, selects, contenteditable). */
  function isTypingTarget(target) {
    if (!target) return false;
    const tag = (target.tagName || "").toUpperCase();
    if (tag === "TEXTAREA" || tag === "SELECT") return true;
    if (tag === "INPUT") {
      const type = (target.type || "text").toLowerCase();
      return !["button", "checkbox", "radio", "submit", "reset", "file", "range", "color"].includes(type);
    }
    return !!target.isContentEditable;
  }

  /** The action for a keydown (or null): see the "?" overlay. */
  function shortcutFor(e) {
    const mod = e.ctrlKey || e.metaKey;
    const key = (e.key || "").toLowerCase();
    if (mod && key === "enter") return e.shiftKey ? "queue-front" : "queue";
    if (mod && key === "z") return e.shiftKey ? "redo" : "undo";
    if (mod && key === "y") return "redo";
    if (mod && key === "s") return "save";
    if (mod && key === "o") return "open";
    if (mod && key === "m") return "mute";
    if (mod && key === "b") return "bypass";
    if (!mod && !e.altKey && e.key === "?") return "help";
    if (!mod && e.key === "Escape") return "escape";
    return null;
  }

  /** Undo stack push with a cap (F6): returns the new [stack, index]. */
  function pushUndo(stack, index, snapshot, max) {
    if (stack[index] === snapshot) return [stack, index];
    const next = stack.slice(0, index + 1);
    next.push(snapshot);
    const trimmed = next.length > max ? next.slice(next.length - max) : next;
    return [trimmed, trimmed.length - 1];
  }

  return {
    PRIMITIVE_TYPES,
    widgetKind,
    numberWidgetOptions,
    normalizeNumber,
    parseErrorDetail,
    reconnectDelay,
    statusRow,
    queueRows,
    galleryImages,
    isImageFile,
    MODE_MUTE,
    MODE_BYPASS,
    isFrontendOnlyProperty,
    buildPrompt,
    MAX_SEED,
    SEED_CONTROLS,
    seedForQueue,
    seedAfterQueue,
    readPngTextChunks,
    workflowFromPngChunks,
    formatDuration,
    formatBytes,
    queueDetail,
    progressText,
    overallProgress,
    galleryItems,
    promptMetadata,
    imageUrl,
    stepIndex,
    capItems,
    isTypingTarget,
    shortcutFor,
    pushUndo,
  };
});
