// Unit tests for the frontend logic added for the P2 roadmap items
// (mute/bypass, seed control, PNG workflows, queue/progress text, gallery,
// shortcuts, undo). Run with `node --test tests/js`.
const test = require("node:test");
const assert = require("node:assert/strict");
const L = require("../../frontend/logic.js");

function graphFixture() {
  // 1 Load -> 2 Blur -> 3 Save ; 4 Preview fed by 2 ; links keyed by id.
  const nodes = [
    { id: 1, mode: 0, comfyClass: "LoadImage", properties: { filename: "a.png" }, inputs: [], outputs: [{ type: "IMAGE" }, { type: "MASK" }] },
    { id: 2, mode: 0, comfyClass: "ImageBlur", properties: { radius: 2 }, inputs: [{ name: "image", type: "IMAGE", link: 10 }], outputs: [{ type: "IMAGE" }] },
    { id: 3, mode: 0, comfyClass: "SaveImage", properties: { filename_prefix: "x" }, inputs: [{ name: "image", type: "IMAGE", link: 11 }], outputs: [] },
    { id: 4, mode: 0, comfyClass: "PreviewImage", properties: {}, inputs: [{ name: "image", type: "IMAGE", link: 12 }], outputs: [] },
  ];
  const links = { 10: { origin_id: 1, origin_slot: 0 }, 11: { origin_id: 2, origin_slot: 0 }, 12: { origin_id: 2, origin_slot: 0 } };
  return { nodes, links };
}

test("buildPrompt wires links and drops frontend-only properties (F11/F12)", () => {
  const { nodes, links } = graphFixture();
  nodes[1].properties._seed_control = "fixed";
  const prompt = L.buildPrompt(nodes, links);
  assert.deepEqual(prompt["2"], { class_type: "ImageBlur", inputs: { radius: 2, image: ["1", 0] } });
  assert.deepEqual(prompt["3"].inputs.image, ["2", 0]);
});

test("a muted node is left out, and so are links to it (F12)", () => {
  const { nodes, links } = graphFixture();
  nodes[3].mode = L.MODE_MUTE; // mute the preview
  nodes[1].mode = L.MODE_MUTE; // and the blur
  const prompt = L.buildPrompt(nodes, links);
  assert.deepEqual(Object.keys(prompt).sort(), ["1", "3"]);
  assert.equal(prompt["3"].inputs.image, undefined, "the save node's input is now missing");
});

test("a bypassed node passes its same-typed input through (F12)", () => {
  const { nodes, links } = graphFixture();
  nodes[1].mode = L.MODE_BYPASS;
  const prompt = L.buildPrompt(nodes, links);
  assert.ok(!("2" in prompt));
  assert.deepEqual(prompt["3"].inputs.image, ["1", 0]);
  assert.deepEqual(prompt["4"].inputs.image, ["1", 0]);
});

test("chains of bypassed nodes are followed; a bypass loop doesn't hang", () => {
  const nodes = [
    { id: 1, mode: 0, comfyClass: "A", properties: {}, inputs: [], outputs: [{ type: "IMAGE" }] },
    { id: 2, mode: 4, comfyClass: "B", properties: {}, inputs: [{ name: "i", type: "IMAGE", link: 1 }], outputs: [{ type: "IMAGE" }] },
    { id: 3, mode: 4, comfyClass: "B", properties: {}, inputs: [{ name: "i", type: "IMAGE", link: 2 }], outputs: [{ type: "IMAGE" }] },
    { id: 4, mode: 0, comfyClass: "S", properties: {}, inputs: [{ name: "i", type: "IMAGE", link: 3 }], outputs: [] },
    { id: 5, mode: 4, comfyClass: "B", properties: {}, inputs: [{ name: "i", type: "IMAGE", link: 4 }], outputs: [{ type: "IMAGE" }] },
    { id: 6, mode: 0, comfyClass: "S", properties: {}, inputs: [{ name: "i", type: "IMAGE", link: 5 }], outputs: [] },
  ];
  const links = {
    1: { origin_id: 1, origin_slot: 0 },
    2: { origin_id: 2, origin_slot: 0 },
    3: { origin_id: 3, origin_slot: 0 },
    4: { origin_id: 5, origin_slot: 0 },
    5: { origin_id: 5, origin_slot: 0 },
  };
  const prompt = L.buildPrompt(nodes, links);
  assert.deepEqual(prompt["4"].inputs.i, ["1", 0]);
  assert.equal(prompt["6"].inputs.i, undefined);
});

test("seeds: never -1 on the wire; control after generate (F11)", () => {
  const rand = () => 0.5;
  assert.equal(L.seedForQueue(-1, rand), Math.floor(0.5 * (L.MAX_SEED + 1)));
  assert.equal(L.seedForQueue(42, rand), 42);
  assert.equal(L.seedAfterQueue("increment", 41), 42);
  assert.equal(L.seedAfterQueue("increment", L.MAX_SEED), 0);
  assert.equal(L.seedAfterQueue("fixed", 7), 7);
  assert.equal(L.seedAfterQueue("randomize", 7, () => 0), 0);
});

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const b of bytes) {
    let c = (crc ^ b) & 0xff;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crc = (crc >>> 8) ^ c;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  out.set([...type].map((ch) => ch.charCodeAt(0)), 4);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

function pngWith(textChunks) {
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", new Uint8Array(13))];
  for (const [key, value] of Object.entries(textChunks)) {
    parts.push(chunk("tEXt", new Uint8Array([...Buffer.from(key, "latin1"), 0, ...Buffer.from(value, "utf8")])));
  }
  parts.push(chunk("IEND", new Uint8Array(0)));
  return Buffer.concat(parts.map((p) => Buffer.from(p)));
}

test("the workflow embedded in a PNG is read back (F7)", () => {
  const workflow = { nodes: [{ id: 1, type: "PixelFlow/image/Blur" }], links: [], note: "héllo" };
  const png = pngWith({ prompt: "{}", workflow: JSON.stringify(workflow) });
  const chunks = L.readPngTextChunks(png);
  assert.equal(chunks.prompt, "{}");
  assert.deepEqual(L.workflowFromPngChunks(chunks), workflow);
  assert.equal(L.readPngTextChunks(Buffer.from("not a png")), null);
  assert.equal(L.workflowFromPngChunks(L.readPngTextChunks(pngWith({ prompt: "{}" }))), null);
  assert.equal(L.workflowFromPngChunks({ workflow: "{broken" }), null);
});

test("queue rows show position and timing (F8)", () => {
  assert.equal(L.queueDetail({ status: "queued" }, 1, 0), "#2 in queue");
  assert.equal(L.queueDetail({ status: "completed", started_at: 10, finished_at: 12.5 }, -1, 0), "took 2.5s");
  assert.equal(L.queueDetail({ status: "running", started_at: 10 }, -1, 85), "running 1m 15s");
  assert.equal(L.formatBytes(1536 * 1024 * 1024), "1.5 GB");
});

test("the CPU device label never says 'torch null' (UI3)", () => {
  assert.equal(L.cpuDeviceLabel({ test_mode: true, torch: null }), "CPU · test mode (no models)");
  assert.equal(L.cpuDeviceLabel({ torch: null }), "CPU · PyTorch not available");
  assert.equal(L.cpuDeviceLabel({ torch: "2.14.1", cuda: false }), "CPU · torch 2.14.1 (no CUDA)");
});

test("canvas zoom steps and label (UI5)", () => {
  assert.equal(L.zoomPercent(0.756), "76%");
  assert.equal(L.zoomStep(1, 1), 1.25);
  assert.equal(L.zoomStep(1, -1), 0.75);
  assert.equal(L.zoomStep(0.8, -1), 0.75);
  assert.equal(L.zoomStep(0.8, 1), 1);
  assert.equal(L.zoomStep(3, 1), 3);
  assert.equal(L.zoomStep(0.1, -1), 0.1);
});

test("minimap layout fits nodes and the view, and maps clicks back (UI5)", () => {
  const nodes = [{ x: 0, y: 0, w: 100, h: 50 }, { x: 300, y: 200, w: 100, h: 100 }];
  const layout = L.minimapLayout(nodes, [-100, -50, 200, 100], 200, 140, 0);
  // Bounds -100..400 x -50..300: 500 x 350, so the scale is min(200/500, 140/350) = 0.4.
  assert.ok(Math.abs(layout.scale - 0.4) < 1e-9);
  const [gx, gy] = L.minimapToGraph(layout, 100, 70); // the map's centre
  assert.ok(Math.abs(gx - 150) < 1e-9 && Math.abs(gy - 125) < 1e-9);
  const empty = L.minimapLayout([], [0, 0, 800, 600], 200, 140);
  assert.ok(empty.scale > 0);
});

test("app mode fields, usual inputs and the latest result (UI8)", () => {
  assert.deepEqual(L.toggledAppInputs(undefined, "text"), ["text"]);
  assert.deepEqual(L.toggledAppInputs(["text", "seed"], "text"), ["seed"]);
  const node = (id, comfyClass, widgets, exposed) => ({ id, comfyClass, title: comfyClass, widgets: widgets.map((name) => ({ name })),
    properties: exposed ? { [L.APP_INPUTS_PROPERTY]: exposed } : {} });
  const prompt = node(1, "CLIPTextEncode", ["text", "clip_skip"], ["text"]);
  const sampler = node(2, "KSampler", ["seed", "steps", "cfg"], ["seed", "gone"]);
  const save = node(3, "SaveImage", ["filename_prefix"]);
  assert.deepEqual(L.appFields([prompt, sampler, save]).map((f) => f.label), ["CLIPTextEncode · text", "KSampler · seed"]);
  assert.deepEqual(L.usualAppInputs([prompt, sampler, save]), { 1: ["text"], 2: ["seed", "steps"] });
  const items = [{ promptId: "a", filename: "1.png" }, { promptId: "b", filename: "2.png" }, { promptId: "b", filename: "3.png" }];
  assert.deepEqual(L.latestPromptItems(items).map((i) => i.filename), ["2.png", "3.png"]);
  assert.deepEqual(L.latestPromptItems([]), []);
  const twin = node(4, "CLIPTextEncode", ["text"], ["text"]);
  assert.deepEqual(L.appFields([prompt, twin]).map((f) => f.label), ["CLIPTextEncode #1 · text", "CLIPTextEncode #4 · text"]);
});

test("template thumbnails draw the graph fitted into the box (UI7)", () => {
  const spec = {
    nodes: [{ id: "a", type: "LoadCheckpoint", pos: [0, 0] }, { id: "b", type: "KSampler", pos: [760, 240] }],
    links: [["a", 0, "b", "model"], ["a", 0, "missing", "x"]],
  };
  const { boxes, lines } = L.templateDiagram(spec, 100, 50, 0);
  // Span 1000 x 360: the scale is min(100/1000, 50/360) = 0.1, centred vertically.
  assert.equal(boxes.length, 2);
  assert.ok(boxes.every((box) => box.x >= 0 && box.y >= 0 && box.x + box.w <= 100.0001 && box.y + box.h <= 50.0001));
  assert.ok(Math.abs(boxes[1].x - 76) < 1e-9);
  assert.equal(lines.length, 1); // a link to a node that isn't there is dropped
  assert.deepEqual(lines[0].map((v) => Math.round(v)), [24, 13, 76, 37]);
  assert.deepEqual(L.templateDiagram({ nodes: [] }, 100, 50), { boxes: [], lines: [] });
});

test("workflow tabs: labels, ids, closing, storage (UI6)", () => {
  const graphWith = (n) => JSON.stringify({ nodes: Array.from({ length: n }, (_, i) => ({ id: i })) });
  const a = { id: "t1", name: "portrait", graph: graphWith(2), saved: graphWith(2) };
  const b = { id: "t2", name: "", graph: graphWith(1), saved: null };
  const c = { id: "t3", name: "", graph: graphWith(0), saved: null };
  assert.equal(L.tabIsDirty(a), false);
  assert.equal(L.tabIsDirty(b), true);
  assert.equal(L.tabIsDirty(c), false); // an empty graph has nothing to lose
  assert.equal(L.tabLabel(a, false), "portrait");
  assert.equal(L.tabLabel(b, true), "Untitled •");
  assert.equal(L.newTabId([a, b]), "t3");
  assert.equal(L.newTabId([b]), "t1");
  assert.equal(L.tabAfterClose([a, b, c], "t2"), "t3");
  assert.equal(L.tabAfterClose([a, b, c], "t3"), "t2");
  assert.equal(L.tabAfterClose([a], "t1"), null);
  const stored = L.parseStoredTabs(L.storeTabs([a, { ...b, undoStack: ["x"] }], "t2"));
  assert.deepEqual(stored, { active: "t2", tabs: [{ ...a, hint: "" }, { ...b, hint: "" }] });
  // UI10: an unsaved graph is labelled by what it was loaded from, and the label survives a reload.
  const fromTemplate = { ...b, hint: "Text to image" };
  assert.equal(L.tabLabel(fromTemplate, true), "Text to image •");
  assert.equal(L.tabLabel({ ...fromTemplate, name: "mine" }, false), "mine");
  assert.equal(L.parseStoredTabs(L.storeTabs([fromTemplate], "t2")).tabs[0].hint, "Text to image");
  assert.equal(L.parseStoredTabs(L.storeTabs([a], "gone")).active, "t1");
  assert.equal(L.parseStoredTabs("not json"), null);
  assert.equal(L.parseStoredTabs(JSON.stringify({ tabs: [{ id: 3 }] })), null);
});

test("progress text and overall fraction (F10)", () => {
  const text = L.progressText({ nodeIndex: 2, nodeCount: 4, title: "KSampler", step: 5, total: 20, itPerS: 3.14159, etaS: 4.8 });
  assert.equal(text, "Node 2/4 · KSampler · step 5/20 · 3.14 it/s · ~4.8s left");
  assert.equal(
    L.progressText({ step: 512 * 1024 * 1024, total: 2 * 1024 ** 3, unit: "bytes" }),
    "downloading 512 MB / 2.0 GB",
  );
  assert.equal(L.overallProgress(2, 4, 10, 20), 0.375);
  assert.equal(L.overallProgress(0, 0, 5, 10), 0.5);
});

test("gallery items carry type and sampler metadata (F9)", () => {
  const history = {
    p: {
      status: "completed",
      prompt: {
        c: { class_type: "LoadCheckpoint", inputs: { ckpt_name: "m.safetensors" } },
        t: { class_type: "CLIPTextEncode", inputs: { text: "a cat" } },
        s: { class_type: "KSampler", inputs: { model: ["c", 0], positive: ["t", 0], seed: 9, steps: 4, cfg: 0, sampler: "euler" } },
      },
      outputs: { s: { seed: 9 }, v: { images: ["prev.png"], type: "temp" }, o: { images: ["sub/out.png"], type: "output" } },
    },
  };
  const items = L.galleryItems(history);
  assert.deepEqual(items.map((i) => [i.filename, i.type]), [["prev.png", "temp"], ["sub/out.png", "output"]]);
  assert.deepEqual(items[0].meta, { sampler: "euler", seed: 9, steps: 4, cfg: 0, positive: "a cat", model: "m.safetensors" });
  assert.equal(L.imageUrl("sub/a b.png", "output"), "/outputs/sub/a%20b.png");
  assert.equal(L.imageUrl("p.png", "temp"), "/temp/p.png");
  assert.equal(L.stepIndex(0, -1, 3), 2);
  assert.equal(L.stepIndex(2, 1, 3), 0);
  assert.deepEqual(L.capItems([1, 2, 3, 4], 2), [3, 4]);
});

test("shortcuts never fire while typing; the shortcut table (F6/F14)", () => {
  assert.ok(L.isTypingTarget({ tagName: "INPUT", type: "text" }));
  assert.ok(L.isTypingTarget({ tagName: "textarea" }));
  assert.ok(L.isTypingTarget({ tagName: "DIV", isContentEditable: true }));
  assert.ok(!L.isTypingTarget({ tagName: "INPUT", type: "checkbox" }));
  assert.ok(!L.isTypingTarget({ tagName: "CANVAS" }));
  assert.equal(L.shortcutFor({ ctrlKey: true, key: "Enter" }), "queue");
  assert.equal(L.shortcutFor({ ctrlKey: true, shiftKey: true, key: "Enter" }), "queue-front");
  assert.equal(L.shortcutFor({ ctrlKey: true, key: "z" }), "undo");
  assert.equal(L.shortcutFor({ ctrlKey: true, shiftKey: true, key: "Z" }), "redo");
  assert.equal(L.shortcutFor({ metaKey: true, key: "s" }), "save");
  assert.equal(L.shortcutFor({ ctrlKey: true, key: "b" }), "bypass");
  assert.equal(L.shortcutFor({ key: "?" }), "help");
  assert.equal(L.shortcutFor({ key: "a" }), null);
});

test("undo history is capped and skips duplicate snapshots (F6)", () => {
  let [stack, index] = [[], -1];
  for (let i = 0; i < 5; i++) [stack, index] = L.pushUndo(stack, index, `s${i}`, 3);
  assert.deepEqual(stack, ["s2", "s3", "s4"]);
  assert.equal(index, 2);
  [stack, index] = L.pushUndo(stack, index, "s4", 3);
  assert.equal(stack.length, 3);
  assert.equal(index, 2);
  [stack, index] = L.pushUndo(stack, 0, "branch", 3);
  assert.deepEqual(stack, ["s2", "branch"]);
  assert.equal(index, 1);
});

test("key combos, the new keys and rebinding (UI14)", () => {
  assert.equal(L.keyCombo({ ctrlKey: true, shiftKey: true, key: "Enter" }), "Ctrl+Shift+Enter");
  assert.equal(L.keyCombo({ shiftKey: true, key: "?" }), "?");  // Shift is part of the symbol
  assert.equal(L.keyCombo({ altKey: true, key: "ç", code: "KeyC" }), "Alt+C");
  assert.equal(L.keyCombo({ metaKey: true, key: "," }), "Ctrl+,");
  assert.equal(L.keyCombo({ key: "Shift" }), null);
  assert.equal(L.shortcutFor({ key: "p" }), "pin");
  assert.equal(L.shortcutFor({ key: "." }), "fit");
  assert.equal(L.shortcutFor({ altKey: true, key: "c", code: "KeyC" }), "collapse");
  assert.equal(L.shortcutFor({ key: "n" }), "toggle-library");
  assert.equal(L.shortcutFor({ key: "q" }), "toggle-queue");
  assert.equal(L.shortcutFor({ key: "w" }), "open");
  assert.equal(L.shortcutFor({ ctrlKey: true, key: "," }), "settings");
  assert.equal(L.shortcutFor({ shiftKey: true, key: "P" }), null);
  assert.equal(L.shortcutFor({ key: "Escape" }), "escape");

  // The defaults never share a key.
  assert.deepEqual(L.shortcutConflicts(L.shortcutKeys()), {});
  // Giving P to "fit" takes it from "pin"; the old "." no longer fits.
  const { overrides, takenFrom } = L.rebindShortcut({}, "fit", "P");
  assert.deepEqual(takenFrom, ["pin"]);
  assert.equal(L.shortcutFor({ key: "p" }, overrides), "fit");
  assert.equal(L.shortcutFor({ key: "." }, overrides), null);
  assert.deepEqual(L.shortcutKeys(overrides).pin, []);
  assert.deepEqual(L.shortcutConflicts(L.shortcutKeys(overrides)), {});
  assert.deepEqual(L.rebindShortcut(overrides, "fit", null).overrides.fit, []);
  assert.deepEqual(L.shortcutKeys({ bogus: ["X"] }).pin, ["P"]);  // unknown actions are ignored
});

test("prompt boxes wrap, grow from 3 to 10 lines, then scroll (UI11)", () => {
  const measure = (t) => t.length * 7;
  assert.deepEqual(L.wrapPromptLines("a photo of a cat", 70, measure), ["a photo of", "a cat"]);
  assert.deepEqual(L.wrapPromptLines("one\n\ntwo", 70, measure), ["one", "", "two"]);
  assert.deepEqual(L.wrapPromptLines("abcdefghijklmnop", 70, measure), ["abcdefghij", "klmnop"]); // a word wider than the box
  assert.deepEqual(L.wrapPromptLines("", 70, measure), [""]);
  const box = L.PROMPT_BOX;
  assert.equal(L.promptBoxHeight(1), L.promptBoxHeight(box.minLines));
  assert.ok(L.promptBoxHeight(6) > L.promptBoxHeight(3));
  assert.equal(L.promptBoxHeight(50), L.promptBoxHeight(box.maxLines));
  assert.deepEqual(L.promptScroll(box.maxLines, 40), { max: 0, scroll: 0 });
  assert.deepEqual(L.promptScroll(box.maxLines + 4, 999), { max: 4 * box.lineHeight, scroll: 4 * box.lineHeight });
  assert.equal(L.promptScroll(box.maxLines + 4, -5).scroll, 0);
});

test("the model library filters by every word and names models without folders or extensions (UI13)", () => {
  const library = {
    loras: { label: "LoRAs", node: "LoraLoader", input: "lora_name", items: [{ name: "styles/Ink Wash.safetensors" }, { name: "pixel.pt" }] },
    vae: { label: "VAEs", node: "LoadSplitModel", input: "vae", items: [{ name: "sdxl_vae.safetensors" }] },
  };
  assert.deepEqual(L.filterModelLibrary(library, "").map((g) => g.items.length), [2, 1]);
  const found = L.filterModelLibrary(library, "ink WASH");
  assert.deepEqual(found.map((g) => [g.kind, g.node, g.items.map((i) => i.name)]), [["loras", "LoraLoader", ["styles/Ink Wash.safetensors"]]]);
  assert.deepEqual(L.filterModelLibrary(library, "nothing"), []);
  assert.equal(L.modelDisplayName("styles/Ink Wash.safetensors"), "Ink Wash");
  assert.equal(L.modelDisplayName("controlnet-canny"), "controlnet-canny");
});

test("subgraphs are inlined before /prompt, wires joined through their inputs and outputs (UI12)", () => {
  const node = (id, comfyClass, inputs = [], outputs = [], extra = {}) =>
    ({ id, mode: 0, kind: "node", comfyClass, properties: {}, inputs, outputs, ...extra });
  // Inside the inner subgraph (node 20 of the outer one): input "img" -> Blur -> output "out".
  const innerMost = {
    nodes: [
      { id: 1, kind: "input", name: "img", inputs: [], outputs: [{ type: "IMAGE" }] },
      node(2, "Blur", [{ name: "image", type: "IMAGE", link: 1 }], [{ type: "IMAGE" }]),
      { id: 3, kind: "output", name: "out", inputs: [{ name: "", type: "IMAGE", link: 2 }], outputs: [] },
    ],
    links: { 1: { origin_id: 1, origin_slot: 0 }, 2: { origin_id: 2, origin_slot: 0 } },
  };
  // The outer subgraph (node 7): input "image" -> nested subgraph 20 -> Resize -> output "result".
  const outer = {
    nodes: [
      { id: 10, kind: "input", name: "image", inputs: [], outputs: [{ type: "IMAGE" }] },
      { id: 20, kind: "subgraph", mode: 0, subgraph: innerMost, inputs: [{ name: "img", type: "IMAGE", link: 5 }],
        outputs: [{ name: "out", type: "IMAGE" }] },
      node(21, "ResizeImage", [{ name: "image", type: "IMAGE", link: 6 }], [{ type: "IMAGE" }]),
      { id: 30, kind: "output", name: "result", inputs: [{ name: "", type: "IMAGE", link: 7 }], outputs: [] },
    ],
    links: { 5: { origin_id: 10, origin_slot: 0 }, 6: { origin_id: 20, origin_slot: 0 }, 7: { origin_id: 21, origin_slot: 0 } },
  };
  const root = {
    nodes: [
      node(1, "LoadImage", [], [{ type: "IMAGE" }]),
      { id: 7, kind: "subgraph", mode: 0, subgraph: outer, inputs: [{ name: "image", type: "IMAGE", link: 100 }],
        outputs: [{ name: "result", type: "IMAGE" }] },
      node(8, "SaveImage", [{ name: "images", type: "IMAGE", link: 101 }]),
    ],
    links: { 100: { origin_id: 1, origin_slot: 0 }, 101: { origin_id: 7, origin_slot: 0 } },
  };
  const flat = L.flattenSubgraphs(root);
  assert.deepEqual(flat.nodes.map((n) => n.id).sort(), ["1", "7:20:2", "7:21", "8"]);
  const from = (id, input = 0) => {
    const n = flat.nodes.find((x) => x.id === id);
    return flat.links[n.inputs[input].link].origin_id;
  };
  assert.equal(from("7:20:2"), "1"); // LoadImage -> into two subgraphs -> Blur
  assert.equal(from("7:21"), "7:20:2"); // Blur -> out of the inner subgraph -> Resize
  assert.equal(from("8"), "7:21"); // Resize -> out of the outer subgraph -> Save

  // A muted subgraph mutes everything inside it.
  root.nodes[1].mode = L.MODE_MUTE;
  assert.ok(L.flattenSubgraphs(root).nodes.filter((n) => n.id.startsWith("7:")).every((n) => n.mode === L.MODE_MUTE));
  assert.deepEqual(L.uniqueNames(["image", "image", "", "model"]), ["image", "image 2", "value", "model"]);
});
