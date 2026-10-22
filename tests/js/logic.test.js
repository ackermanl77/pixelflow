// Unit tests for frontend/logic.js. Run with `node --test tests/js`
// (tests/test_frontend_logic.py runs this from pytest when Node is installed).
const test = require("node:test");
const assert = require("node:assert/strict");
const L = require("../../frontend/logic.js");

test("every non-primitive type is a socket; BOOLEAN is a toggle (F4)", () => {
  assert.equal(L.widgetKind(["IMAGE"]), "socket");
  assert.equal(L.widgetKind(["MASK"]), "socket");
  assert.equal(L.widgetKind(["SOME_FUTURE_TYPE"]), "socket");
  assert.equal(L.widgetKind(["BOOLEAN", { default: true }]), "toggle");
  assert.equal(L.widgetKind(["INT", {}]), "number");
  assert.equal(L.widgetKind(["FLOAT", {}]), "number");
  assert.equal(L.widgetKind(["STRING", {}]), "text");
  assert.equal(L.widgetKind([["a", "b"], {}]), "combo");
});

test("litegraph gets 10x the real step, since it moves by 0.1*step (F3)", () => {
  assert.equal(L.numberWidgetOptions("INT", {}).step, 10);
  assert.equal(L.numberWidgetOptions("INT", {}).precision, 0);
  assert.equal(L.numberWidgetOptions("FLOAT", {}).step, 1);
  assert.ok(Math.abs(L.numberWidgetOptions("FLOAT", { step: 0.01 }).step - 0.1) < 1e-12);
  const opts = L.numberWidgetOptions("INT", { min: 1, max: 150 });
  assert.equal(opts.min, 1);
  assert.equal(opts.max, 150);
});

test("INT values are rounded and clamped, never fractional (F3)", () => {
  assert.equal(L.normalizeNumber(4.1, "INT", { min: 1, max: 150 }), 4);
  assert.equal(L.normalizeNumber(4.6, "INT", {}), 5);
  assert.equal(L.normalizeNumber(-0.9, "INT", { min: -1 }), -1);
  assert.equal(L.normalizeNumber(-1.2, "INT", { min: -1 }), -1);
  assert.equal(L.normalizeNumber(999, "INT", { max: 150 }), 150);
  assert.equal(L.normalizeNumber("12", "INT", {}), 12);
  assert.equal(L.normalizeNumber("abc", "INT", { default: 20 }), 20);
  assert.ok(Number.isInteger(L.normalizeNumber(7.4999, "INT", {})));
});

test("FLOAT values lose binary noise, snap to `round`, and clamp", () => {
  assert.equal(L.normalizeNumber(0.1 + 0.2, "FLOAT", {}), 0.3);
  assert.equal(L.normalizeNumber(0.37, "FLOAT", { round: 0.05 }), 0.35);
  assert.equal(L.normalizeNumber(1.7, "FLOAT", { min: 0, max: 1 }), 1);
  assert.equal(L.normalizeNumber(-0.00001, "FLOAT", {}), 0);
});

test("structured 400 details are unpacked (F5)", () => {
  assert.deepEqual(
    L.parseErrorDetail({ message: "Node '3': bad", node_id: "3", class_type: "ImageBlur", input: "radius" }),
    { message: "Node '3': bad", nodeId: "3", input: "radius" },
  );
  assert.deepEqual(L.parseErrorDetail("plain"), { message: "plain", nodeId: null, input: null });
  assert.equal(L.parseErrorDetail([{ msg: "field required" }]).message, "field required");
});

test("reconnect backs off exponentially up to 10s (F1)", () => {
  assert.equal(L.reconnectDelay(0), 500);
  assert.equal(L.reconnectDelay(1), 1000);
  assert.equal(L.reconnectDelay(3), 4000);
  assert.equal(L.reconnectDelay(50), 10000);
});

test("queue rows and gallery are rebuilt from server history (F1)", () => {
  const history = {
    a: { status: "completed", outputs: { s: { seed: 5 }, save: { images: ["a1.png", "a2.png"] } } },
    b: { status: "error", error: "boom", class_type: "KSampler" },
    c: { status: "running" },
    d: { status: "queued" },
  };
  const rows = L.queueRows(history, { running: "c", pending: ["d"] });
  assert.deepEqual(rows.map((r) => r.statusClass), ["status-completed", "status-error", "status-running", "status-queued"]);
  assert.equal(rows[0].text, "Done: a (seed 5)");
  assert.equal(rows[1].text, "Error: b — KSampler — boom");
  assert.deepEqual(L.galleryImages(history), [
    { filename: "a1.png", promptId: "a" },
    { filename: "a2.png", promptId: "a" },
  ]);
});

test("only image files are accepted for drop/paste (F2)", () => {
  assert.ok(L.isImageFile({ type: "image/png" }));
  assert.ok(!L.isImageFile({ type: "application/json" }));
  assert.ok(!L.isImageFile(null));
});

test("Ctrl+G groups; groupBounds wraps the boxes with padding and a title band", () => {
  assert.equal(L.shortcutFor({ ctrlKey: true, key: "g" }), "group");
  assert.equal(L.groupBounds([]), null);
  assert.deepEqual(L.groupBounds([[100, 50, 200, 80], [400, 200, 100, 60]], 10, 34), [90, 6, 420, 264]);
});

test("side panel widths stay within 160-520 px and a third of the window (F17)", () => {
  assert.equal(L.clampPanelWidth(300), 300);
  assert.equal(L.clampPanelWidth(50), 160);
  assert.equal(L.clampPanelWidth(900), 520);
  assert.equal(L.clampPanelWidth(500, 1200), 400);
  assert.equal(L.clampPanelWidth("junk"), 160);
});

test("Reroutes are followed and Primitives inlined into the inputs they drive (F18)", () => {
  const nodes = [
    { id: 1, mode: 0, comfyClass: "EmptyImage", properties: { width: 512 }, inputs: [], outputs: [{ type: "IMAGE" }] },
    { id: 2, mode: 0, comfyClass: "Reroute", properties: {}, inputs: [{ name: "", type: "*", link: 10 }], outputs: [{ type: "*" }] },
    { id: 3, mode: 0, comfyClass: "Reroute", properties: {}, inputs: [{ name: "", type: "*", link: 11 }], outputs: [{ type: "*" }] },
    { id: 4, mode: 0, comfyClass: "Primitive", properties: { value: "1234" }, inputs: [], outputs: [{ type: "*" }] },
    {
      id: 5, mode: 0, comfyClass: "KSamplerImage",
      properties: { seed: 7, steps: 20 },
      inputs: [{ name: "image", type: "IMAGE", link: 12 }, { name: "seed", type: "INT", link: 13 }, { name: "steps", type: "INT", link: null }],
      outputs: [{ type: "IMAGE" }],
    },
  ];
  const links = {
    10: { origin_id: 1, origin_slot: 0 },
    11: { origin_id: 2, origin_slot: 0 },
    12: { origin_id: 3, origin_slot: 0 },
    13: { origin_id: 4, origin_slot: 0 },
  };
  const prompt = L.buildPrompt(nodes, links);
  assert.deepEqual(Object.keys(prompt).sort(), ["1", "5"]); // frontend-only nodes never reach the backend
  assert.deepEqual(prompt["5"].inputs, { seed: 1234, steps: 20, image: ["1", 0] });
  // A reroute with nothing upstream drops the input like a missing link.
  nodes[1].inputs[0].link = null;
  assert.equal("image" in L.buildPrompt(nodes, links)["5"].inputs, false);
});

test("coercePrimitive turns a Primitive's text into the driven widget's type", () => {
  assert.equal(L.coercePrimitive("42.9", "INT"), 42);
  assert.equal(L.coercePrimitive("x", "INT"), 0);
  assert.equal(L.coercePrimitive("0.35", "FLOAT"), 0.35);
  assert.equal(L.coercePrimitive("True", "BOOLEAN"), true);
  assert.equal(L.coercePrimitive(false, "BOOLEAN"), false);
  assert.equal(L.coercePrimitive(12, "STRING"), "12");
});

test("X/Y plot axis values: lists, numeric ranges and typing (F18)", () => {
  assert.deepEqual(L.parseAxisValues("4, 8 ,20", "INT"), [4, 8, 20]);
  assert.deepEqual(L.parseAxisValues("1-9:2", "INT"), [1, 3, 5, 7, 9]);
  assert.deepEqual(L.parseAxisValues("0.2-0.6:0.2", "FLOAT"), [0.2, 0.4, 0.6]);
  assert.deepEqual(L.parseAxisValues("euler, dpm++", "COMBO"), ["euler", "dpm++"]);
  assert.deepEqual(L.parseAxisValues("true,false", "BOOLEAN"), [true, false]);
  assert.deepEqual(L.parseAxisValues("-2-2", "INT"), [-2, -1, 0, 1, 2]);
  assert.throws(() => L.parseAxisValues("", "INT"));
  assert.throws(() => L.parseAxisValues("4, x", "INT"));
  assert.throws(() => L.parseAxisValues("9-1", "INT"));
});

test("firstImage prefers saved outputs over previews", () => {
  assert.equal(L.firstImage({}), null);
  assert.deepEqual(L.firstImage({ 3: { images: ["p.png"], type: "temp" }, 5: { images: ["s.png", "t.png"], type: "output" } }), { filename: "s.png", type: "output" });
  assert.deepEqual(L.firstImage({ 3: { images: ["p.png"], type: "temp" } }), { filename: "p.png", type: "temp" });
});

test("a window shows its own prompts, window-less ones, or all of them", () => {
  assert.equal(L.isOwnPrompt({ client_id: "a" }, "a"), true);
  assert.equal(L.isOwnPrompt({ client_id: "b" }, "a"), false);
  assert.equal(L.isOwnPrompt({ status: "completed" }, "a"), true); // API / old history: everyone's
  assert.equal(L.isOwnPrompt(undefined, "a"), true); // not known yet: assume ours
  const history = { p1: { client_id: "a" }, p2: { client_id: "b" }, p3: {} };
  assert.deepEqual(Object.keys(L.visibleHistory(history, "a", false)), ["p1", "p3"]);
  assert.deepEqual(Object.keys(L.visibleHistory(history, "a", true)), ["p1", "p2", "p3"]);
  assert.deepEqual(L.visibleHistory(null, "a", false), {});
});

test("the System panel describes custom node plugins (F22)", () => {
  assert.equal(L.describeCustomNodes([]), "none (data folder's custom_nodes)");
  assert.equal(L.describeCustomNodes(undefined), "none (data folder's custom_nodes)");
  assert.equal(
    L.describeCustomNodes([
      { name: "tools", nodes: ["A", "B"], error: null },
      { name: "one", nodes: ["C"], error: null },
      { name: "broken", nodes: [], error: "Traceback ..." },
    ]),
    "tools: 2 nodes; one: 1 node; broken: failed, see the log",
  );
});

test("a saved workflow carries its API prompt, stripped again on load (F24)", () => {
  const serialized = { nodes: [], links: [], extra: { ds: { scale: 1 } } };
  const prompt = { 3: { class_type: "SaveImage", inputs: {} } };
  const saved = L.withSavedPrompt(serialized, prompt);
  assert.deepEqual(saved.extra, { ds: { scale: 1 }, pixelflow_prompt: prompt });
  assert.deepEqual(serialized.extra, { ds: { scale: 1 } }, "the live graph's data is not modified");
  assert.deepEqual(L.withoutSavedPrompt(saved), serialized);
  assert.deepEqual(L.withSavedPrompt({ nodes: [] }, {}).extra, { pixelflow_prompt: {} });
  const old = { nodes: [], extra: { ds: {} } };
  assert.equal(L.withoutSavedPrompt(old), old);
});

test("download list lines (F23)", () => {
  const base = { url: "https://civitai.com/models/1", filename: "style.safetensors", folder: "loras" };
  assert.equal(L.describeDownload({ ...base, filename: null, folder: null, status: "resolving" }),
    "https://civitai.com/models/1: finding the file…");
  assert.equal(L.describeDownload({ ...base, status: "downloading", bytes: 512 * 1024 * 1024, total: 2048 * 1024 * 1024 }),
    "style.safetensors → loras: 25% (512 MB / 2.0 GB)");
  assert.equal(L.describeDownload({ ...base, status: "downloading", bytes: 1024, total: null }),
    "style.safetensors → loras: (1.0 KB)");
  assert.equal(L.describeDownload({ ...base, status: "done", verified: true }),
    "style.safetensors → loras: done, SHA-256 verified");
  assert.equal(L.describeDownload({ ...base, status: "error", error: "SHA-256 doesn't match" }),
    "style.safetensors: failed — SHA-256 doesn't match");
});

test("a node widget's label and value never overlap: the fit width covers both (UI2)", () => {
  const measure = (text) => text.length * 7; // 7 px per character
  const widgets = [
    { type: "combo", name: "ckpt_name", value: "stabilityai/sd-turbo" },
    { type: "number", name: "steps", value: 4, options: { precision: 0 } },
    { type: "button", name: "Upload" },
  ];
  // combo: (9 + 20) * 7 + 95 = 298
  assert.equal(L.widgetFitWidth(widgets, measure), 298);
  assert.equal(L.widgetDisplayValue({ type: "number", value: 0.5, options: {} }), "0.500");
  assert.equal(L.widgetDisplayValue({ type: "toggle", value: true, options: { on: "yes" } }), "yes");
  assert.equal(L.widgetDisplayValue({ type: "text", value: "x".repeat(40) }).length, 30);
  // A very long value is capped instead of making the node enormous.
  assert.equal(L.widgetFitWidth([{ type: "combo", name: "m", value: "y".repeat(200) }], measure), L.WIDGET_MAX_WIDTH);
});
