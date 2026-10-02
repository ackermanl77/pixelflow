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
