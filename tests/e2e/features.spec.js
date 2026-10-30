// Browser checks for the P3 features (F14/F17/F18): templates, groups, panels,
// reroute/primitive nodes and the mask editor. Same test-mode backend as smoke.spec.js.
const { test, expect } = require("@playwright/test");

/** Click a topbar command that lives in a menu (UI4): open its menu first. */
async function menuClick(page, selector) {
  await page.locator(".menu").filter({ has: page.locator(selector) }).locator(".menu-button").click();
  await page.click(selector);
}

// The page's own globals, used inside page.evaluate() callbacks (they run in the browser).
/* global CLIENT_ID, queueState, galleryItems, Logic, convertToSubgraph, openSubgraph, leaveSubgraphs */

test.beforeEach(async ({ page }) => {
  page.errors = [];
  page.on("pageerror", (err) => page.errors.push(err.message));
  await page.goto("/");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");
});

test.afterEach(async ({ page }) => {
  expect(page.errors).toEqual([]);
});

test("templates: every one builds its graph, wired as its file says", async ({ page, request }) => {
  const { templates } = await (await request.get("/templates/index.json")).json();
  for (const template of templates) {
    const spec = await (await request.get(`/templates/${template.file}`)).json();
    await menuClick(page, "#btn-templates");
    await expect(page.locator("#template-overlay")).toBeVisible();
    await page.locator(`.template-item[data-file="${template.file}"]`).click();
    // A dirty graph asks first (the app's own modal), which can open a moment after the click:
    // wait for either outcome -- the question, or the template dialog closing -- before checking.
    await page.waitForFunction(() =>
      document.getElementById("template-overlay").classList.contains("hidden") ||
      !document.getElementById("modal-overlay").classList.contains("hidden"));
    const confirm = page.locator("#modal-overlay:not(.hidden) .primary");
    if (await confirm.count()) await confirm.click();
    await expect(page.locator("#template-overlay")).toBeHidden();
    await expect.poll(() => page.evaluate(() => graph._nodes.length)).toBe(spec.nodes.length);
    const linkCount = await page.evaluate(() => Object.keys(graph.links).length);
    expect(linkCount, template.file).toBe(spec.links.length);
    const prompt = await page.evaluate(() => buildPrompt());
    expect(Object.keys(prompt).length, template.file).toBe(spec.nodes.length);
  }
  // Load Example is the text-to-image template.
  await menuClick(page, "#btn-example");
  const confirm = page.locator("#modal-overlay:not(.hidden) .primary");
  if (await confirm.count()) await confirm.click();
  await expect.poll(() => page.evaluate(() => graph._nodes.map((n) => n.comfyClass).join(","))).toBe(
    "LoadCheckpoint,CLIPTextEncode,EmptyImage,KSampler,SaveImage",
  );
});

test("groups: Ctrl+G wraps the selection, and the group menu mutes and bypasses it", async ({ page }) => {
  await menuClick(page, "#btn-example");
  await expect.poll(() => page.evaluate(() => graph._nodes.length)).toBe(5);
  await page.evaluate(() => {
    const picked = graph._nodes.filter((n) => ["CLIPTextEncode", "EmptyImage"].includes(n.comfyClass));
    canvas.selectNodes(picked);
  });
  await page.locator("#graph-canvas").focus();
  await page.keyboard.press("Control+g");
  const group = await page.evaluate(() => {
    const g = graph._groups[0];
    g.recomputeInsideNodes();
    return { count: graph._groups.length, inside: g._nodes.map((n) => n.comfyClass).sort() };
  });
  expect(group).toEqual({ count: 1, inside: ["CLIPTextEncode", "EmptyImage"] });

  // The group's context menu (Edit Group) leads with Mute / Bypass / Select.
  const labels = await page.evaluate(() => canvas.getGroupMenuOptions(graph._groups[0]).filter(Boolean).map((o) => o.content));
  expect(labels.slice(0, 3)).toEqual(["Mute group", "Bypass group", "Select nodes"]);
  await page.evaluate(() => canvas.getGroupMenuOptions(graph._groups[0])[0].callback());
  const modes = await page.evaluate(() => graph._nodes.map((n) => [n.comfyClass, n.mode]));
  for (const [cls, mode] of modes) expect(mode, cls).toBe(["CLIPTextEncode", "EmptyImage"].includes(cls) ? 2 : 0);

  // Groups are part of the saved graph.
  const saved = await page.evaluate(() => graph.serialize().groups.length);
  expect(saved).toBe(1);
});

test("side panels resize and collapse, and the canvas follows; the layout is remembered", async ({ page }) => {
  const width = (sel) => page.evaluate((s) => document.querySelector(s).getBoundingClientRect().width, sel);
  const canvasWidth = () => page.evaluate(() => document.getElementById("graph-canvas").width);
  const before = await width("#sidebar");
  const splitter = page.locator('.splitter[data-panel="sidebar"]');
  await splitter.focus();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowRight");
  expect(await width("#sidebar")).toBe(before + 40);

  // Drag the right panel's splitter 100 px to the left: that panel widens.
  const right = page.locator('.splitter[data-panel="rightpanel"]');
  const box = await right.boundingBox();
  const rightBefore = await width("#rightpanel");
  await page.mouse.move(box.x + 3, box.y + 40);
  await page.mouse.down();
  await page.mouse.move(box.x - 97, box.y + 40, { steps: 5 });
  await page.mouse.up();
  expect(await width("#rightpanel")).toBe(rightBefore + 100);

  const canvasBefore = await canvasWidth();
  await splitter.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#sidebar")).toBeHidden();
  expect(await canvasWidth()).toBeGreaterThan(canvasBefore);

  await page.reload();
  await expect(page.locator("#sidebar")).toBeHidden();
  expect(await width("#rightpanel")).toBe(rightBefore + 100);
  await page.locator('.splitter[data-panel="sidebar"] .splitter-toggle').click();
  await expect(page.locator("#sidebar")).toBeVisible();
  expect(await width("#sidebar")).toBe(before + 40);
});

test("reroute and primitive nodes: a primitive drives a converted widget through the real run", async ({ page }) => {
  // The palette lists the two frontend-only nodes under "utils".
  await expect(page.locator("#node-list .node-item", { hasText: /^Reroute$/ })).toBeVisible();
  await expect(page.locator("#node-list .node-item", { hasText: /^Primitive$/ })).toBeVisible();

  const prompt = await page.evaluate(() => {
    const empty = addNodeToGraph("EmptyImage", [60, 100]);
    const reroute = addNodeToGraph("Reroute", [380, 140]);
    const save = addNodeToGraph("SaveImage", [560, 100]);
    const primitive = addNodeToGraph("Primitive", [60, 380]);
    // Right-click menu: "Convert width to input".
    const option = empty.getExtraMenuOptions().find((o) => o && o.content === "Convert width to input");
    option.callback();
    empty.connect(0, reroute, 0);
    reroute.connect(0, save, findInputSlot(save, "image"));
    primitive.connect(0, empty, findInputSlot(empty, "width"));
    setNodeValue(primitive, "value", "200");
    return buildPrompt();
  });
  const classes = Object.values(prompt).map((n) => n.class_type).sort();
  expect(classes).toEqual(["EmptyImage", "SaveImage"]);
  const empty = Object.values(prompt).find((n) => n.class_type === "EmptyImage");
  expect(empty.inputs.width).toBe(200);
  const save = Object.values(prompt).find((n) => n.class_type === "SaveImage");
  expect(save.inputs.image[0]).toBe(Object.keys(prompt).find((id) => prompt[id] === empty));

  // It runs for real (Empty Image -> Save Image needs no model): the output is 200 px wide.
  await page.click("#btn-queue");
  const img = page.locator("#gallery .gallery-item img").first();
  await expect(img).toBeVisible({ timeout: 30_000 });
  await expect.poll(() => img.evaluate((el) => el.naturalWidth)).toBe(200);

  // A converted widget survives a reload of the graph: still an input, still linked, no widget.
  const restored = await page.evaluate(() => {
    const data = graph.serialize();
    graph.clear();
    graph.configure(data);
    syncWidgetsFromProperties();
    const node = graph._nodes.find((n) => n.comfyClass === "EmptyImage");
    return {
      input: node.inputs[findInputSlot(node, "width")].link !== null,
      widget: node.widgets.some((w) => w.name === "width"),
      width: buildPrompt()[String(node.id)].inputs.width,
    };
  });
  expect(restored).toEqual({ input: true, widget: false, width: 200 });
});

test("mask editor: paint, save, and the mask lands in a Load Mask wired to the inpaint sampler", async ({ page, request }) => {
  // A 300x200 grey PNG to mask, uploaded the way the Upload button does.
  const png = await page.evaluate(async () => {
    const c = document.createElement("canvas");
    c.width = 300;
    c.height = 200;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#777";
    ctx.fillRect(0, 0, 300, 200);
    const blob = await new Promise((r) => c.toBlob(r, "image/png"));
    return Array.from(new Uint8Array(await blob.arrayBuffer()));
  });
  const upload = await request.post("/api/upload", {
    multipart: { file: { name: "maskme.png", mimeType: "image/png", buffer: Buffer.from(png) } },
  });
  const { filename } = await upload.json();

  const ids = await page.evaluate(async (file) => {
    await refreshUploadChoices();
    const image = addNodeToGraph("LoadImage", [60, 80]);
    const sampler = addNodeToGraph("KSamplerInpaint", [480, 80]);
    setNodeValue(image, "filename", file);
    image.connect(0, sampler, findInputSlot(sampler, "image"));
    image.getExtraMenuOptions().find((o) => o && o.content === "Open in mask editor").callback();
    return { image: image.id, sampler: sampler.id };
  }, filename);
  await expect(page.locator("#mask-overlay")).toBeVisible();
  await expect.poll(() => page.evaluate(() => document.getElementById("mask-canvas").width)).toBe(300);

  // Paint a horizontal stroke across the middle with the real mouse.
  const box = await page.locator("#mask-canvas").boundingBox();
  await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.8, box.y + box.height * 0.5, { steps: 10 });
  await page.mouse.up();
  await page.click("#mask-save");
  await expect(page.locator("#mask-overlay")).toBeHidden();

  const result = await page.evaluate((idsArg) => {
    const image = graph.getNodeById(idsArg.image);
    const mask = graph._nodes.find((n) => n.comfyClass === "LoadMask");
    const sampler = graph.getNodeById(idsArg.sampler);
    const link = graph.links[sampler.inputs[findInputSlot(sampler, "mask")].link];
    return { file: mask.properties.filename, linked: link && link.origin_id === mask.id, remembered: image.properties._mask_node === mask.id };
  }, ids);
  expect(result.linked && result.remembered).toBe(true);
  expect(result.file).toMatch(/mask/);

  // The saved PNG is the image's size: white under the stroke, black elsewhere.
  const pixels = await page.evaluate(async (file) => {
    const img = new Image();
    img.src = `/uploads/${encodeURIComponent(file)}`;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const at = (x, y) => ctx.getImageData(x, y, 1, 1).data[0];
    return { size: [c.width, c.height], center: at(150, 100), corner: at(5, 5), belowStroke: at(150, 190) };
  }, result.file);
  expect(pixels).toEqual({ size: [300, 200], center: 255, corner: 0, belowStroke: 0 });
});

test("X/Y plot: a 2x2 grid runs for real and its contact sheet opens", async ({ page }) => {
  await page.evaluate(() => {
    const empty = addNodeToGraph("EmptyImage", [60, 100]);
    const save = addNodeToGraph("SaveImage", [420, 100]);
    empty.connect(0, save, findInputSlot(save, "image"));
  });
  await menuClick(page, "#btn-xy");
  await expect(page.locator("#xy-overlay")).toBeVisible();
  const emptyId = await page.evaluate(() => String(graph._nodes.find((n) => n.comfyClass === "EmptyImage").id));
  await page.selectOption("#xy-x-node", emptyId);
  await page.selectOption("#xy-x-widget", "width");
  await page.fill("#xy-x-values", "64, 96");
  await page.selectOption("#xy-y-node", emptyId);
  await page.selectOption("#xy-y-widget", "color");
  await page.fill("#xy-y-values", "#ff0000, #0000ff");
  await page.click("#xy-run");
  await expect(page.locator("#lightbox-overlay")).toBeVisible({ timeout: 30_000 });
  const sheet = page.locator("#lightbox-overlay img").first();
  await expect.poll(() => sheet.evaluate((el) => el.naturalWidth)).toBeGreaterThan(2 * 96);
  expect(await sheet.getAttribute("src")).toContain("xy_plot");
  // The graph's own values are back after the grid was queued.
  const width = await page.evaluate(() => graph._nodes.find((n) => n.comfyClass === "EmptyImage").properties.width);
  expect(width).toBe(512);
});

test("per-window queue: another window's run stays out of this one until 'All windows'", async ({ page, context }) => {
  // A second tab of the same browser: its own sessionStorage, so its own client id.
  const other = await context.newPage();
  await other.goto("/");
  await expect(other.locator("#conn-status")).toHaveAttribute("data-state", "connected");
  await other.evaluate(() => {
    const empty = addNodeToGraph("EmptyImage", [60, 100]);
    const save = addNodeToGraph("SaveImage", [420, 100]);
    empty.connect(0, save, findInputSlot(save, "image"));
  });
  const ids = await Promise.all([page, other].map((p) => p.evaluate(() => CLIENT_ID)));
  expect(ids[0]).not.toBe(ids[1]);
  await other.click("#btn-queue");
  await expect(other.locator("#gallery .gallery-item img").first()).toBeVisible({ timeout: 30_000 });
  const ownRow = other.locator(".queue-entry:not([hidden])").first();
  await expect(ownRow).toBeVisible();

  // This window got the same messages, but shows neither the row nor the image.
  const promptId = await other.evaluate(() => [...queueState.entries.keys()].pop());
  await expect.poll(() => page.evaluate((id) => queueState.entries.get(id)?.status, promptId)).toBe("completed");
  expect(await page.evaluate((id) => queueState.rows.get(id).hidden, promptId)).toBe(true);
  expect(await page.evaluate((id) => galleryItems.some((item) => item.promptId === id), promptId)).toBe(false);

  await page.check("#queue-all-windows");
  expect(await page.evaluate((id) => queueState.rows.get(id).hidden, promptId)).toBe(false);
  await expect(page.locator(".queue-entry.foreign").first()).toContainText("another window");
  expect(await page.evaluate((id) => galleryItems.some((item) => item.promptId === id), promptId)).toBe(true);
  await page.uncheck("#queue-all-windows");
  await other.close();
});

test("shortcuts: pin, collapse, fit and panel keys; rebinding is remembered and Esc only cancels it (UI14)", async ({ page }) => {
  await menuClick(page, "#btn-example");
  await expect.poll(() => page.evaluate(() => graph._nodes.length)).toBe(5);
  await page.evaluate(() => canvas.selectNodes(graph._nodes.slice(0, 2)));
  await page.locator("#graph-canvas").focus();
  await page.keyboard.press("p");
  expect(await page.evaluate(() => graph._nodes.slice(0, 2).map((n) => !!n.flags.pinned))).toEqual([true, true]);
  await page.keyboard.press("Alt+c");
  expect(await page.evaluate(() => graph._nodes.slice(0, 2).map((n) => !!n.flags.collapsed))).toEqual([true, true]);
  await page.keyboard.press("Alt+c");
  expect(await page.evaluate(() => graph._nodes[0].flags.collapsed)).toBeFalsy();
  await page.keyboard.press("n");
  await expect(page.locator("#sidebar")).toHaveClass(/collapsed/);
  await page.keyboard.press("n");
  await expect(page.locator("#sidebar")).not.toHaveClass(/collapsed/);
  await page.keyboard.press("q");
  await expect(page.locator("#rightpanel")).toHaveClass(/collapsed/);
  await page.keyboard.press("q");

  // The "?" list is the shortcut table; give "Fit" the key P (taken from Pin).
  await page.keyboard.press("?");
  await expect(page.locator("#help-overlay")).toBeVisible();
  const rows = page.locator("#shortcut-table tr");
  await expect(rows).toHaveCount(await page.evaluate(() => Logic.SHORTCUTS.length + 2));
  await page.locator('.shortcut-change[data-action="pin"]').click();
  await page.keyboard.press("Escape"); // cancels the change, the dialog stays open
  await expect(page.locator("#help-overlay")).toBeVisible();
  await page.locator('.shortcut-change[data-action="fit"]').click();
  await page.keyboard.press("p");
  await expect(page.locator("#shortcut-message")).toContainText("no longer does: Pin");
  await expect(page.locator('.shortcut-change[data-action="fit"]')).toBeFocused();
  await page.click("#help-close");

  // After a reload the graph is restored, so no example is loaded (it would ask "Replace?").
  await page.reload();
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");
  await expect.poll(() => page.evaluate(() => graph._nodes.length)).toBe(5);
  await expect(page.locator("#modal-overlay")).toBeHidden();
  await page.evaluate(() => { canvas.ds.scale = 3; canvas.ds.offset[0] = 5000; canvas.setDirty(true, true); });
  await page.locator("#graph-canvas").focus();
  await page.keyboard.press("p"); // now Fit, after the reload
  expect(await page.evaluate(() => [canvas.ds.scale < 3, graph._nodes.some((n) => n.flags.pinned)])).toEqual([true, false]);

  await page.keyboard.press("?");
  await page.click("#shortcut-reset");
  await expect(page.locator("#shortcut-table tr").first()).toContainText("Ctrl+Enter");
  expect(await page.evaluate(() => localStorage.getItem("pixelflow.shortcuts"))).toBeNull();
});

test("prompt boxes: drawn on the node, edited in place, growing with the text and scrolling (UI11)", async ({ page }) => {
  await menuClick(page, "#btn-example");
  await expect.poll(() => page.evaluate(() => graph._nodes.length)).toBe(5);
  const info = () => page.evaluate(() => {
    const node = graph._nodes.find((n) => n.comfyClass === "CLIPTextEncode");
    const widget = node.widgets.find((w) => w.name === "text");
    return { id: node.id, type: widget.type, height: node.size[1], value: node.properties.text, scroll: widget.scroll };
  });
  const before = await info();
  expect(before.type).toBe("prompt");

  // Click the box: a textarea over it; Esc keeps the old text.
  const point = () => page.evaluate((id) => {
    const node = graph.getNodeById(id);
    const widget = node.widgets.find((w) => w.name === "text");
    const rect = canvas.canvas.getBoundingClientRect();
    const [x, y] = canvas.ds.convertOffsetToCanvas([node.pos[0] + node.size[0] / 2, node.pos[1] + widget.last_y + 30]);
    return { x: rect.left + x, y: rect.top + y };
  }, before.id);
  let at = await point();
  await page.mouse.click(at.x, at.y);
  const editor = page.locator("textarea.prompt-editor");
  await expect(editor).toBeVisible();
  await editor.fill("changed");
  await page.keyboard.press("Escape");
  await expect(editor).toHaveCount(0);
  expect((await info()).value).toBe(before.value);

  // Long text: kept on leaving the box, the node grows (to the cap), the wheel scrolls inside.
  at = await point();
  await page.mouse.click(at.x, at.y);
  const long = Array.from({ length: 30 }, (_, i) => `line ${i + 1} of a long prompt`).join("\n");
  await editor.fill(long);
  await page.keyboard.press("Control+Enter");
  const after = await info();
  expect(after.value).toBe(long);
  expect(after.height).toBeGreaterThan(before.height);
  at = await point();
  await page.mouse.move(at.x, at.y);
  await page.mouse.wheel(0, 200);
  expect((await info()).scroll).toBeGreaterThan(0);

  // It still reaches the backend as the node's text.
  const prompt = await page.evaluate(() => buildPrompt());
  expect(Object.values(prompt.prompt || prompt).some((n) => n.inputs && n.inputs.text === long)).toBe(true);
});

test("model library: installed models with size and preview; click or drag adds the loader with it (UI13)", async ({ page, request }) => {
  const fs = require("node:fs");
  const path = require("node:path");
  const { models_dir: modelsDir } = await (await request.get("/api/system")).json();
  const lora = path.join(modelsDir, "loras", "ink-wash.safetensors");
  fs.mkdirSync(path.dirname(lora), { recursive: true });
  fs.writeFileSync(lora, Buffer.alloc(2048));
  // A 1x1 PNG beside it is its preview.
  fs.writeFileSync(path.join(modelsDir, "loras", "ink-wash.png"), Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "base64"));

  await page.click("#tab-models");
  await expect(page.locator("#sidebar-models")).toBeVisible();
  await expect(page.locator("#sidebar-nodes")).toBeHidden();
  const card = page.locator(".model-item", { hasText: "ink-wash" });
  await expect(card).toBeVisible();
  await expect(card.locator(".model-size")).toHaveText("2.0 KB");
  await expect(card.locator("img.model-thumb")).toHaveJSProperty("naturalWidth", 1);
  await page.fill("#model-search", "nothing like it");
  await expect(page.locator("#model-list-empty")).toHaveText("No models match your search.");
  await page.fill("#model-search", "ink");

  // Click: a LoRA loader with this LoRA chosen.
  await card.click();
  const loaders = () => page.evaluate(() => graph._nodes.filter((n) => n.comfyClass === "LoraLoader")
    .map((n) => ({ lora: n.properties.lora_name, widget: n.widgets.find((w) => w.name === "lora_name").value, x: n.pos[0] })));
  expect(await loaders()).toEqual([expect.objectContaining({ lora: "ink-wash.safetensors", widget: "ink-wash.safetensors" })]);

  // Drag onto the canvas: another one where it's dropped.
  await card.dragTo(page.locator("#graph-canvas"), { targetPosition: { x: 200, y: 150 } });
  await expect.poll(async () => (await loaders()).length).toBe(2);

  // Back to the nodes; the tab keys move between the two.
  await page.locator("#tab-models").focus();
  await page.keyboard.press("ArrowLeft");
  await expect(page.locator("#sidebar-nodes")).toBeVisible();
  await expect(page.locator("#tab-nodes")).toBeFocused();
});

test("subgraphs: a selection becomes one node with its wires, opens in place, Esc backs out, and still runs (UI12)", async ({ page }) => {
  await menuClick(page, "#btn-example");
  await expect.poll(() => page.evaluate(() => graph._nodes.length)).toBe(5);
  const before = await page.evaluate(() => JSON.stringify(buildPrompt()));

  // KSampler + Empty Image -> one subgraph node: model and positive come in, the image goes out.
  const sub = await page.evaluate(() => {
    const picked = graph._nodes.filter((n) => ["KSampler", "EmptyImage"].includes(n.comfyClass));
    let node;
    try { node = convertToSubgraph(picked); } catch (err) { return { error: String(err.stack) }; }
    return {
      id: node.id, count: graph._nodes.length, inner: node.subgraph._nodes.map((n) => n.type).sort(),
      inputs: (node.inputs || []).map((i) => [i.name, i.link != null]),
      outputs: (node.outputs || []).map((o) => [o.name, (o.links || []).length]),
    };
  });
  expect(sub.error).toBeUndefined();
  expect(sub.count).toBe(4); // 5 - 2 + the subgraph node
  expect(sub.inputs.sort()).toEqual([["model", true], ["positive", true]]);
  expect(sub.outputs.map(([, wired]) => wired)).toEqual([1]);
  expect(sub.inner.filter((t) => t.startsWith("graph/"))).toEqual(["graph/input", "graph/input", "graph/output"]);

  // What goes to /prompt is the same graph as before, with the moved nodes' ids prefixed.
  const after = await page.evaluate(() => buildPrompt());
  const shape = (prompt) => Object.values(prompt.prompt || prompt).map((n) => n.class_type).sort();
  expect(shape(after)).toEqual(shape(JSON.parse(before)));
  expect(Object.keys(after.prompt || after).filter((id) => id.startsWith(`${sub.id}:`)).length).toBe(2);

  // Open it in place (double-click), the bar says where we are, Esc goes back.
  await page.evaluate((id) => openSubgraph(graph.getNodeById(id)), sub.id);
  await expect(page.locator("#subgraph-bar")).toBeVisible();
  await expect(page.locator("#subgraph-path")).toHaveText("Graph › Subgraph");
  expect(await page.evaluate(() => canvas.graph !== graph)).toBe(true);
  await page.locator("#graph-canvas").focus();
  await page.keyboard.press("Escape");
  await expect(page.locator("#subgraph-bar")).toBeHidden();
  expect(await page.evaluate(() => canvas.graph === graph)).toBe(true);

  // Undo brings the two nodes back; the subgraph survives a reload when redone.
  await page.keyboard.press("Control+z");
  await expect.poll(() => page.evaluate(() => graph._nodes.length)).toBe(5);
  await page.keyboard.press("Control+y");
  await expect.poll(() => page.evaluate(() => graph._nodes.some((n) => n.type === "graph/subgraph"))).toBe(true);
  await page.waitForTimeout(600); // the debounced auto-save
  await page.reload();
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");
  await expect.poll(() => page.evaluate(() => {
    const node = graph._nodes.find((n) => n.type === "graph/subgraph");
    return node ? node.subgraph._nodes.length : 0;
  })).toBe(5);
  expect(shape(await page.evaluate(() => buildPrompt()))).toEqual(shape(JSON.parse(before)));

  // And it runs (no model needed): Empty Image -> [Blur in a subgraph] -> Save Image.
  await page.evaluate(() => {
    leaveSubgraphs();
    graph.clear();
    const make = (type, x) => { const n = LiteGraph.createNode(registeredType[type]); n.pos = [x, 100]; graph.add(n); return n; };
    const empty = make("EmptyImage", 0), blur = make("ImageBlur", 300), save = make("SaveImage", 600);
    setNodeValue(empty, "width", 120);
    empty.connect(0, blur, 0);
    blur.connect(0, save, 0);
    convertToSubgraph([blur]);
  });
  await page.click("#btn-queue");
  const img = page.locator("#gallery .gallery-item img").first();
  await expect(img).toBeVisible({ timeout: 30_000 });
  await expect.poll(() => img.evaluate((el) => el.naturalWidth)).toBe(120);
});
