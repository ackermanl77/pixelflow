// Browser checks for the P3 features (F14/F17/F18): templates, groups, panels,
// reroute/primitive nodes and the mask editor. Same test-mode backend as smoke.spec.js.
const { test, expect } = require("@playwright/test");

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
    await page.click("#btn-templates");
    await expect(page.locator("#template-overlay")).toBeVisible();
    await page.locator(`.template-item[data-file="${template.file}"]`).click();
    // A dirty graph asks first (the app's own modal).
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
  await page.click("#btn-example");
  const confirm = page.locator("#modal-overlay:not(.hidden) .primary");
  if (await confirm.count()) await confirm.click();
  await expect.poll(() => page.evaluate(() => graph._nodes.map((n) => n.comfyClass).join(","))).toBe(
    "LoadCheckpoint,CLIPTextEncode,EmptyImage,KSampler,SaveImage",
  );
});

test("groups: Ctrl+G wraps the selection, and the group menu mutes and bypasses it", async ({ page }) => {
  await page.click("#btn-example");
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
  await page.click("#btn-xy");
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
