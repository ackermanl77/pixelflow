// Browser smoke test (report item T3): the page loads, the node palette
// lists the backend's nodes, nodes can be added and wired with the mouse, a
// torch-free graph runs, and a workflow saves and loads again.
// Run with `npm run e2e` (playwright.config.js starts the test-mode backend).
const { test, expect } = require("@playwright/test");

/** Click a topbar command that lives in a menu (UI4): open its menu first. */
async function menuClick(page, selector) {
  await page.locator(".menu").filter({ has: page.locator(selector) }).locator(".menu-button").click();
  await page.click(selector);
}

/** Screen position of a node's input/output slot on the litegraph canvas. */
async function slotPosition(page, nodeId, isInput, slot) {
  return page.evaluate(
    ([id, input, index]) => {
      const node = graph.getNodeById(id);
      const [x, y] = canvas.ds.convertOffsetToCanvas(node.getConnectionPos(input, index));
      const rect = document.getElementById("graph-canvas").getBoundingClientRect();
      return { x: rect.left + x, y: rect.top + y };
    },
    [nodeId, isInput, slot],
  );
}

test("load, palette, add + connect nodes, run, save and load a workflow", async ({ page, request }) => {
  const errors = [];
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto("/");
  await expect(page).toHaveTitle("PixelFlow");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");

  // The palette lists every node the backend reports (plus the frontend-only ones).
  const nodes = await (await request.get("/api/nodes")).json();
  const items = page.locator("#node-list .node-item");
  await expect(items).toHaveCount(Object.keys(nodes).length + 2); // + the frontend-only Reroute and Primitive
  await expect(items.filter({ hasText: /^KSampler$/ })).toBeVisible();
  await page.fill("#node-search", "blur");
  await expect(page.locator("#node-list .node-item:not(.hidden)")).toHaveText(["Blur"]);
  await page.fill("#node-search", "");

  // Add two nodes from the palette.
  await expect(page.locator("#empty-state")).toBeVisible();
  await items.filter({ hasText: /^Empty Image$/ }).click();
  await items.filter({ hasText: /^Save Image$/ }).click();
  await expect(page.locator("#empty-state")).toBeHidden();
  const ids = await page.evaluate(() => {
    const [source, target] = graph._nodes;
    source.pos = [80, 120];
    target.pos = [520, 160];
    canvas.ds.offset = [0, 0];
    canvas.ds.scale = 1;
    canvas.setDirty(true, true);
    return [source.id, target.id];
  });
  expect(await page.evaluate(() => graph._nodes.map((n) => n.type))).toEqual([
    expect.stringContaining("Empty Image"),
    expect.stringContaining("Save Image"),
  ]);

  // Wire Empty Image's output into Save Image's input by dragging.
  const from = await slotPosition(page, ids[0], false, 0);
  const to = await slotPosition(page, ids[1], true, 0);
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move((from.x + to.x) / 2, (from.y + to.y) / 2, { steps: 5 });
  await page.mouse.move(to.x, to.y, { steps: 5 });
  await page.mouse.up();
  await expect.poll(() => page.evaluate(() => Object.keys(graph.links).length)).toBe(1);
  expect(await page.evaluate((id) => graph.getNodeById(id).inputs[0].link !== null, ids[1])).toBe(true);

  // Queue it: Empty Image -> Save Image needs no model, so it runs for real.
  await page.click("#btn-queue");
  await expect(page.locator("#gallery .gallery-item img").first()).toBeVisible({ timeout: 30_000 });

  // Save the workflow, clear the graph, load it back from the manager.
  const name = `smoke-${Date.now()}`;
  await menuClick(page, "#btn-save-workflow");
  await page.fill("#modal-input", name);
  await page.click("#modal-confirm");
  await expect(page.locator("#toast-area")).toContainText(`Saved workflow "${name}"`);
  const saved = await (await request.get(`/api/workflows/${name}`)).json();
  expect(saved.nodes).toHaveLength(2);
  // F24: the runnable prompt rides along for the MCP server...
  const savedPrompt = saved.extra.pixelflow_prompt;
  expect(Object.values(savedPrompt).map((n) => n.class_type).sort()).toEqual(["EmptyImage", "SaveImage"]);

  await menuClick(page, "#btn-clear");
  await expect(page.locator("#empty-state")).toBeVisible();
  await menuClick(page, "#btn-workflows");
  const row = page.locator(".workflow-row", { hasText: name });
  await row.getByRole("button", { name: "Load" }).click();
  await expect(page.locator("#workflow-overlay")).toBeHidden();
  expect(await page.evaluate(() => [graph._nodes.length, Object.keys(graph.links).length])).toEqual([2, 1]);
  // ...but isn't kept in the loaded graph (it would go stale and mark it modified).
  expect(await page.evaluate(() => "pixelflow_prompt" in (graph.extra || {}))).toBe(false);

  expect(errors).toEqual([]);
});

test("UI3: device label, one Import control, category headings, favicon", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");
  await expect(page.locator("#system-status")).toHaveText("CPU · test mode (no models)");
  // Import is one named item (in the Workflow menu, UI4) that opens the file dialog; the file input is hidden.
  await page.getByRole("button", { name: "Workflow ▾" }).click();
  await expect(page.getByRole("menuitem", { name: "Import…" })).toHaveCount(1);
  await expect(page.locator("#file-import")).toBeHidden();
  const [chooser] = await Promise.all([page.waitForEvent("filechooser"), page.getByRole("menuitem", { name: "Import…" }).click()]);
  expect(chooser).toBeTruthy();
  // Node-library categories are headings a screen reader can jump between.
  await expect(page.getByRole("heading", { level: 3, name: "loaders" })).toBeVisible();
  expect((await request.get("/favicon.ico")).status()).toBe(200);
});

test("UI5: canvas toolbar zooms, fits, toggles the minimap and links", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");
  await menuClick(page, "#btn-example");
  const zoom = page.locator("#btn-zoom-reset");
  await page.click("#btn-canvas-fit");
  // Fit puts every node inside the canvas, whatever the zoom it picked.
  const inside = await page.evaluate(() => {
    const rect = document.getElementById("graph-canvas").getBoundingClientRect();
    return graph._nodes.every((node) => {
      const [x, y] = canvas.ds.convertOffsetToCanvas(node.pos);
      const [x2, y2] = canvas.ds.convertOffsetToCanvas([node.pos[0] + node.size[0], node.pos[1] + node.size[1]]);
      return x >= 0 && y >= 0 && x2 <= rect.width && y2 <= rect.height;
    });
  });
  expect(inside).toBe(true);
  await page.click("#btn-zoom-reset");
  await expect(zoom).toHaveText("100%");
  await page.click("#btn-zoom-in");
  await expect(zoom).toHaveText("125%");
  await page.click("#btn-zoom-out");
  await page.click("#btn-zoom-out");
  await expect(zoom).toHaveText("75%");
  // Minimap: off by default, remembered when turned on, and a click moves the view.
  await expect(page.locator("#minimap")).toBeHidden();
  await page.click("#btn-minimap");
  await expect(page.locator("#minimap")).toBeVisible();
  const before = await page.evaluate(() => [...canvas.ds.offset]);
  await page.locator("#minimap").click({ position: { x: 20, y: 20 } });
  expect(await page.evaluate(() => [...canvas.ds.offset])).not.toEqual(before);
  // Links: hidden and shown again.
  await page.click("#btn-links");
  await expect(page.locator("#btn-links")).toHaveAttribute("aria-pressed", "false");
  expect(await page.evaluate(() => canvas.links_render_mode)).toBe(-1);
  await page.reload();
  await expect(page.locator("#minimap")).toBeVisible();
  await expect(page.locator("#btn-links")).toHaveAttribute("aria-pressed", "false");
  await page.click("#btn-links");
  expect(await page.evaluate(() => canvas.links_render_mode)).toBe(2);
});

test("UI4: topbar menus open, run their item, close, and work from the keyboard", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");
  const workflow = page.getByRole("button", { name: "Workflow ▾" });
  await workflow.click();
  await expect(workflow).toHaveAttribute("aria-expanded", "true");
  await page.getByRole("menuitem", { name: "Load Example" }).click();
  await expect(page.locator("#menu-workflow")).toBeHidden();
  await expect.poll(() => page.evaluate(() => graph._nodes.length)).toBeGreaterThan(0);  // the template loads async
  // Keyboard: ArrowDown opens and focuses the first item, arrows move, Escape closes and returns focus.
  const edit = page.getByRole("button", { name: "Edit ▾" });
  await edit.focus();
  await page.keyboard.press("ArrowDown");
  await expect(page.getByRole("menuitem", { name: /^Undo/ })).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(page.getByRole("menuitem", { name: /^Redo/ })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.locator("#menu-edit")).toBeHidden();
  await expect(edit).toBeFocused();
  // Opening one menu closes another; a click outside closes it.
  await page.getByRole("button", { name: "View ▾" }).click();
  await workflow.click();
  await expect(page.locator("#menu-view")).toBeHidden();
  await page.mouse.click(600, 500);
  await expect(page.locator("#menu-workflow")).toBeHidden();
});

test("UI6: workflow tabs keep separate graphs, open saved workflows in a tab, survive a reload", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");
  const tabs = page.locator("#workflow-tab-list [role='tab']");
  const nodeCount = () => page.evaluate(() => graph._nodes.length);
  await expect(tabs).toHaveCount(1);
  await menuClick(page, "#btn-example");
  await expect.poll(nodeCount).toBe(5);
  await expect(tabs.first()).toHaveText("Untitled"); // a template counts as unchanged until edited
  // Save it, then a new tab starts empty and the first keeps its graph.
  await menuClick(page, "#btn-save-workflow");
  await page.fill("#modal-overlay:not(.hidden) .modal-box input", "tab-test");
  await page.keyboard.press("Enter");
  await expect(tabs.first()).toHaveText("tab-test");
  await page.click("#btn-new-tab");
  await expect(tabs).toHaveCount(2);
  await expect(tabs.nth(1)).toHaveAttribute("aria-selected", "true");
  expect(await nodeCount()).toBe(0);
  await page.locator("#node-list .node-item").filter({ hasText: /^Blur$/ }).click();
  await expect.poll(nodeCount).toBe(1);
  await tabs.first().click();
  await expect.poll(nodeCount).toBe(5);
  // Opening the saved workflow while it's open in a tab switches to that tab instead.
  await tabs.nth(1).click();
  await menuClick(page, "#btn-workflows");
  await page.locator(".workflow-row").filter({ has: page.getByText("tab-test", { exact: true }) })
    .getByRole("button", { name: "Load" }).click();
  await expect(tabs.first()).toHaveAttribute("aria-selected", "true");
  await expect(tabs).toHaveCount(2);
  // A reload brings back both tabs; closing the unsaved one asks first.
  await page.waitForTimeout(600); // the debounced auto-save
  await page.reload();
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");
  await expect(tabs).toHaveCount(2);
  await expect(tabs.nth(1)).toHaveText("Untitled •");
  await page.getByRole("button", { name: "Close Untitled" }).click();
  await expect(page.locator("#modal-overlay:not(.hidden) .modal-box")).toContainText("unsaved changes");
  await page.getByRole("button", { name: "Close", exact: true }).click();
  await expect(tabs).toHaveCount(1);
  expect(await nodeCount()).toBe(5);
});

test("UI7: every template card shows a diagram of its graph", async ({ page, request }) => {
  const { templates } = await (await request.get("/templates/index.json")).json();
  await page.goto("/");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");
  await menuClick(page, "#btn-templates");
  await expect(page.locator(".template-thumb")).toHaveCount(templates.length);
  // Each canvas gets drawn on (its template file is fetched first).
  await expect.poll(() => page.evaluate(() => [...document.querySelectorAll(".template-thumb")].every((canvas) => {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    return data.some((value, index) => index % 4 === 3 && value > 0);
  }))).toBe(true);
});
