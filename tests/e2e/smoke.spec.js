// Browser smoke test (report item T3): the page loads, the node palette
// lists the backend's nodes, nodes can be added and wired with the mouse, a
// torch-free graph runs, and a workflow saves and loads again.
// Run with `npm run e2e` (playwright.config.js starts the test-mode backend).
const { test, expect } = require("@playwright/test");

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
  await page.click("#btn-save-workflow");
  await page.fill("#modal-input", name);
  await page.click("#modal-confirm");
  await expect(page.locator("#toast-area")).toContainText(`Saved workflow "${name}"`);
  const saved = await (await request.get(`/api/workflows/${name}`)).json();
  expect(saved.nodes).toHaveLength(2);

  await page.click("#btn-clear");
  await expect(page.locator("#empty-state")).toBeVisible();
  await page.click("#btn-workflows");
  const row = page.locator(".workflow-row", { hasText: name });
  await row.getByRole("button", { name: "Load" }).click();
  await expect(page.locator("#workflow-overlay")).toBeHidden();
  expect(await page.evaluate(() => [graph._nodes.length, Object.keys(graph.links).length])).toEqual([2, 1]);

  expect(errors).toEqual([]);
});
