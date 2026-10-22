// The 2026-10-21 competitor rows in a real browser: Image Compare's in-node A/B
// split follows the mouse (F28), a template is inserted beside the current graph
// as a reusable block (F28), and the System panel shows the MCP configuration
// to copy into an assistant (F27).
const { test, expect } = require("@playwright/test");

test("Image Compare shows A and B with a split that follows the mouse", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (err) => errors.push(err.message));
  await page.goto("/");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");
  const compareId = await page.evaluate(() => {
    const a = addNodeToGraph("EmptyImage", [40, 60]);
    const b = addNodeToGraph("EmptyImage", [40, 360]);
    const compare = addNodeToGraph("ImageCompare", [420, 120]);
    setNodeValue(a, "color", "#ff0000");
    setNodeValue(b, "color", "#0000ff");
    a.connect(0, compare, 0);
    b.connect(0, compare, 1);
    canvas.ds.offset = [0, 0];
    canvas.ds.scale = 1;
    canvas.setDirty(true, true);
    return compare.id;
  });
  await page.click("#btn-queue");
  await expect.poll(() => page.evaluate((id) => {
    const pair = comparePreviews.get(id);
    return Boolean(pair && pair.every((img) => img.complete && img.naturalWidth));
  }, compareId), { timeout: 30_000 }).toBe(true);
  await expect.poll(() => page.evaluate((id) => Boolean(graph.getNodeById(id).compareBox), compareId)).toBe(true);

  // Hover near the left edge of the images: the split follows.
  const point = await page.evaluate((id) => {
    const node = graph.getNodeById(id);
    const [x, top, w, h] = node.compareBox;
    const [cx, cy] = canvas.ds.convertOffsetToCanvas([node.pos[0] + x + w * 0.2, node.pos[1] + top + h / 2]);
    const rect = document.getElementById("graph-canvas").getBoundingClientRect();
    return { x: rect.left + cx, y: rect.top + cy };
  }, compareId);
  await page.mouse.move(point.x, point.y, { steps: 4 });
  const split = await page.evaluate((id) => graph.getNodeById(id).compareSplit, compareId);
  expect(split).toBeGreaterThan(0.1);
  expect(split).toBeLessThan(0.3);
  expect(errors).toEqual([]);
});

test("a template can be inserted beside the current graph", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#conn-status")).toHaveAttribute("data-state", "connected");
  await page.evaluate(() => {
    addNodeToGraph("EmptyImage", [40, 60]);
  });
  await page.click("#btn-templates");
  await page.check("#template-insert");
  await page.locator(".template-item", { hasText: "Remove background" }).click();
  await expect(page.locator("#template-overlay")).toBeHidden();
  await expect.poll(() => page.evaluate(() => graph._nodes.length)).toBe(5);
  const types = await page.evaluate(() => graph._nodes.map((n) => n.comfyClass || n.type));
  expect(types).toEqual(expect.arrayContaining(["EmptyImage", "RemoveBackground", "ImageCompare"]));
  expect(await page.evaluate(() => Object.keys(canvas.selected_nodes).length)).toBe(4);
});

test("the System panel shows the MCP configuration for an assistant", async ({ page }) => {
  await page.goto("/");
  await page.click("#system-status");
  await expect(page.locator("#mcp-config")).toContainText('"mcpServers"');
  await expect(page.locator("#mcp-config")).toContainText('"pixelflow"');
  await expect(page.locator("#mcp-claude-code")).toContainText("claude mcp add pixelflow");
});
