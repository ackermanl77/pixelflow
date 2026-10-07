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
