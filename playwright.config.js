// Playwright smoke test of the real frontend against a lightweight backend
// (report item T3): `npm run e2e` (once: `npm run e2e:install` for Chromium).
// tests/e2e/server.py starts the backend in test mode (PIXELFLOW_TEST_MODE:
// no torch, no model loads) on a throwaway data folder.
const fs = require("node:fs");
const path = require("node:path");
const { defineConfig, devices } = require("@playwright/test");

const PORT = Number(process.env.PIXELFLOW_E2E_PORT || 8299);

function python() {
  if (process.env.PIXELFLOW_PYTHON) return process.env.PIXELFLOW_PYTHON;
  for (const candidate of [".venv/Scripts/python.exe", ".venv/bin/python"]) {
    if (fs.existsSync(path.join(__dirname, candidate))) return path.join(__dirname, candidate);
  }
  return "python";
}

module.exports = defineConfig({
  testDir: "tests/e2e",
  testMatch: "*.spec.js",
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1400, height: 900 } } }],
  webServer: {
    command: `"${python()}" tests/e2e/server.py ${PORT}`,
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
