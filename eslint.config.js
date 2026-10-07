// ESLint flat config for the frontend and its JS tests (report item T3).
// `npm run lint`. The frontend is plain browser scripts (no bundler, no
// modules): logic.js, then app.js, after the vendored litegraph.js.
const js = require("@eslint/js");
const globals = require("globals");

module.exports = [
  {
    // Vendored third-party code and everything that isn't ours.
    ignores: [
      "frontend/vendor/**",
      "node_modules/**",
      "vendor/**",
      "release/**",
      "build/**",
      "build-release/**",
      "dist/**",
      "dist-release/**",
      "PixelFlow-*/**",
      ".venv/**",
      "venv/**",
      "build-venv/**",
      "test-results/**",
      "playwright-report/**",
      "models/**",
      "outputs/**",
    ],
  },
  js.configs.recommended,
  {
    files: ["frontend/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: {
        ...globals.browser,
        // From frontend/vendor/litegraph.min.js.
        LiteGraph: "readonly",
        LGraph: "readonly",
        LGraphCanvas: "readonly",
        LGraphNode: "readonly",
        // logic.js is UMD: CommonJS under Node (the unit tests), a global in the browser.
        module: "readonly",
      },
    },
  },
  {
    files: ["tests/**/*.js", "*.config.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "commonjs",
      globals: { ...globals.node },
    },
  },
  {
    // Playwright specs: callbacks passed to page.evaluate() run in the page,
    // where app.js's top-level `graph` / `canvas` are globals.
    files: ["tests/e2e/**/*.js"],
    languageOptions: { globals: { ...globals.browser, graph: "readonly", canvas: "readonly", buildPrompt: "readonly" } },
  },
  {
    rules: {
      "no-undef": "error",
      "no-unused-vars": ["error", { args: "none", caughtErrors: "none" }],
    },
  },
];
