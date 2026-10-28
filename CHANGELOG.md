# Changelog

All notable changes to PixelFlow. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). No version has been published yet; the version
is stamped at build time (`scripts/build_release.ps1 -Version`).

## [Unreleased] -- the first release

### Added
- A node-based image-generation app (litegraph.js frontend, FastAPI backend, desktop window):
  SD 1.x/2.x, SDXL, SD3, Flux and split-file models such as Z-Image Turbo; txt2img, img2img,
  inpainting and outpainting; LoRA, ControlNet with Canny, depth, soft edges, lineart, scribble
  and pose preprocessors; IP-Adapter; regional prompts; Detail Faces & Hands; model upscaling.
- Groups, a mask editor, Reroute and Primitive nodes, workflow templates and tabs, an X/Y plot,
  App mode, wildcards, embeddings, node plugins, an MCP server, and model downloads from
  Hugging Face and Civitai.
- Live latent previews, cached prompts and outputs, and a self-test of every release build.
- Uncaught errors from any thread are written to `logs/pixelflow.log`.
- An update check that stays off until a release feed is configured; when a newer release
  exists, a notice with a Download link appears.
- Dependabot, and a `pip-audit` job in CI.
- The MIT license, `THIRD_PARTY_NOTICES.md` (shipped in the build), `SECURITY.md` and a
  privacy note in the README.

### Security
- Upgraded Pillow to 13.0 (10 advisories in 11.3), starlette to 1.10 with FastAPI 0.162
  (6 advisories), filelock, certifi and setuptools. The lock file has no known
  vulnerabilities.
- Token authentication, pixel limits, upload buffering and plugin isolation (earlier rounds).
