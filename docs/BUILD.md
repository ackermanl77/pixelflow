# Building the desktop app

PixelFlow can run three ways:

1. **Dev web app** — `python run.py`, opened in a normal browser tab, with
   `uvicorn --reload` for live-reloading backend code.
2. **Dev desktop window** — `python desktop.py`, the same server opened in a
   native pywebview window (no `--reload`).
3. **Packaged `.exe`** — `desktop.py` frozen with PyInstaller into a
   standalone Windows app, with its own bundled WebView2 Runtime so it works
   on a machine that doesn't have WebView2 installed.

This doc covers building #3.

> **Package size note:** since `LoadCheckpoint`/`KSampler` run real
> `diffusers`/`torch` inference (see [PROJECT_STATUS.md](PROJECT_STATUS.md)),
> the packaged `.exe` now bundles those libraries too, which makes the build
> considerably larger and slower than before torch was a dependency. Model
> *weights* are not bundled — they're downloaded to `models/checkpoints/`
> (local files) or the Hugging Face Hub cache on first use, not packaged into
> the archive. `PixelFlow.spec` collects `torch`/`diffusers`/`transformers`/
> `accelerate`/`safetensors` via `collect_all`. A full
> `scripts/build_release.ps1` run was verified on 2026-10-01 (Python 3.14.7,
> `torch 2.14.1+cu130`, RTX 5080): about 5.5 minutes with cached wheels,
> `release\` 3.89 GB in 7,366 files (`_internal` 3.11 GB, of which
> `torch\lib` is 2.87 GB; `WebView2Runtime` 0.70 GB), and a 1.94 GB `.7z`
> (1,944,501,485 bytes, only ~48 MB under the 1900 MB split threshold; the
> split path was checked separately with a smaller `-MaxAssetBytes`). The
> packaged app started, answered `/api/health`, listed all nodes and reported
> `device cuda` (RTX 5080) in `/api/system`. Actually *running* the generation
> nodes with a real model in the packaged `.exe` hasn't been done yet — do
> that before shipping a release, since PyInstaller-freezing ML libraries with
> dynamic imports can surface missing-hidden-import errors that only show up
> at runtime. (PyInstaller logs many `Hidden import
> 'torch.distributed._shard.checkpoint.…' not found` ERROR lines: those are
> torch's deprecated alias modules and don't affect the build.)

## One-time setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -m pip install -r requirements.txt -r requirements-desktop.txt
```

**Install torch from the CUDA index first.** On Windows, PyPI's `torch`
wheel is CPU-only, and once *any* torch satisfies `requirements.txt`'s
`torch>=2.7`, pip won't replace it. PyInstaller bundles whatever torch the
building interpreter has, so a CPU torch means a CPU-only app. The pinned
index is **cu130** (what the last verified build used, `torch 2.14.1+cu130`;
needs an NVIDIA driver 580 or newer). **cu128** also works (driver 570+);
anything older than CUDA 12.8 has no kernels for RTX 50-series (sm_120)
GPUs, and the build script refuses it.

### Locked versions

`requirements.txt` holds version *ranges*; `requirements-lock.txt` pins the
exact versions (including every transitive dependency) the app was last
tested with, and the release build installs with
`-c requirements-lock.txt`. Regenerate it from the dev `.venv` (where the
tests run) after upgrading something and re-running the tests:

```powershell
.\.venv\Scripts\python.exe scripts\make_lock.py
```

torch is pinned without its `+cu130` suffix: the CUDA flavour comes from the
index URL (`https://download.pytorch.org/whl/cu130`). Every package in
`requirements.txt` is pinned, `peft` (LoRA support) included; a package the
dev venv lacks is listed as unpinned in the lock's header, so install
`requirements.txt` into `.venv` before regenerating.

## Quick build (no packaging extras)

```powershell
pyinstaller --noconfirm PixelFlow.spec
dist\PixelFlow.exe
```

This produces a working `.exe`, but it depends on the **system-installed**
WebView2 Runtime (the normal "Evergreen" one that ships with Windows 10/11
and auto-updates). That's fine for testing on this machine. For a build you
intend to hand to someone else, use the full release script below instead.

## CI: building automatically

You don't have to run the release script by hand. `.github/workflows/release.yml`
(mirroring `photo_professional_tool`'s own release workflow) runs the exact
same `scripts\build_release.ps1` on a GitHub-hosted `windows-latest` runner:

- **Push a version tag** (`git tag v0.1.0 && git push origin v0.1.0`) to build
  *and* publish a GitHub Release with `PixelFlow-<version>-Windows.7z` +
  `.sha256` attached.
- **Or trigger it manually** — GitHub → Actions tab → "Release Build" → "Run
  workflow", type a version — which builds and uploads the same files as a
  workflow artifact without creating a tag or a Release.

This is triggered by a tag or a manual run, not by every push to `main` — a
routine commit doesn't kick off a multi-minute Windows CI build on its own.
The CI runner has no `photo_professional_tool` checkout next to it, so it
always downloads the WebView2 Runtime fresh instead of reusing that cache
(see below). The workflow passes `-Python` and `-TorchIndexUrl
https://download.pytorch.org/whl/cu130` explicitly (keep that in sync with
the script's default), and after the build a separate step fails the run
unless `release\build-info.json` and the bundled `torch\version.py` both
name a CUDA version. The release job only runs after the full test suite
(`ci.yml`, called as a reusable workflow) passes on Linux and on
Windows/Python 3.14. A CUDA build is several GB: when the `.7z` is larger
than `-MaxAssetBytes` (1900 MB, under GitHub's 2 GiB per-asset limit) the
script re-creates it as split volumes `PixelFlow-<version>-Windows.7z.001`,
`.002`, … and the `.sha256` file lists every volume. To unpack, download all
volumes into one folder and open the `.001` with 7-Zip (or
`7z x PixelFlow-<version>-Windows.7z.001`).

**Code signing in CI** is optional: add a base64-encoded `.pfx` as the
`SIGNING_CERT_PFX` secret and its password as `SIGNING_CERT_PASSWORD`; the
workflow imports it and passes its thumbprint to the script.

## Full release build (local)

```powershell
.\scripts\build_release.ps1 -Version "0.1.0"
```

This mirrors the sibling `photo_professional_tool` project's
`build_release.ps1` exactly in approach:

1. Creates a fresh `build-venv\` (gitignored) from a base interpreter:
   `-Python <path>`, else `py -3.14`, else `python` on `PATH` (no
   machine-specific paths). Everything is installed into that venv and
   PyInstaller runs from it, so the build never touches your dev `.venv` and
   what ships depends only on the requirement files. The venv (several GB of
   CUDA torch) is deleted after a successful build; `-ReuseVenv` keeps it and
   reuses it next time for a faster rebuild.
2. Installs torch from `-TorchIndexUrl` (default
   `https://download.pytorch.org/whl/cu130`), then `requirements.txt` +
   `requirements-desktop.txt`, all
   constrained by `requirements-lock.txt` (the torch step too, so a newer
   torch on the index can't later be swapped for PyPI's CPU-only locked
   version).
3. **CUDA guard:** prints `torch.__version__`/`torch.version.cuda` and stops
   before PyInstaller if torch is a CPU-only build or older than CUDA 12.8.
   `-AllowCpu` turns that into a warning (for a deliberate CPU build). It
   checks the torch build, not a GPU, so it works on a GPU-less CI runner.
4. Writes `backend\_version.py` (`-Version`; the app shows it in
   `/api/health` and the System panel, "dev" from source), runs PyInstaller
   against `PixelFlow.spec` (which collects `webview`, drops torch's C++
   headers and `.lib` files, ~80 MB, and excludes tensorboard/tkinter), and
   removes `_version.py` again.
5. Moves (not copies) the built exe folder to `release\` — the only
   unpacked copy, so a multi-GB CUDA build is never duplicated on disk.
6. Downloads (or reuses) a pinned "Fixed Version" copy of the Microsoft Edge
   WebView2 Runtime and bundles it as a sibling `WebView2Runtime\` folder next
   to `PixelFlow.exe` — **the packaged app doesn't depend on the target
   machine having WebView2 installed at all.** Then writes
   `release\build-info.json` (version, git SHA, build time, Python, torch,
   torch CUDA, diffusers, transformers) so a shipped build can be identified.
7. Optionally code-signs the exe (`-Sign -CertificateThumbprint <sha1>`); a
   failing `signtool` stops the build. Without `-Sign` this step is skipped
   silently. With `-Sign`, a missing thumbprint or a missing `signtool` (looked
   up on `PATH`, then in the newest `Windows Kits\10\bin\10.*\x64`) fails at
   the start instead of after the build.
8. Compresses `release\` to `PixelFlow-<version>-Windows.7z` + `.sha256` if
   7-Zip is found (`7z` on `PATH` or `%ProgramFiles%\7-Zip\7z.exe`;
   otherwise just warns), split into volumes when it's over
   `-MaxAssetBytes` (pass a plain byte count, e.g. `-MaxAssetBytes
   1048576000`, when calling through `powershell -File`: `1000MB` is only
   understood inside PowerShell). `-SkipArchive` skips this and removes a
   stale archive from an earlier build.
9. Deletes `build-release\` and `dist-release\` after a successful build
   (`-KeepBuildFiles` keeps PyInstaller's cache for a faster rebuild; a failed
   build always keeps them for debugging).
The result always lands in the fixed-name `release\` folder, so
`release\PixelFlow.exe` is where the latest build is — no need to know the
version number to find it.

**Why bundle WebView2 at all?** pywebview's Windows backend (`edgechromium`)
needs the WebView2 Runtime to open a window at all. The sibling
photo_professional_tool project hit this directly — its app's window failed
to open on a second, freshly-provisioned PC that didn't have it installed.
Bundling a pinned copy avoids depending on what's already on the target
machine.

**Reusing the sibling project's cache.** `photo_professional_tool` already
downloaded and extracted this exact pinned WebView2 version
(`vendor\webview2-cache\`) for its own build. `build_release.ps1` checks for
that cache first and copies from it instead of re-downloading ~300MB when
it's there. If you don't have that sibling project checked out, or its cache
was cleared, the script falls back to downloading straight from Microsoft.

**Bumping the WebView2 version.** Unlike the Evergreen runtime, a bundled
Fixed Version copy does *not* auto-update with Chromium security patches, so
`$WebView2Version` in `scripts\build_release.ps1` should be bumped
deliberately every so often. There's no stable "latest" URL — get a new one
from the [WebView2 downloads page](https://developer.microsoft.com/en-us/microsoft-edge/webview2/#download-section):
pick the version + x64 under "Fixed Version", accept the EULA, then read the
`.cab` URL out of the page's own HTML (search it for `filestreamingservice`
— each version gets its own URL).

## How the packaged app finds its files

`backend/paths.py` splits "where bundled resources live" from "where user
data lives" specifically because PyInstaller's onefile mode extracts bundled
data (`frontend/`) into a temp directory (`sys._MEIPASS`) that's cleaned up
between runs:

- **Bundled/read-only** (`frontend/`) — comes from `sys._MEIPASS` when
  frozen, or the repo's `frontend/` folder in dev.
- **User data/writable** (`outputs/`, `uploads/`, `workflows/`, `models/`,
  `logs/`, …) — `%LOCALAPPDATA%\PixelFlow` when frozen (so an install under
  `C:\Program Files` works), or next to `PixelFlow.exe` when a
  `portable.txt` file is there (or an older version's data already is);
  `PIXELFLOW_HOME` overrides both, and `PIXELFLOW_MODELS_DIR` /
  `settings.json`'s `models_dir` move the models. See
  [PROJECT_STATUS.md](PROJECT_STATUS.md#where-data-lives-backendpathspy).
  Checked against the packaged 2026-10-01 build (via `/api/system`'s
  `data_dir`/`models_dir`): `%LOCALAPPDATA%\PixelFlow` by default, the exe's
  folder with `portable.txt` or an existing `outputs` folder there, and
  `PIXELFLOW_HOME`/`PIXELFLOW_MODELS_DIR` take precedence.

## GPU checks before a release

Tests marked `gpu` (`tests/test_gpu.py`) need a CUDA device: they're skipped
in CI and on CPU-only machines. Run them on the RTX 5080 before a release:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m gpu
$env:PIXELFLOW_GPU_SMOKE = "1"   # also sample a real 512x512 image with sd-turbo (~2.5 GB download)
.\.venv\Scripts\python.exe -m pytest -q -m gpu
```

## Offline operation

`frontend/vendor/litegraph.min.js` and `litegraph.css` are vendored locally
(downloaded once from the `litegraph.js` npm package) rather than loaded from
a CDN `<script src>`, so the packaged desktop app works with no internet
connection at all — this matters because a `.exe` handed to someone else
can't assume they have connectivity, or that a CDN will always be reachable.

## Testing a build

```powershell
release\PixelFlow.exe
```

A native "PixelFlow" window should open. If it doesn't:

- Check `logs\pixelflow.log` in the data folder (System panel → Open log
  folder; `%LOCALAPPDATA%\PixelFlow\logs` by default; rotating, 5 MB × 3):
  the packaged app runs with `console=False`, so everything — startup
  errors, failing nodes' tracebacks, and a startup line naming the torch
  version, CUDA version and device — is logged there.
  `logs\server-error.log` records a server that failed to start at all.
- If the startup line says `CUDA none - CPU-only build`, the build used a
  CPU torch (only possible with `-AllowCpu`); `release\build-info.json`
  says the same.
- Confirm `release\WebView2Runtime\msedgewebview2.exe` exists — if the
  bundling step silently failed, pywebview falls back to any
  system-installed WebView2, which may not exist on the target machine.
