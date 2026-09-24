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
> `accelerate`/`safetensors` via `collect_all`, but this hasn't been
> verified against a full `scripts/build_release.ps1` run — do a real build
> and smoke-test the generation nodes in the packaged `.exe` before shipping
> a release, since PyInstaller-freezing ML libraries with dynamic imports can
> surface missing-hidden-import errors that only show up at runtime.

## One-time setup

```powershell
pip install -r requirements.txt
pip install -r requirements-desktop.txt
```

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
(see below).

## Full release build (local)

```powershell
.\scripts\build_release.ps1 -Version "0.1.0"
```

This mirrors the sibling `photo_professional_tool` project's
`build_release.ps1` exactly in approach:

1. Installs `requirements.txt` + `requirements-desktop.txt`.
2. Runs PyInstaller against `PixelFlow.spec`.
3. Moves (not copies) the built exe folder to `release\` — the only
   unpacked copy, so a multi-GB CUDA build is never duplicated on disk.
4. Downloads (or reuses) a pinned "Fixed Version" copy of the Microsoft Edge
   WebView2 Runtime and bundles it as a sibling `WebView2Runtime\` folder next
   to `PixelFlow.exe` — **the packaged app doesn't depend on the target
   machine having WebView2 installed at all.**
5. Optionally code-signs the exe (`-Sign -CertificateSubject "..."`).
6. Compresses `release\` to `PixelFlow-<version>-Windows.7z` + `.sha256` if
   `7z` is on `PATH` (otherwise just warns). `-SkipArchive` skips this and
   removes a stale archive from an earlier build.
7. Deletes `build-release\` and `dist-release\` after a successful build
   (`-KeepBuildFiles` keeps PyInstaller's cache for a faster rebuild; a failed
   build always keeps them for debugging).
7. Mirrors the result to a fixed-name `release\` folder, so
   `release\PixelFlow.exe` is always where the latest build landed — no need
   to know the version number to find it.

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
- **User data/writable** (`outputs/`, `uploads/`, `workflows/`) — created
  next to the actual `.exe` (`Path(sys.executable).parent`) when frozen, so a
  saved workflow or a generated image survives after the app closes and
  reopens, instead of vanishing with the temp extraction dir.

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

- Check for a Python traceback — the packaged app runs with `console=False`
  in `PixelFlow.spec`, so set it to `True` temporarily to see console output
  while debugging a broken build.
- Confirm `release\WebView2Runtime\msedgewebview2.exe` exists — if the
  bundling step silently failed, pywebview falls back to any
  system-installed WebView2, which may not exist on the target machine.
