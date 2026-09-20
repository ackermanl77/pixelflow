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

## Full release build

```powershell
.\scripts\build_release.ps1 -Version "0.1.0"
```

This mirrors the sibling `photo_professional_tool` project's
`build_release.ps1` exactly in approach:

1. Installs `requirements.txt` + `requirements-desktop.txt`.
2. Runs PyInstaller against `PixelFlow.spec`.
3. Copies the built exe into a version-named folder,
   `PixelFlow-<version>-Windows\`.
4. Downloads (or reuses) a pinned "Fixed Version" copy of the Microsoft Edge
   WebView2 Runtime and bundles it as a sibling `WebView2Runtime\` folder next
   to `PixelFlow.exe` — **the packaged app doesn't depend on the target
   machine having WebView2 installed at all.**
5. Optionally code-signs the exe (`-Sign -CertificateSubject "..."`).
6. Compresses to a `.7z` + `.sha256` if `7z` is on `PATH` (otherwise leaves
   the unpacked folder, with a warning).
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
