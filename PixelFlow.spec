# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

# K8: the version resource (publisher MELEJ, the version build_release.ps1 stamps into
# backend/_version.py) and the manifest (UTF-8 active code page, PerMonitorV2), from
# scripts/windows_resources.py. SPECPATH/workpath are PyInstaller's globals.
sys.path.insert(0, os.path.join(SPECPATH, "scripts"))
import windows_resources  # noqa: E402

VERSION_PATH = windows_resources.write_version_file(workpath)
MANIFEST_PATH = windows_resources.write_manifest(workpath)

datas = [("frontend", "frontend")]
binaries = []
hiddenimports = ["backend.main"]
hiddenimports += collect_submodules("backend")

# "webview" is pywebview's import name (collect_all("pywebview") collected nothing). peft (LoRA
# support) is optional: bundled when installed in the build interpreter.
packages = ["uvicorn", "webview", "torch", "diffusers", "transformers", "accelerate", "safetensors"]
if importlib.util.find_spec("peft") is not None:
    packages.append("peft")
# spandrel (model upscalers) registers its architectures by importing them; collect them all.
if importlib.util.find_spec("spandrel") is not None:
    packages.append("spandrel")
# transformers' image utilities import torchvision whenever it's installed, and every diffusers
# pipeline goes through them. Import analysis only picks up torchvision's .py files, not its
# compiled operators (_C_stable.pyd, image_stable.pyd and their DLLs), so the frozen app failed
# with "operator torchvision::nms does not exist" and no pipeline could load. Found by --self-test.
if importlib.util.find_spec("torchvision") is not None:
    packages.append("torchvision")
for package in packages:
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports


def _needed_at_runtime(src, dest):
    """torch's C++ headers (torch/include, ~38 MB) and .lib import libraries (~44 MB) are only
    needed to compile extensions, never by the running app."""
    folder = dest.replace("\\", "/").lower().rstrip("/") + "/"
    if folder.startswith("torch/include/"):
        return False
    return not (folder.startswith("torch/lib/") and src.lower().endswith(".lib"))


datas = [entry for entry in datas if _needed_at_runtime(entry[0], entry[1])]
binaries = [entry for entry in binaries if _needed_at_runtime(entry[0], entry[1])]

# diffusers and transformers verify their dependencies' installed versions at import time via
# importlib.metadata (see their dependency_versions_check.py). PyInstaller doesn't bundle
# .dist-info metadata by default, so without this `import diffusers` raises PackageNotFoundError
# ("The 'requests' distribution was not found") inside the frozen app -- on the background server
# thread, which is what made the packaged PixelFlow look like it hung on startup.
for package in [
    "requests", "filelock", "numpy", "tqdm", "regex", "packaging", "tokenizers",
    "huggingface-hub", "safetensors", "accelerate", "pyyaml",
]:
    datas += copy_metadata(package)

a = Analysis(
    ["desktop.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch.utils.tensorboard", "tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)
# One-folder build (exe + _internal\), not one-file: with the CUDA build of torch the bundle is
# several GB, which a one-file exe would re-extract to a temp folder on every launch.
# scripts/build_release.ps1 already copies either layout into the release folder.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PixelFlow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    version=VERSION_PATH,
    manifest=MANIFEST_PATH,
    icon="assets/icon.ico",  # rendered from assets/icon.svg
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="PixelFlow",
)
