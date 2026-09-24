# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas = [("frontend", "frontend")]
binaries = []
hiddenimports = ["backend.main"]
hiddenimports += collect_submodules("backend")

for package in ["uvicorn", "pywebview", "torch", "diffusers", "transformers", "accelerate", "safetensors"]:
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

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
    excludes=[],
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="PixelFlow",
)
