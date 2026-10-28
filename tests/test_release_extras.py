"""Release/CI configuration checks for K3 (2 GiB asset limit), K4 (lock
file), K5 (spec cleanups), K6 (version + signing) and T5 (CI matrix and
gating). These read the files - nothing is built."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = (ROOT / "scripts" / "build_release.ps1").read_text(encoding="utf-8")
RELEASE = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
SPEC = (ROOT / "PixelFlow.spec").read_text(encoding="utf-8")


def test_big_archives_are_split_below_githubs_asset_limit():
    assert "$MaxAssetBytes = 1900MB" in BUILD
    assert '"-v$volumeSize"' in BUILD
    assert "PixelFlow-*-Windows.7z*" in RELEASE
    assert "Get-ChildItem -Path \"PixelFlow-*-Windows.7z*\"" in RELEASE, "pwsh doesn't glob for gh"
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "PixelFlow-*-Windows.7z.[0-9][0-9][0-9]" in gitignore, "split volumes must not show up in git status"


def test_lock_file_pins_the_tested_versions_and_the_build_uses_it():
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    pins = dict(line.split("==") for line in lock.splitlines() if line and not line.startswith("#"))
    for name in ("diffusers", "transformers", "torch", "fastapi", "pywebview", "pyinstaller"):
        assert name in pins, name
    assert "+" not in pins["torch"], "the CUDA flavour comes from the index URL, not the pin"
    assert '@("-c", "requirements-lock.txt")' in BUILD
    # Constrained too, or a newer torch on the CUDA index is swapped for PyPI's CPU-only locked one.
    assert "--index-url $TorchIndexUrl @constraints" in BUILD
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^diffusers>=[\d.]+,<[\d.]+$", requirements, re.MULTILINE), "diffusers needs an upper bound"
    assert re.search(r"^torch>=2\.7$", requirements, re.MULTILINE), "sm_120 needs torch 2.7+"
    assert re.search(r"^peft", requirements, re.MULTILINE)


def test_build_runs_in_its_own_venv_and_never_installs_into_the_base_interpreter():
    assert '$BuildVenv = Join-Path $ProjectRoot "build-venv"' in BUILD
    assert "& $Python -m venv $BuildVenv" in BUILD
    # $Python is only the base interpreter: the one thing it may run is "-m venv".
    uses = re.findall(r"& \$Python\b(.*)", BUILD)
    assert uses == [" -m venv $BuildVenv"], uses
    for step in ("-m pip install torch", "-m pip install -r requirements.txt", "-m PyInstaller"):
        assert f"& $BuildPython {step}" in BUILD, step
    assert "-Python (Get-Command python).Source" in RELEASE, "CI passes the base interpreter"
    assert "/build-venv/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_spec_collects_webview_and_drops_torch_build_files():
    tree = ast.parse(SPEC)  # it's Python
    assert tree
    assert 'packages = ["uvicorn", "webview"' in SPEC
    assert "torch/include/" in SPEC and '.endswith(".lib")' in SPEC
    assert '"torch.utils.tensorboard"' in SPEC


def test_version_is_stamped_and_signing_is_checked():
    assert "backend\\_version.py" in BUILD and "__version__" in BUILD
    assert "$CertificateThumbprint" in BUILD and "$CertificateSubject" not in BUILD
    sign = BUILD.index("& $SignTool sign")
    assert "$LASTEXITCODE" in BUILD[sign : sign + 400], "a failed signature must stop the build"
    required = BUILD.index("-Sign needs -CertificateThumbprint (or the CODESIGN_THUMBPRINT environment variable).")
    assert required < BUILD.index("-m PyInstaller"), "a missing thumbprint must fail before the build"
    assert "SIGNING_CERT_PFX" in RELEASE
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/backend/_version.py" in gitignore


def test_ci_matches_the_shipped_configuration_and_gates_releases():
    assert "windows-latest" in CI and '"3.14"' in CI
    assert 'PIXELFLOW_REQUIRE_HUB: "1"' in CI
    assert "pytest -q -rs" in CI
    assert "workflow_call:" in CI
    assert "uses: ./.github/workflows/ci.yml" in RELEASE and "needs: test" in RELEASE
    assert 'python-version: "3.14"' in RELEASE


def test_installer_packages_the_release_folder_in_slices_below_githubs_asset_limit():
    iss = (ROOT / "installer" / "PixelFlow.iss").read_text(encoding="utf-8")
    assert r'Source: "..\release\*"' in iss, "the finished release folder, with the WebView2 runtime"
    assert "DiskSpanning=yes" in iss
    slice_size = int(re.search(r"^DiskSliceSize=(\d+)$", iss, re.M).group(1))
    assert slice_size < 2 * 1024**3, "each slice must fit a GitHub release asset"
    assert "PrivilegesRequired=lowest" in iss, "installs per user without admin rights"
    # The deepest bundled file is ~140 characters below {app}; Windows paths stop at 260.
    assert int(re.search(r"MaxAppDirLength = (\d+);", iss).group(1)) <= 110
    assert "function PrepareToInstall" in iss, "a too-long folder is refused before copying, silent or not"
    assert r"{localappdata}\PixelFlow" in iss and "IDNO" in iss, "user data is kept unless the user agrees"
    assert '"/DMyAppVersion=$Version"' in BUILD and r"installer\PixelFlow.iss" in BUILD
    installer = BUILD.index("if ($Installer)")
    assert "$LASTEXITCODE" in BUILD[installer : installer + 2000], "a failed compile must stop the build"
    assert "-Installer" in RELEASE and r"installer-output\PixelFlow-*-Setup*" in RELEASE
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/installer-output/" in gitignore
