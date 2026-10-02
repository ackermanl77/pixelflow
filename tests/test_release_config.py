"""Guards against the release build silently shipping CPU-only torch
(report items K1/K2). These check the scripts' text - nothing is built."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = (ROOT / "scripts" / "build_release.ps1").read_text(encoding="utf-8")
RELEASE_WORKFLOW = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
CUDA_INDEX = re.compile(r"https://download\.pytorch\.org/whl/cu(\d+)")


def test_build_script_has_no_machine_specific_interpreter_path():
    assert "C:\\Users\\" not in BUILD_SCRIPT


def test_build_script_installs_torch_from_a_cuda_index_before_requirements():
    match = CUDA_INDEX.search(BUILD_SCRIPT)
    assert match and int(match.group(1)) >= 128, "RTX 50-series needs a CUDA 12.8+ wheel"
    assert BUILD_SCRIPT.index("pip install torch --index-url") < BUILD_SCRIPT.index("pip install -r requirements.txt")


def test_build_script_refuses_cpu_torch_unless_allowed():
    assert "torch.version.cuda" in BUILD_SCRIPT
    assert "[switch]$AllowCpu" in BUILD_SCRIPT
    guard = BUILD_SCRIPT.index("torch.version.cuda")
    assert guard < BUILD_SCRIPT.index("-m PyInstaller"), "the guard must run before PyInstaller"
    assert "build-info.json" in BUILD_SCRIPT


def test_release_workflow_pins_a_cuda_index_and_verifies_the_result():
    match = CUDA_INDEX.search(RELEASE_WORKFLOW)
    assert match and int(match.group(1)) >= 128
    assert "-TorchIndexUrl" in RELEASE_WORKFLOW
    assert "-AllowCpu" not in RELEASE_WORKFLOW
    assert "torch_cuda" in RELEASE_WORKFLOW


@pytest.mark.skipif(shutil.which("powershell") is None and shutil.which("pwsh") is None, reason="no PowerShell")
def test_build_script_parses():
    shell = shutil.which("pwsh") or shutil.which("powershell")
    command = (
        "$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{ROOT / 'scripts' / 'build_release.ps1'}', [ref]$null, [ref]$e); exit $e.Count"
    )
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
