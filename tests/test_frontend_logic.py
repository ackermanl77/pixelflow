"""Runs the frontend's pure-logic unit tests (tests/js/*.test.js, plain `node --test`)
and a syntax check of the frontend scripts, when Node.js is installed."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


@pytest.mark.parametrize("script", ["frontend/app.js", "frontend/logic.js"])
def test_frontend_script_parses(script):
    result = subprocess.run([NODE, "--check", str(ROOT / script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_frontend_logic_unit_tests():
    files = sorted(str(p) for p in (ROOT / "tests" / "js").glob("*.test.js"))
    assert len(files) >= 2
    result = subprocess.run([NODE, "--test", *files], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_element_app_js_looks_up_exists_in_index_html():
    """A cheap DOM wiring check (no browser needed): every getElementById()
    id used by app.js is defined in index.html, and every Logic.* helper it
    calls is exported by logic.js."""
    import re

    app = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    logic = (ROOT / "frontend" / "logic.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'getElementById\("([^"]+)"\)', app))
    ids |= set(re.findall(r'(?:openOverlay|closeOverlay|setProgressBar)\("([^"]+)"', app))
    defined = set(re.findall(r'id="([^"]+)"', html))
    assert ids and not (ids - defined), sorted(ids - defined)
    exported = set(re.findall(r"^\s{4}(\w+),$", logic.split("return {")[-1], re.MULTILINE))
    used = set(re.findall(r"Logic\.(\w+)", app))
    assert used and not (used - exported), sorted(used - exported)
