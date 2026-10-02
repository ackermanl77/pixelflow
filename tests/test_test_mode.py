"""PIXELFLOW_TEST_MODE: the lightweight server the browser smoke test
(tests/e2e, report item T3) runs against - no torch import, no model loads."""

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import config, main
from backend.nodes import models

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8288"


@pytest.fixture
def test_mode(monkeypatch):
    monkeypatch.setenv("PIXELFLOW_TEST_MODE", "1")


@pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("", False), ("0", False)])
def test_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("PIXELFLOW_TEST_MODE", value)
    assert config.test_mode() is expected


def test_startup_skips_the_torch_warm_up(test_mode, monkeypatch):
    calls = []
    monkeypatch.setattr(main.runtime, "warm_up_in_background", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(main.queue, "start", lambda: None)
    with TestClient(main.app, base_url=BASE) as client:
        system = client.get("/api/system").json()
    assert calls == []
    assert system["test_mode"] is True and system["torch"] is None and system["loaded_models"] == []


def test_model_loading_is_refused(test_mode):
    with pytest.raises(RuntimeError, match="test mode"):
        models.PIPELINES.get("stabilityai/sd-turbo")


def test_server_imports_and_answers_without_torch():
    """What the CI e2e job relies on: only the web dependencies installed."""
    code = (
        "import sys, os\n"
        "for name in ('torch', 'diffusers', 'transformers', 'accelerate'): sys.modules[name] = None\n"
        "os.environ['PIXELFLOW_TEST_MODE'] = '1'\n"
        "from fastapi.testclient import TestClient\n"
        "from backend import main\n"
        "c = TestClient(main.app, base_url='http://127.0.0.1:8288')\n"
        "assert c.get('/api/system').json()['test_mode']\n"
        "assert 'KSampler' in c.get('/api/nodes').json()\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
