"""--self-test (backend/selftest.py): routing from desktop.main(), the report
and exit code, and the real embedded server running the test prompt."""

import pytest

pytest.importorskip("webview")  # desktop.py needs pywebview: requirements-desktop.txt only; not installed in CI

import desktop  # noqa: E402
from backend import selftest  # noqa: E402


def test_main_runs_the_self_test_without_a_window(monkeypatch):
    calls = []
    monkeypatch.setattr("sys.argv", ["PixelFlow.exe", "--self-test"])
    monkeypatch.setattr(selftest, "run_self_test", lambda desktop: calls.append(desktop) or 0)
    monkeypatch.setattr(desktop.webview, "create_window", lambda *a, **k: calls.append("window"))
    assert desktop.main() == 0
    assert calls == [desktop]


def test_a_failing_check_fails_the_run_and_writes_the_report(monkeypatch, tmp_path):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    for name in ("_check_files", "_check_torch", "_check_libraries"):
        monkeypatch.setattr(selftest, name, lambda log: True)

    def broken(log, desktop_module):
        raise RuntimeError("server did not start")

    monkeypatch.setattr(selftest, "_check_server", broken)
    lines = []
    assert selftest.run_self_test(lines.append, desktop=desktop) == 1
    report = (tmp_path / selftest.REPORT_NAME).read_text(encoding="utf-8")
    assert "FAIL  RuntimeError: server did not start" in report
    assert "RESULT: FAIL" in lines


def test_the_embedded_server_runs_the_test_prompt(monkeypatch):
    # The real uvicorn server on a free port, the conftest's per-test data folders.
    monkeypatch.setattr(desktop, "_server", {})
    lines = []
    assert selftest._check_server(lines.append, desktop), "\n".join(lines)
    assert lines[-1] == "ok    test prompt ran: selftest_00001_.png written"
