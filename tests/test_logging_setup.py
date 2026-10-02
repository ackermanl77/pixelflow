"""Rotating log file for the windowed packaged app (report item R2)."""

import logging
import logging.handlers

import pytest

from backend import logging_setup


@pytest.fixture(autouse=True)
def _clean_handlers():
    logging_setup.remove_logging_handlers()
    yield
    logging_setup.remove_logging_handlers()


def test_setup_logging_writes_to_a_rotating_file(tmp_path):
    log_file = logging_setup.setup_logging(tmp_path / "logs")
    assert log_file == tmp_path / "logs" / "pixelflow.log"
    logging.getLogger("backend.test").error("node exploded")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "node exploded" in log_file.read_text(encoding="utf-8")
    rotating = [h for h in logging.getLogger().handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert rotating[0].maxBytes == 5 * 1024 * 1024 and rotating[0].backupCount == 3


def test_setup_logging_is_idempotent(tmp_path):
    first = logging_setup.setup_logging(tmp_path)
    count = len(logging.getLogger().handlers)
    assert logging_setup.setup_logging(tmp_path / "elsewhere") == first
    assert len(logging.getLogger().handlers) == count


def test_no_stderr_still_logs_to_the_file(tmp_path, monkeypatch):
    # A windowed PyInstaller build has sys.stderr = None.
    monkeypatch.setattr(logging_setup.sys, "stderr", None)
    log_file = logging_setup.setup_logging(tmp_path)
    assert log_file is not None
    assert not any(
        type(h) is logging.StreamHandler for h in logging.getLogger().handlers if getattr(h, "_pixelflow_handler", False)
    )


def test_runtime_info_names_torch_and_cuda(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        logging_setup.log_runtime_info(logging.getLogger("backend.main"))
    assert "torch" in caplog.text and "CUDA" in caplog.text
