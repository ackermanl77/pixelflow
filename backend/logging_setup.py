"""One place that configures logging for both entry points (run.py's
uvicorn worker and desktop.py / the packaged exe).

The packaged app is windowed (console=False in PixelFlow.spec), so
sys.stderr is None there and anything logged only to stderr - including a
failing node's traceback - would be lost. Everything therefore also goes to
a rotating file, logs/pixelflow.log next to the other user data."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from .paths import LOG_DIR

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_FILENAME = "pixelflow.log"
MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3

# Marks the handlers setup_logging() installed, so a second call (e.g. from
# desktop.py and again from the app's startup hook) doesn't add duplicates.
_HANDLER_MARKER = "_pixelflow_handler"


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> Optional[Path]:
    """Attach a RotatingFileHandler (5 MB x 3 backups) to the root logger,
    plus a stderr handler when there is a stderr. Idempotent. Returns the
    log file's path, or None if the log folder couldn't be created (the app
    must still start on a read-only install)."""
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    existing = [h for h in root.handlers if getattr(h, _HANDLER_MARKER, False)]
    for handler in existing:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            return Path(handler.baseFilename)
    if existing:  # only the stderr handler made it last time; don't add another
        return None

    if sys.stderr is not None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        setattr(stream, _HANDLER_MARKER, True)
        root.addHandler(stream)

    log_dir = Path(log_dir) if log_dir is not None else LOG_DIR
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / LOG_FILENAME, maxBytes=MAX_LOG_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        )
    except OSError:
        logging.getLogger(__name__).warning("could not open a log file in %s", log_dir, exc_info=True)
        return None
    file_handler.setFormatter(formatter)
    setattr(file_handler, _HANDLER_MARKER, True)
    root.addHandler(file_handler)
    return Path(file_handler.baseFilename)


def remove_logging_handlers() -> None:
    """Detach and close the handlers setup_logging() added (tests use this)."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _HANDLER_MARKER, False)]:
        root.removeHandler(handler)
        handler.close()


def log_runtime_info(logger: logging.Logger) -> None:
    """Log the Python/torch/CUDA/diffusers versions and the device, so a log
    file from a user's machine shows at a glance whether the GPU is used."""
    import platform

    import torch

    try:
        import diffusers

        diffusers_version = diffusers.__version__
    except Exception:  # noqa: BLE001 - informational only
        diffusers_version = "?"
    if torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
    logger.info(
        "PixelFlow starting: python %s, torch %s (CUDA %s), diffusers %s, device %s",
        platform.python_version(),
        torch.__version__,
        torch.version.cuda or "none - CPU-only build",
        diffusers_version,
        device,
    )
