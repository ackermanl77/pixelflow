"""Lazy access to torch (report item P6).

Importing torch + diffusers takes several seconds (more for the CUDA build
and in the frozen app). Node *declarations* (INPUT_TYPES etc.) don't need
either, so modules under backend/nodes import them only inside the
functions that actually run a model. The server is therefore up in about a
second, and main.py warms the import up in a background thread right after
startup so the first queued prompt doesn't pay for it.
"""

import gc
import threading
from typing import Tuple, Type

_warm_lock = threading.Lock()
_warm_started = False


def torch():
    import torch as _torch

    return _torch


def oom_error() -> Type[BaseException]:
    """torch's out-of-memory exception class. torch >= 2.5 exposes
    torch.OutOfMemoryError; older builds only the CUDA-specific name (the
    same class). Usable directly in an `except` clause (it's evaluated only
    when an exception is being matched)."""
    t = torch()
    return getattr(t, "OutOfMemoryError", t.cuda.OutOfMemoryError)


def free_memory() -> None:
    """Release unreferenced tensors and hand cached CUDA blocks back to the
    driver. A no-op on CPU-only machines beyond the gc pass (and torch isn't
    imported just for this)."""
    gc.collect()
    import sys

    t = sys.modules.get("torch")
    if t is not None and t.cuda.is_available():
        t.cuda.empty_cache()


def cuda_available() -> bool:
    return torch().cuda.is_available()


def device_memory() -> Tuple[int, int] | None:
    """(free, total) bytes of the current CUDA device, or None without CUDA."""
    t = torch()
    if not t.cuda.is_available():
        return None
    free, total = t.cuda.mem_get_info()
    return int(free), int(total)


def warm_up_in_background(callback=None) -> bool:
    """Import torch and diffusers on a daemon thread (once). `callback`, if
    given, runs on that thread after the imports. Returns False when a warm
    up was already started."""
    global _warm_started
    with _warm_lock:
        if _warm_started:
            return False
        _warm_started = True

    def _run():
        try:
            torch()
            import diffusers  # noqa: F401
        except Exception:  # noqa: BLE001 - the first real use reports the error properly
            import logging

            logging.getLogger(__name__).warning("background import of torch/diffusers failed", exc_info=True)
            return
        if callback is not None:
            callback()

    threading.Thread(target=_run, name="pixelflow-warmup", daemon=True).start()
    return True
