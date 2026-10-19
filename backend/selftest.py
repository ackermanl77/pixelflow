"""``--self-test``: check a build starts its server and runs a graph, without
opening the window.

Run as ``PixelFlow.exe --self-test`` (or ``python desktop.py --self-test``)
before publishing a release. Checks:

- the bundled frontend files and, in the packaged app, the WebView2 runtime
  next to the exe (and build-info.json, reported);
- torch: its CUDA build and, when there's a GPU, a matrix product on it
  matching the CPU's (no GPU is reported, not a failure);
- diffusers/transformers: the pipeline and model classes the nodes use
  resolve (diffusers imports them lazily, so a module missing from the
  bundle only shows up here);
- the real embedded server on a free port: /api/health, the frontend page,
  /api/nodes, /api/system, and an Empty Image -> Blur -> Save Image prompt
  run through the queue until a PNG is written.

All data goes to a throwaway folder (PIXELFLOW_HOME), so the user's outputs,
history, settings and logs are not touched, and no model is loaded or
downloaded. Exit code 0 means every check passed. The windowed build has no
console, so the report is also written to ``%TEMP%/PixelFlow-selftest.txt``
(path printed at the end).

PIXELFLOW_HOME must be set before backend.paths is imported (it resolves
its folders at import), so run_self_test() does that first; desktop.main()
calls it before anything else.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, List

SELF_TEST_FLAG = "--self-test"
REPORT_NAME = "PixelFlow-selftest.txt"
FRONTEND_FILES = ("index.html", "app.js", "logic.js", "style.css")
DIFFUSERS_CLASSES = (
    "DiffusionPipeline",
    "AutoPipelineForText2Image",
    "AutoPipelineForImage2Image",
    "AutoPipelineForInpainting",
    "ControlNetModel",
    "StableDiffusionXLPipeline",
)
TRANSFORMERS_CLASSES = ("CLIPTextModel", "CLIPTokenizer", "T5EncoderModel")
SELF_TEST_PROMPT = {
    "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "color": "#3366cc"}},
    "2": {"class_type": "ImageBlur", "inputs": {"image": ["1", 0], "radius": 2.0}},
    "3": {"class_type": "SaveImage", "inputs": {"image": ["2", 0], "filename_prefix": "selftest"}},
}
PROMPT_TIMEOUT_S = 60.0


def wants_self_test(argv) -> bool:
    return SELF_TEST_FLAG in argv


def _check_files(log: Callable[[str], None]) -> bool:
    from backend import paths

    ok = True
    missing = [name for name in FRONTEND_FILES if not (paths.FRONTEND_DIR / name).is_file()]
    if missing:
        log(f"FAIL  frontend files missing from {paths.FRONTEND_DIR}: {', '.join(missing)}")
        ok = False
    else:
        log(f"ok    frontend: {len(FRONTEND_FILES)} files in {paths.FRONTEND_DIR}")
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        runtime = exe_dir / "WebView2Runtime" / "msedgewebview2.exe"
        if runtime.is_file():
            log("ok    bundled WebView2 runtime")
        else:
            log(f"FAIL  bundled WebView2 runtime missing ({runtime})")
            ok = False
        info = exe_dir / "build-info.json"
        if info.is_file():
            data = json.loads(info.read_text(encoding="utf-8"))
            log(f"info  build: {data.get('version')} ({str(data.get('git_sha', ''))[:7]}), built {data.get('built_at')}")
    else:
        log("info  source run: WebView2 comes from the system")
    return ok


def _check_torch(log: Callable[[str], None]) -> bool:
    import torch

    log(f"ok    torch {torch.__version__} (CUDA build: {torch.version.cuda or 'none'})")
    if not torch.cuda.is_available():
        log("info  no CUDA GPU available: generation would run on the CPU")
        return True
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(256, 256, generator=generator)
    b = torch.randn(256, 256, generator=generator)
    expected = a @ b
    actual = (a.cuda() @ b.cuda()).cpu()
    diff = float((actual - expected).abs().max())
    name = torch.cuda.get_device_name(0)
    if diff > 1e-2:
        log(f"FAIL  GPU {name}: matrix product differs from the CPU by {diff:.2e}")
        return False
    log(f"ok    GPU {name}: matrix product matches the CPU (max {diff:.1e})")
    return True


def _check_libraries(log: Callable[[str], None]) -> bool:
    import diffusers
    import transformers

    missing = []
    for module, names in ((diffusers, DIFFUSERS_CLASSES), (transformers, TRANSFORMERS_CLASSES)):
        for name in names:
            try:
                getattr(module, name)
            except Exception:  # noqa: BLE001 - a lazy import that fails raises various errors
                missing.append(f"{module.__name__}.{name}")
    if missing:
        log(f"FAIL  classes that don't import: {', '.join(missing)}")
        return False
    log(f"ok    diffusers {diffusers.__version__}, transformers {transformers.__version__}: pipeline classes import")
    return True


def _check_server(log: Callable[[str], None], desktop) -> bool:
    import requests

    from backend.nodes import io_nodes  # SaveImage writes to its OUTPUT_DIR

    port = desktop._free_port()
    base = desktop._url(port)
    thread = threading.Thread(target=desktop._run_server, args=(port,), daemon=True)
    thread.start()
    try:
        desktop._wait_for_server(thread, timeout_s=60.0, port=port)
        log(f"ok    server: /api/health on port {port}")
        page = requests.get(base + "/", timeout=10)
        if not page.ok or "<html" not in page.text.lower():
            log(f"FAIL  frontend page: HTTP {page.status_code}")
            return False
        nodes = requests.get(base + "/api/nodes", timeout=30).json()
        missing = [node["class_type"] for node in SELF_TEST_PROMPT.values() if node["class_type"] not in nodes]
        if missing:
            log(f"FAIL  nodes missing: {', '.join(missing)}")
            return False
        system = requests.get(base + "/api/system", timeout=60).json()
        log(f"ok    frontend page, {len(nodes)} nodes, device {system.get('device')} ({system.get('device_name')})")

        response = requests.post(base + "/api/queue", json={"prompt": SELF_TEST_PROMPT}, timeout=30)
        if not response.ok:
            log(f"FAIL  queueing the test prompt: HTTP {response.status_code} {response.text[:300]}")
            return False
        prompt_id = response.json()["prompt_id"]
        deadline = time.monotonic() + PROMPT_TIMEOUT_S
        entry: dict = {}
        while time.monotonic() < deadline:
            entry = requests.get(f"{base}/api/history/{prompt_id}", timeout=10).json()
            if entry.get("status") in ("completed", "error", "interrupted", "cancelled"):
                break
            time.sleep(0.2)
        pngs = sorted(io_nodes.OUTPUT_DIR.glob("selftest*.png"))
        if entry.get("status") != "completed" or not pngs:
            detail = entry.get("exception_message") or entry.get("status") or "timed out"
            log(f"FAIL  test prompt (Empty Image -> Blur -> Save Image): {detail}")
            return False
        log(f"ok    test prompt ran: {pngs[0].name} written")
        return True
    finally:
        server = desktop._server.get("server")
        if server is not None:
            server.should_exit = True
        thread.join(10)


def run_self_test(out: Callable[[str], None] = print, desktop=None) -> int:
    """`desktop`: the desktop module whose server helpers start the embedded
    server (desktop.main() passes itself -- in the packaged app it is
    __main__, not an importable "desktop")."""
    if desktop is None:
        import desktop
    lines: List[str] = []

    def log(line: str) -> None:
        lines.append(line)
        try:
            out(line)
        except Exception:  # noqa: BLE001 - a windowed build may have no usable stdout
            pass

    home = tempfile.mkdtemp(prefix="pixelflow_selftest_")
    saved_home = os.environ.get("PIXELFLOW_HOME")
    os.environ["PIXELFLOW_HOME"] = home
    ok = True
    try:
        from backend.config import app_version

        log(f"PixelFlow {app_version()} self-test")
        log(f"Python {sys.version.split()[0]} ({'frozen' if getattr(sys, 'frozen', False) else 'source'})")
        checks = (_check_files, _check_torch, _check_libraries, lambda log: _check_server(log, desktop))
        for check in checks:
            try:
                ok = check(log) and ok
            except Exception as exc:  # noqa: BLE001 - report, don't crash
                ok = False
                log(f"FAIL  {type(exc).__name__}: {exc}")
                log(traceback.format_exc().strip())
    finally:
        if saved_home is None:
            os.environ.pop("PIXELFLOW_HOME", None)
        else:
            os.environ["PIXELFLOW_HOME"] = saved_home
        shutil.rmtree(home, ignore_errors=True)

    log("RESULT: PASS" if ok else "RESULT: FAIL")
    report = os.path.join(tempfile.gettempdir(), REPORT_NAME)
    try:
        with open(report, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        try:
            out(f"(report written to {report})")
        except Exception:  # noqa: BLE001
            pass
    except OSError:
        pass
    return 0 if ok else 1
