"""Custom node plugins (report item F22): every `*.py` file and every folder
with an `__init__.py` in `<data folder>/custom_nodes/` is imported once at
startup, and the nodes it registers with `@register_node` appear in the
palette like the built-in ones (docs/NODE_AUTHORING.md, "Custom node
plugins"). The public API is `from backend.nodes import NodeBase,
register_node` plus the IMAGE helpers in `backend.nodes.base`.

**A plugin runs arbitrary Python code with the app's permissions** -- only
put files there that you trust, as with ComfyUI's custom_nodes. Turn loading
off with PIXELFLOW_DISABLE_CUSTOM_NODES=1 or {"custom_nodes": false} in
settings.json.

A plugin that fails to import is logged and reported in /api/system without
stopping the server, and one that would replace a built-in node is refused
(the built-in is restored), so a plugin can't silently change what an
existing workflow does. Files/folders starting with `_` or `.` are skipped
(helpers, `__pycache__`). In the packaged app a plugin can import only what
PixelFlow bundles (the standard library, torch, diffusers, PIL, numpy...)."""

import importlib.util
import logging
import os
import sys
import traceback
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from . import paths
from .nodes import NODE_REGISTRY

logger = logging.getLogger(__name__)

MODULE_PREFIX = "pixelflow_custom_nodes"
DISABLE_ENV = "PIXELFLOW_DISABLE_CUSTOM_NODES"


@dataclass
class PluginResult:
    name: str
    path: str
    nodes: List[str] = field(default_factory=list)
    error: Optional[str] = None


# What the last load_custom_nodes() found (shown by /api/system).
LOADED: List[PluginResult] = []


def custom_nodes_enabled(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    if environ.get(DISABLE_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return paths.load_settings(paths.ROOT_DIR).get("custom_nodes", True) is not False


def _candidates(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    found = []
    for entry in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith(("_", ".")):
            continue
        if entry.is_file() and entry.suffix == ".py":
            found.append(entry)
        elif entry.is_dir() and (entry / "__init__.py").is_file():
            found.append(entry)
    return found


def _import(entry: Path, module_name: str) -> None:
    if MODULE_PREFIX not in sys.modules:  # the plugins' common parent package
        parent = types.ModuleType(MODULE_PREFIX)
        parent.__path__ = []
        sys.modules[MODULE_PREFIX] = parent
    if entry.is_dir():
        spec = importlib.util.spec_from_file_location(
            module_name, entry / "__init__.py", submodule_search_locations=[str(entry)]
        )
    else:
        spec = importlib.util.spec_from_file_location(module_name, entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # so the plugin's own relative imports resolve
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise


def load_custom_nodes(folder: Optional[Path] = None) -> List[PluginResult]:
    """Import every plugin in `folder` (default: paths.CUSTOM_NODES_DIR) and
    return one result per plugin. Never raises for a broken plugin."""
    folder = Path(folder or paths.CUSTOM_NODES_DIR)
    results = []
    for entry in _candidates(folder):
        name = entry.stem if entry.is_file() else entry.name
        result = PluginResult(name=name, path=str(entry))
        module_name = f"{MODULE_PREFIX}.{name}"
        if module_name in sys.modules:  # already imported (a second start in the same process)
            result.nodes = sorted(k for k, cls in NODE_REGISTRY.items() if getattr(cls, "PLUGIN", None) == name)
            results.append(result)
            continue
        before = dict(NODE_REGISTRY)
        try:
            _import(entry, module_name)
        except Exception:  # noqa: BLE001 - any plugin error is reported, never fatal
            result.error = traceback.format_exc(limit=8).strip()
            logger.error("custom node plugin %s failed to load:\n%s", entry, result.error)
        replaced = sorted(k for k, cls in before.items() if NODE_REGISTRY.get(k) is not cls)
        for key in replaced:
            NODE_REGISTRY[key] = before[key]
        added = sorted(k for k in NODE_REGISTRY if k not in before)
        for key in added:
            NODE_REGISTRY[key].PLUGIN = name
        if replaced:
            message = f"refused to replace existing node(s): {', '.join(replaced)}"
            result.error = f"{result.error}\n{message}" if result.error else message
            logger.error("custom node plugin %s %s", entry, message)
        result.nodes = added
        if added:
            logger.info("custom node plugin %s: %s", name, ", ".join(added))
        results.append(result)
    LOADED[:] = results
    return results


def describe() -> List[dict]:
    return [asdict(result) for result in LOADED]
