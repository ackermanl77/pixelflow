"""The output cache's copy on disk, so re-queueing a graph after a restart
still reuses the sampled images (engine.graph.OutputCache).

Each cached node output lives in its own folder, named after its signature
(a content hash, stable across restarts), as plain files - PNG for images,
.npy for arrays (loaded with allow_pickle=False), JSON for everything else -
never pickle, so a tampered cache folder can't run code. Outputs that aren't
plain data (a loaded MODEL, a CONTROL_NET) aren't written; their nodes just
run again after a restart, like LoadCheckpoint always does.

`manifest.json` maps node ids to those folders and records the app version:
a different version (or a manifest that can't be read) starts empty, since a
node's code may have changed what the same inputs produce. Writing happens
on one background thread, in commit order, so a finished run isn't held up
by PNG encoding; a folder is written once and reused while its signature
stays in the cache.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
FORMAT = 1


class Unpersistable(Exception):
    """A value the disk cache doesn't store (a model, a pipeline, ...)."""


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def encode_value(value: Any, folder: Path, counter: list) -> Any:
    """`value` as a JSON descriptor, writing images/arrays into `folder`."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"v": value}
    if isinstance(value, (list, tuple)):
        return {"tuple" if isinstance(value, tuple) else "list": [encode_value(v, folder, counter) for v in value]}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise Unpersistable("dict with non-string keys")
        return {"dict": {k: encode_value(v, folder, counter) for k, v in value.items()}}
    from PIL import Image

    if isinstance(value, Image.Image):
        counter[0] += 1
        name = f"{counter[0]}.png"
        if value.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"):
            raise Unpersistable(f"image mode {value.mode}")
        value.save(folder / name, format="PNG", compress_level=1)
        info = {k: v for k, v in value.info.items() if isinstance(k, str) and _json_safe(v)}
        return {"png": name, "info": info}
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise Unpersistable("object array")
        counter[0] += 1
        name = f"{counter[0]}.npy"
        np.save(folder / name, value, allow_pickle=False)
        return {"npy": name}
    raise Unpersistable(type(value).__name__)


def decode_value(desc: Any, folder: Path) -> Any:
    if not isinstance(desc, dict) or len(desc) not in (1, 2):
        raise ValueError(f"bad cache descriptor {desc!r:.80}")
    if "v" in desc:
        return desc["v"]
    if "list" in desc:
        return [decode_value(v, folder) for v in desc["list"]]
    if "tuple" in desc:
        return tuple(decode_value(v, folder) for v in desc["tuple"])
    if "dict" in desc:
        return {k: decode_value(v, folder) for k, v in desc["dict"].items()}
    if "png" in desc:
        from PIL import Image

        with Image.open(_inside(folder, desc["png"])) as img:
            img.load()
            image = img.copy()
        image.info = dict(desc.get("info") or {})
        return image
    if "npy" in desc:
        import numpy as np

        return np.load(_inside(folder, desc["npy"]), allow_pickle=False)
    raise ValueError(f"bad cache descriptor {desc!r:.80}")


def _inside(folder: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"bad file name in cache: {name!r}")
    return folder / name


def _is_signature(name: Any) -> bool:
    return isinstance(name, str) and len(name) == 64 and all(c in "0123456789abcdef" for c in name)


class DiskStore:
    """Reads and writes the cache folder; see the module docstring."""

    def __init__(self, root: Path, version: str) -> None:
        self.root = Path(root)
        self.version = version
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="output-cache")

    def load_manifest(self) -> Dict[str, str]:
        """{node_id: signature} of the entries on disk (empty when there are
        none, or they're from another version)."""
        try:
            data = json.loads((self.root / MANIFEST).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            logger.warning("ignoring an unreadable output cache manifest in %s", self.root, exc_info=True)
            return {}
        if not isinstance(data, dict) or data.get("format") != FORMAT or data.get("version") != self.version:
            logger.info("output cache in %s is from another version; starting empty", self.root)
            return {}
        entries = data.get("entries")
        if not isinstance(entries, dict):
            return {}
        return {
            str(node_id): sig for node_id, sig in entries.items()
            if _is_signature(sig) and (self.root / sig / "entry.json").is_file()
        }

    def read(self, signature: str) -> Optional[Tuple[Tuple[Any, ...], Optional[Dict[str, Any]]]]:
        """(outputs, ui) stored under `signature`, or None when it's missing or broken."""
        if not _is_signature(signature):
            return None
        folder = self.root / signature
        try:
            data = json.loads((folder / "entry.json").read_text(encoding="utf-8"))
            outputs = decode_value(data["outputs"], folder)
            ui = data.get("ui")
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("could not read cached outputs %s", folder, exc_info=True)
            return None
        if not isinstance(outputs, tuple) or not (ui is None or isinstance(ui, dict)):
            return None
        return outputs, ui

    def save(self, snapshot: Dict[str, Tuple[str, Any]]) -> concurrent.futures.Future:
        """Write `snapshot` ({node_id: (signature, (outputs, ui) or None)};
        None = already on disk) in the background."""
        return self._executor.submit(self._save, snapshot)

    def clear(self) -> concurrent.futures.Future:
        return self._executor.submit(self._wipe)

    def flush(self) -> None:
        """Wait for every write queued so far (tests, shutdown)."""
        self._executor.submit(lambda: None).result()

    def _save(self, snapshot: Dict[str, Tuple[str, Any]]) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            kept: Dict[str, str] = {}
            for node_id, (sig, payload) in snapshot.items():
                if not _is_signature(sig):
                    continue
                folder = self.root / sig
                if (folder / "entry.json").is_file() or (payload is not None and self._write_entry(folder, *payload)):
                    kept[str(node_id)] = sig
            manifest = {"format": FORMAT, "version": self.version, "entries": kept}
            tmp = self.root / f"{MANIFEST}.{uuid.uuid4().hex}.tmp"
            tmp.write_text(json.dumps(manifest), encoding="utf-8")
            os.replace(tmp, self.root / MANIFEST)
            self._remove_unlisted(set(kept.values()))
        except OSError:
            logger.warning("could not write the output cache to %s", self.root, exc_info=True)

    def _write_entry(self, folder: Path, outputs: Tuple[Any, ...], ui: Optional[Dict[str, Any]]) -> bool:
        staging = self.root / f".{folder.name}.{uuid.uuid4().hex}.tmp"
        staging.mkdir()
        try:
            data = {"outputs": encode_value(tuple(outputs), staging, [0]), "ui": ui}
            (staging / "entry.json").write_text(json.dumps(data), encoding="utf-8")
            if folder.exists():
                shutil.rmtree(folder)
            os.replace(staging, folder)
            return True
        except Unpersistable as exc:
            logger.debug("not caching %s on disk: %s", folder.name, exc)
        except (TypeError, ValueError):  # a ui dict that isn't JSON
            logger.debug("not caching %s on disk", folder.name, exc_info=True)
        shutil.rmtree(staging, ignore_errors=True)
        return False

    def _remove_unlisted(self, keep: set) -> None:
        for child in self.root.iterdir():
            if child.is_dir() and child.name not in keep and (_is_signature(child.name) or child.name.endswith(".tmp")):
                shutil.rmtree(child, ignore_errors=True)
            elif child.is_file() and child.name.endswith(".tmp"):
                child.unlink(missing_ok=True)

    def _wipe(self) -> None:
        try:
            if self.root.is_dir():
                (self.root / MANIFEST).unlink(missing_ok=True)
                self._remove_unlisted(set())
        except OSError:
            logger.warning("could not clear the output cache in %s", self.root, exc_info=True)
