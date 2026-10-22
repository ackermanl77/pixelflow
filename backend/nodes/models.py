"""Model discovery, architecture detection, downloading and the pipeline cache.

- Discovery (B13): list_models() scans models/<kind>/ for single files and
  diffusers folders, plus diffusers pipelines already in the Hugging Face
  cache, for the checkpoint/LoRA combos and GET /api/models/{kind}.
- Architecture (B4): hub repos and diffusers folders are identified from
  model_index.json's pipeline class; single .safetensors/.ckpt files from
  their state-dict keys (diffusers' infer_diffusers_model_type). SD1.x/2.x,
  SDXL, SD3 and Flux load into their own pipeline classes.
- Downloads (B19): a hub id that isn't cached is fetched with
  snapshot_download (only the files the pipeline loads), reporting bytes to
  the node's on_progress, and fails with a clear message when offline.
- Precision (P2): auto/fp16/bf16/fp32, part of the cache key.
- Cache (P4/P5): pipelines stay on the GPU while they fit in its memory
  minus a working reserve; older ones are moved to CPU RAM (a second tier
  bounded by its own budget) and moved back on reuse. A model too big for
  the GPU on its own runs with diffusers' model CPU offload.

torch and diffusers are imported lazily (see runtime.py).
"""

import io
import json
import logging
import os
import re
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from . import runtime
from .. import config, model_roots, paths
from ..paths import PathTraversalError, safe_path

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "stabilityai/sd-turbo"
CHECKPOINT_EXTENSIONS = (".safetensors", ".ckpt")
LORA_EXTENSIONS = (".safetensors",)
# Textual-inversion embeddings (B22), used in prompts as embedding:name (nodes/embeddings.py).
EMBEDDING_EXTENSIONS = (".safetensors", ".pt", ".bin")
# Split-file components (B26): ComfyUI's diffusion_models / text_encoders / vae single files.
COMPONENT_EXTENSIONS = (".safetensors", ".sft")
MODEL_KINDS = {
    "checkpoints": CHECKPOINT_EXTENSIONS,
    "loras": LORA_EXTENSIONS,
    "embeddings": EMBEDDING_EXTENSIONS,
    "diffusion_models": COMPONENT_EXTENSIONS,
    "text_encoders": COMPONENT_EXTENSIONS,
    "vae": COMPONENT_EXTENSIONS,
}

DTYPE_CHOICES = ["auto", "fp16", "bf16", "fp32"]

ARCH_SD = "sd"  # Stable Diffusion 1.x / 2.x
ARCH_SDXL = "sdxl"
ARCH_SD3 = "sd3"
ARCH_FLUX = "flux"
ARCH_OTHER = "other"

# infer_diffusers_model_type() result -> (arch, diffusers pipeline class).
_SINGLE_FILE_TYPES = {
    "v1": (ARCH_SD, "StableDiffusionPipeline"),
    "v2": (ARCH_SD, "StableDiffusionPipeline"),
    "inpainting": (ARCH_SD, "StableDiffusionInpaintPipeline"),
    "inpainting_v2": (ARCH_SD, "StableDiffusionInpaintPipeline"),
    "xl_base": (ARCH_SDXL, "StableDiffusionXLPipeline"),
    "playground-v2-5": (ARCH_SDXL, "StableDiffusionXLPipeline"),
    "xl_inpaint": (ARCH_SDXL, "StableDiffusionXLInpaintPipeline"),
    "xl_refiner": (ARCH_SDXL, "StableDiffusionXLImg2ImgPipeline"),
    "sd3": (ARCH_SD3, "StableDiffusion3Pipeline"),
    "sd35_medium": (ARCH_SD3, "StableDiffusion3Pipeline"),
    "sd35_large": (ARCH_SD3, "StableDiffusion3Pipeline"),
    "flux-dev": (ARCH_FLUX, "FluxPipeline"),
    "flux-schnell": (ARCH_FLUX, "FluxPipeline"),
}


# ---------------------------------------------------------------- discovery


def _checkpoint_dir() -> Path:
    return paths.CHECKPOINT_DIR


def _model_dir(kind: str) -> Path:
    """PixelFlow's own folder for `kind` (downloads land here); model_roots.model_dirs()
    adds the ComfyUI library's folders after it (B25)."""
    return model_roots.model_dirs(kind)[0]


def list_local_models(folder: Path, extensions: Iterable[str], diffusers_dirs: bool) -> List[str]:
    """Relative ('/'-separated) names of model files under `folder` (any
    depth), plus, when `diffusers_dirs`, folders holding a model_index.json
    (not descended into). Hidden files/folders are skipped."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    extensions = tuple(e.lower() for e in extensions)
    found: List[str] = []
    for current, dirnames, filenames in os.walk(folder):
        current_path = Path(current)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if diffusers_dirs and current_path != folder and (current_path / "model_index.json").is_file():
            found.append(current_path.relative_to(folder).as_posix())
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            if not name.startswith(".") and name.lower().endswith(extensions):
                found.append((current_path / name).relative_to(folder).as_posix())
    return sorted(found, key=str.lower)


_HUB_SCAN_TTL_S = 30.0
_hub_scan_cache: Dict[str, Any] = {"at": -1e9, "repos": []}


def list_cached_hub_pipelines() -> List[str]:
    """Repo ids of diffusers pipelines (a model_index.json in some cached
    revision) already in the local Hugging Face cache. The scan is reused
    for a few seconds: INPUT_TYPES() (and so this) runs on every
    /api/nodes call and every prompt validation."""
    now = time.monotonic()
    if now - _hub_scan_cache["at"] < _HUB_SCAN_TTL_S:
        return list(_hub_scan_cache["repos"])
    repos = _scan_hub_cache()
    _hub_scan_cache.update(at=now, repos=repos)
    return list(repos)


def _scan_hub_cache() -> List[str]:
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
    except Exception:  # noqa: BLE001 - no cache yet, or an unreadable one
        return []
    repos = []
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        if any(f.file_name == "model_index.json" for rev in repo.revisions for f in rev.files):
            repos.append(repo.repo_id)
    return sorted(repos, key=str.lower)


def list_models(kind: str) -> List[str]:
    """What GET /api/models/{kind} returns, and the checkpoint/LoRA combos."""
    if kind not in MODEL_KINDS:
        raise KeyError(kind)
    local = model_roots.merge_names(
        list_local_models(folder, MODEL_KINDS[kind], diffusers_dirs=kind == "checkpoints") for folder in model_roots.model_dirs(kind)
    )
    if kind != "checkpoints":
        return local
    names = local + [r for r in list_cached_hub_pipelines() if r not in local]
    if DEFAULT_CHECKPOINT not in names:
        names.append(DEFAULT_CHECKPOINT)
    return names


def local_model_path(kind: str, name: str) -> Optional[Path]:
    """`name` resolved under models/<kind>/ (or a ComfyUI library folder of
    that kind, B25), or None when it isn't a local model there (e.g. a hub
    repo id). safe_path() stops '..' or absolute names from reading outside
    the folders."""
    return model_roots.find(kind, name)


# ------------------------------------------------------------- architecture


def arch_from_single_file_type(model_type: str) -> Tuple[str, str]:
    try:
        return _SINGLE_FILE_TYPES[model_type]
    except KeyError:
        raise ValueError(
            f"unsupported checkpoint type '{model_type}' - PixelFlow loads SD 1.x/2.x, SDXL, SD3 and Flux checkpoints"
        ) from None


def arch_from_pipeline_class(class_name: str) -> str:
    name = class_name or ""
    if name.startswith("StableDiffusionXL"):
        return ARCH_SDXL
    if name.startswith("StableDiffusion3"):
        return ARCH_SD3
    if name.startswith("Flux"):
        return ARCH_FLUX
    if name.startswith("StableDiffusion"):
        return ARCH_SD
    return ARCH_OTHER


class _LazySafetensors:
    """A read-only mapping over a .safetensors file that loads a tensor
    only when it's looked up, so detecting a multi-GB checkpoint's type
    doesn't read the whole file."""

    def __init__(self, path: Path) -> None:
        from safetensors import safe_open

        self._file = safe_open(str(path), framework="pt")
        self._keys = set(self._file.keys())

    def __contains__(self, key) -> bool:
        return key in self._keys

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return self._file.get_tensor(key)

    def keys(self):
        return self._keys

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def get(self, key, default=None):
        return self[key] if key in self._keys else default

    def items(self):
        return ((k, self[k]) for k in self._keys)


def detect_single_file_arch(path: Path) -> Tuple[str, str]:
    from diffusers.loaders.single_file_utils import infer_diffusers_model_type

    if path.suffix.lower() == ".safetensors":
        checkpoint = _LazySafetensors(path)
    else:
        from diffusers.loaders.single_file_utils import load_single_file_checkpoint

        checkpoint = load_single_file_checkpoint(str(path))
    return arch_from_single_file_type(infer_diffusers_model_type(checkpoint))


def read_pipeline_class(folder: Path) -> str:
    try:
        index = json.loads((Path(folder) / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(index.get("_class_name", "")) if isinstance(index, dict) else ""


# ---------------------------------------------------------------- downloads

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".msgpack", ".onnx", ".onnx_data", ".h5", ".ot")
_VARIANT = re.compile(r"\.(fp16|bf16|fp32|non_ema|ema)(-\d+-of-\d+)?\.(safetensors|bin)$")


def pipeline_components(model_index: Dict[str, Any]) -> List[str]:
    """The component folders a model_index.json names (unet, vae, ...)."""
    return sorted(
        key
        for key, value in model_index.items()
        if not key.startswith("_") and isinstance(value, list) and len(value) == 2 and value[0] is not None
    )


def select_pipeline_files(files: Iterable[str], components: Iterable[str]) -> List[str]:
    """Which files of a diffusers repo to download, the way diffusers'
    from_pretrained() picks them: model_index.json plus each component
    folder's configs and its weights - safetensors over .bin, the default
    (non-fp16/bf16) variant when there is one - and nothing else (root
    single-file checkpoints, flax/onnx copies, sample images...)."""
    components = set(components)
    keep = ["model_index.json"]
    by_folder: Dict[str, List[str]] = {}
    for name in files:
        if "/" not in name:
            continue
        folder = name.split("/", 1)[0]
        if folder in components:
            by_folder.setdefault(folder, []).append(name)
    for items in by_folder.values():
        weights = [f for f in items if f.lower().endswith(_WEIGHT_SUFFIXES)]
        keep += [f for f in items if f not in weights]
        chosen = [f for f in weights if f.endswith(".safetensors")] or [f for f in weights if f.endswith(".bin")]
        keep += [f for f in chosen if not _VARIANT.search(f)] or chosen
    return sorted(set(keep))


def _snapshot_is_complete(folder: Path) -> bool:
    try:
        index = json.loads((folder / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    for component in pipeline_components(index):
        sub = folder / component
        if not sub.is_dir() or not any(sub.iterdir()):
            return False
    return True


def cached_snapshot(repo_id: str) -> Optional[Path]:
    """The cached snapshot folder of `repo_id` when it already holds every
    pipeline component, without any network access. (snapshot_download's
    own local_files_only mode refuses snapshots that miss files diffusers
    never needs, like README.md.)"""
    try:
        from huggingface_hub import try_to_load_from_cache

        index = try_to_load_from_cache(repo_id, "model_index.json")
    except Exception:  # noqa: BLE001 - an invalid repo id, an unreadable cache
        return None
    if not isinstance(index, str):
        return None
    folder = Path(index).parent
    return folder if _snapshot_is_complete(folder) else None


def _hub_errors() -> Tuple[type, ...]:
    """Exception types that mean "couldn't talk to the Hub"."""
    from huggingface_hub import errors

    types: List[type] = [errors.HfHubHTTPError, errors.LocalEntryNotFoundError, OSError]
    for name in ("OfflineModeIsEnabled",):
        if hasattr(errors, name):
            types.append(getattr(errors, name))
    try:
        import httpx

        types.append(httpx.HTTPError)
    except ImportError:
        pass
    try:
        import requests

        types.append(requests.exceptions.RequestException)
    except ImportError:
        pass
    return tuple(types)


def _progress_tqdm(on_progress: Optional[Callable[..., None]]):
    """A tqdm subclass for snapshot_download that forwards byte progress
    to `on_progress(done, total, unit="bytes")` (at most ~4 times a second)
    and never writes to a console (the windowed app has no stderr)."""
    from tqdm.auto import tqdm as base_tqdm

    class _Progress(base_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = False
            kwargs["file"] = io.StringIO()
            super().__init__(*args, **kwargs)
            self._last_report = 0.0

        def update(self, n=1):
            result = super().update(n)
            if on_progress is not None and self.unit == "B":
                now = time.monotonic()
                total = int(self.total or 0)
                if now - self._last_report >= 0.25 or (total and self.n >= total):
                    self._last_report = now
                    on_progress(int(self.n), max(total, int(self.n)), unit="bytes")
            return result

    return _Progress


def ensure_hub_snapshot(repo_id: str, on_progress: Optional[Callable[..., None]] = None) -> Path:
    """The local folder of `repo_id`'s cached snapshot, downloading the
    pipeline's files first when they aren't cached yet."""
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    cached = cached_snapshot(repo_id)
    if cached is not None:
        return cached

    hub_errors = _hub_errors()
    try:
        index_path = hf_hub_download(repo_id, "model_index.json")
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        wanted = select_pipeline_files(list_repo_files(repo_id), pipeline_components(index))
        logger.info("downloading %d files of '%s' from the Hugging Face Hub", len(wanted), repo_id)
        return Path(snapshot_download(repo_id, allow_patterns=wanted, tqdm_class=_progress_tqdm(on_progress)))
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            f"'{repo_id}' is not a file or folder in models/checkpoints/ and not a model on the Hugging Face Hub"
        ) from exc
    except hub_errors as exc:
        raise RuntimeError(
            f"Checkpoint '{repo_id}' isn't downloaded yet and the Hugging Face Hub can't be reached "
            f"({type(exc).__name__}). Connect to the internet once to download it, or put the model "
            "in models/checkpoints/."
        ) from exc


# ---------------------------------------------------------------- precision


def resolve_dtype_name(option: str, arch: str, device: str, bf16_supported: bool = True) -> str:
    """The precision a pipeline loads in. "auto": fp16 for SD 1.x/2.x and
    SDXL on CUDA (SDXL's own VAE upcast is left on), bf16 for SD3/Flux when
    the GPU supports it, fp32 on CPU. fp16 on CPU becomes fp32 (most CPU
    kernels don't support half precision)."""
    option = option if option in DTYPE_CHOICES else "auto"
    if device != "cuda":
        return "bf16" if option == "bf16" else "fp32"
    if option == "auto":
        if arch in (ARCH_SD3, ARCH_FLUX):
            return "bf16" if bf16_supported else "fp16"
        return "fp16"
    if option == "bf16" and not bf16_supported:
        return "fp16"
    return option


def torch_dtype(name: str):
    t = runtime.torch()
    return {"fp16": t.float16, "bf16": t.bfloat16, "fp32": t.float32}[name]


# -------------------------------------------------------------------- cache

GB = 1024**3


def _env_gb(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except ValueError:
        return default


# Upper bound on pipelines kept (both tiers together).
MAX_CACHED_PIPELINES = 4


def vram_reserve_bytes() -> int:
    """VRAM kept free for activations/VAE decode (PIXELFLOW_VRAM_RESERVE_GB, default 3)."""
    return int(_env_gb("PIXELFLOW_VRAM_RESERVE_GB", 3.0) * GB)


def ram_cache_bytes() -> int:
    """CPU RAM the second cache tier may use (PIXELFLOW_RAM_CACHE_GB, default 8; 0 disables it)."""
    return int(_env_gb("PIXELFLOW_RAM_CACHE_GB", 8.0) * GB)


def estimate_pipeline_bytes(pipe) -> int:
    """Sum of parameter and buffer sizes over the pipeline's torch modules."""
    total = 0
    components = getattr(pipe, "components", None) or {}
    seen = set()
    for module in components.values():
        if module is None or not hasattr(module, "parameters") or id(module) in seen:
            continue
        seen.add(id(module))
        for tensor in list(module.parameters()) + list(getattr(module, "buffers", lambda: [])()):
            total += tensor.numel() * tensor.element_size()
    return total


@dataclass
class CacheEntry:
    pipe: Any
    nbytes: int
    placement: str  # "gpu" | "cpu" | "offload"
    arch: str = ARCH_SD
    dtype: str = "fp32"
    info: Dict[str, Any] = field(default_factory=dict)


# Per-pipeline metadata the samplers need (arch, the checkpoint's own
# scheduler config). Weak, so it goes away with the pipeline.
_pipeline_meta: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def pipeline_meta(pipe) -> Dict[str, Any]:
    try:
        return _pipeline_meta.get(pipe, {})
    except TypeError:
        return {}


def _set_meta(pipe, **meta) -> None:
    try:
        _pipeline_meta[pipe] = {**_pipeline_meta.get(pipe, {}), **meta}
    except TypeError:  # not weak-referenceable (some test fakes)
        pass


def use_channels_last(pipe) -> bool:
    """Put the UNet and VAE in channels_last memory format (report item
    P3): a free, typically single-digit-percent speed-up of the convolutions
    on NVIDIA tensor cores. Transformer-based models (SD3, Flux) have no
    UNet and are left alone. Returns whether anything was converted."""
    try:
        torch = runtime.torch()
    except ImportError:
        return False
    converted = False
    for name in ("unet", "vae"):
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, "to") and hasattr(module, "parameters"):
            module.to(memory_format=torch.channels_last)
            converted = True
    return converted


_compile_warned = False


COMPILE_MODES = {"default": "default", "max-autotune": "max-autotune-no-cudagraphs"}


def torch_compile_mode() -> str | None:
    """Opt-in torch.compile (report item P3): {"torch_compile": true} in
    settings.json (or PIXELFLOW_TORCH_COMPILE=1) compiles in the default mode;
    "max-autotune" also benchmarks kernel variants -- a slower first compile
    (measured ~4x on an RTX 5080) for a little more speed. None: off."""
    value = os.environ.get("PIXELFLOW_TORCH_COMPILE")
    if value is None:
        value = paths.load_settings(paths.ROOT_DIR).get("torch_compile")
    if value is True or str(value).strip().lower() in ("1", "true", "yes", "on", "default"):
        return COMPILE_MODES["default"]
    if str(value).strip().lower() == "max-autotune":
        return COMPILE_MODES["max-autotune"]
    return None


def maybe_compile(pipe) -> bool:
    """When enabled, wrap the UNet in torch.compile (inductor through Triton;
    on Windows the optional triton-windows package, requirements-compile.txt).
    The first run at each new size/batch compiles (minutes); later runs at
    that size are faster. Compile errors fall back to eager execution
    (dynamo's suppress_errors) instead of failing the run. Returns whether
    the UNet is compiled."""
    global _compile_warned
    unet = getattr(pipe, "unet", None)
    mode = torch_compile_mode()
    if unet is None or mode is None:
        return False
    if hasattr(unet, "_orig_mod"):
        return True  # already compiled (a pipeline moved back onto the GPU)
    import importlib.util

    if importlib.util.find_spec("triton") is None:
        if not _compile_warned:
            logger.warning("torch_compile is on but Triton isn't installed (pip install -r requirements-compile.txt); running uncompiled")
            _compile_warned = True
        return False
    torch = runtime.torch()
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
    pipe.unet = torch.compile(unet, mode=mode, fullgraph=False)
    logger.info("UNet compiled with torch.compile (the first run at each size compiles; later runs are faster)")
    return True


class PipelineCache:
    def __init__(self) -> None:
        self.entries: "OrderedDict[Tuple[str, str, str], CacheEntry]" = OrderedDict()
        self.lock = threading.RLock()

    # -- bookkeeping
    def _bytes(self, placement: str) -> int:
        return sum(e.nbytes for e in self.entries.values() if e.placement == placement)

    def _move(self, entry: CacheEntry, device: str) -> bool:
        oom = False
        try:
            entry.pipe.to(device)
        except runtime.oom_error():
            oom = True
        if oom:
            runtime.free_memory()
            return False
        if device == "cuda":
            use_channels_last(entry.pipe)
            maybe_compile(entry.pipe)
        return True

    def _make_gpu_room(self, nbytes: int, budget: int, keep: Optional[Tuple] = None) -> None:
        """Demote least-recently-used GPU pipelines to CPU RAM until
        `nbytes` more fit in `budget`."""
        demoted = False
        for key in list(self.entries):
            if self._bytes("gpu") + nbytes <= budget:
                break
            entry = self.entries[key]
            if key == keep or entry.placement == "cpu":
                continue
            logger.info("moving cached pipeline %s off the GPU to make room", key)
            if entry.placement == "offload" or not self._move(entry, "cpu"):
                del self.entries[key]
            else:
                entry.placement = "cpu"
            demoted = True
        if demoted:
            runtime.free_memory()

    def _trim(self, keep: Optional[Tuple] = None) -> None:
        """Drop least-recently-used entries past the count cap, and CPU-tier
        entries past the RAM budget."""
        dropped = False
        for key in list(self.entries):
            over_count = len(self.entries) > max(1, MAX_CACHED_PIPELINES)
            over_ram = self._bytes("cpu") > ram_cache_bytes()
            if not (over_count or over_ram):
                break
            if key == keep:
                continue
            entry = self.entries[key]
            if over_count or entry.placement == "cpu":
                logger.info("evicting cached pipeline %s to free memory", key)
                del self.entries[key]
                dropped = True
        if dropped:
            runtime.free_memory()

    def _gpu_budget(self) -> Optional[int]:
        memory = runtime.device_memory()
        if memory is None:
            return None
        free, _total = memory
        return free + self._bytes("gpu") - vram_reserve_bytes()

    # -- public API
    def get(
        self,
        ckpt_name: str,
        dtype_option: str = "auto",
        on_progress: Optional[Callable[..., None]] = None,
    ) -> CacheEntry:
        if config.test_mode():
            raise RuntimeError(
                f"model loading is disabled in test mode (PIXELFLOW_TEST_MODE); not loading '{ckpt_name}'"
            )
        device = "cuda" if runtime.cuda_available() else "cpu"
        key = (ckpt_name, device, dtype_option)
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                self.entries.move_to_end(key)
                if entry.placement == "cpu" and device == "cuda":
                    budget = self._gpu_budget()
                    if budget is not None:
                        self._make_gpu_room(entry.nbytes, budget, keep=key)
                    if self._move(entry, device):
                        entry.placement = "gpu"
                    else:
                        del self.entries[key]
                        raise _oom_loading(ckpt_name)
                return entry

            # Room in CPU RAM for the incoming weights first (they load there).
            self._trim()
            entry = _load_entry(ckpt_name, device, dtype_option, on_progress)
            if device == "cuda":
                budget = self._gpu_budget()
                if budget is not None and entry.nbytes > budget:
                    logger.info(
                        "'%s' (%.1f GB) is larger than the GPU memory budget (%.1f GB); using model CPU offload",
                        ckpt_name,
                        entry.nbytes / GB,
                        max(budget, 0) / GB,
                    )
                    entry.pipe.enable_model_cpu_offload()
                    entry.placement = "offload"
                else:
                    if budget is not None:
                        self._make_gpu_room(entry.nbytes, budget)
                    if not self._move(entry, device):
                        raise _oom_loading(ckpt_name)
                    entry.placement = "gpu"
            else:
                entry.placement = "cpu"
            self.entries[key] = entry
            self._trim(keep=key)
            return entry

    def clear(self) -> int:
        with self.lock:
            count = len(self.entries)
            self.entries.clear()
        runtime.free_memory()
        return count

    def describe(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [
                {
                    "name": key[0],
                    "device": key[1],
                    "dtype": entry.dtype,
                    "arch": entry.arch,
                    "placement": entry.placement,
                    "bytes": entry.nbytes,
                }
                for key, entry in self.entries.items()
            ]


def _oom_loading(ckpt_name: str) -> RuntimeError:
    return RuntimeError(
        f"Out of GPU memory while loading checkpoint '{ckpt_name}'. Close other GPU applications, "
        "use a smaller checkpoint, or free memory with the System panel's 'Free memory' button."
    )


def resolve_source(ckpt_name: str, on_progress=None) -> Tuple[str, Path]:
    """("file" | "dir", path) for a checkpoint name: a local file or folder
    under models/checkpoints/, else a Hugging Face Hub repo snapshot."""
    local = local_model_path("checkpoints", ckpt_name)
    if local is not None and local.is_file():
        return "file", local
    if local is not None and local.is_dir():
        return "dir", local
    return "dir", ensure_hub_snapshot(ckpt_name, on_progress)


def _load_entry(ckpt_name: str, device: str, dtype_option: str, on_progress=None) -> CacheEntry:
    kind, path = resolve_source(ckpt_name, on_progress)
    t = runtime.torch()
    bf16 = device == "cuda" and t.cuda.is_bf16_supported()
    if kind == "file":
        arch, class_name = detect_single_file_arch(path)
    else:
        class_name = read_pipeline_class(path)
        arch = arch_from_pipeline_class(class_name)
    dtype_name = resolve_dtype_name(dtype_option, arch, device, bf16)
    pipe = load_pipeline_files(kind, path, class_name, torch_dtype(dtype_name))
    try:
        pipe.set_progress_bar_config(disable=True)
    except AttributeError:
        pass
    enable_vae_slicing(pipe)
    scheduler = getattr(pipe, "scheduler", None)
    _set_meta(
        pipe,
        arch=arch,
        dtype=dtype_name,
        name=ckpt_name,
        scheduler_config=dict(scheduler.config) if scheduler is not None else None,
        scheduler_class=type(scheduler).__name__ if scheduler is not None else None,
    )
    logger.info("loaded '%s' as %s (%s, %s)", ckpt_name, type(pipe).__name__, arch, dtype_name)
    return CacheEntry(pipe=pipe, nbytes=estimate_pipeline_bytes(pipe), placement="cpu", arch=arch, dtype=dtype_name)


def load_pipeline_files(kind: str, path: Path, class_name: str, dtype):
    """Construct the pipeline (on CPU). Split out so tests can replace it."""
    import diffusers

    kwargs = {dtype_kwarg_name(): dtype}
    if kind == "file":
        logger.info("loading checkpoint file %s as %s", path, class_name)
        return getattr(diffusers, class_name).from_single_file(str(path), **kwargs)
    logger.info("loading diffusers pipeline from %s", path)
    return diffusers.DiffusionPipeline.from_pretrained(str(path), **kwargs)


def dtype_kwarg_name() -> str:
    """diffusers 0.40 renamed from_pretrained's `torch_dtype` to `dtype`
    (the old name still works but warns); older releases only know
    `torch_dtype`, and silently ignore `dtype`."""
    try:
        from diffusers.utils import deprecation_utils
    except ImportError:
        return "torch_dtype"
    return "dtype" if hasattr(deprecation_utils, "_TORCH_DTYPE_DEPRECATION_MESSAGE") else "torch_dtype"


def enable_vae_slicing(pipe) -> None:
    """Make the VAE decode a batch one image at a time. The decode of a
    whole batch at once is the usual out-of-memory point for batch_size > 1,
    and slicing costs next to nothing, so it's always on. (Pipelines derived
    with from_pipe() share this same VAE object.)"""
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()


PIPELINES = PipelineCache()
