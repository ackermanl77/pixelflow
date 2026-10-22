"""Model downloads from a pasted link (report item F23): a Hugging Face or
Civitai URL becomes one file in the right models/ folder, checked against the
SHA-256 the site publishes before it gets its real name.

- Hugging Face: a file link, https://huggingface.co/<owner>/<repo>/resolve/
  <revision>/<path> (or /blob/, the page that shows the file). The hash is
  the `X-Linked-Etag` header, which the Hub sends for every LFS file (all
  model weights); a small non-LFS file has none and isn't hash-checked. A
  gated or private repo uses the token `huggingface-cli login` saved (or
  HF_TOKEN).
- Civitai: a model page (https://civitai.com/models/<id>[/<slug>]
  [?modelVersionId=<v>], the newest version when none is given) or a
  download link (https://civitai.com/api/download/models/<v>). The file,
  its SHA256 and the model type come from Civitai's public API; a model that
  needs an account uses `civitai_token` in settings.json or CIVITAI_TOKEN.

The folder is the one the caller picks, or "auto": Civitai's model type
(Checkpoint, LORA/LoCon/DoRA, TextualInversion, Controlnet, Upscaler), or for
a Hugging Face file a guess from its name and path ("lora", "controlnet",
"embedding", "esrgan"/"upscale"; else checkpoints).

The file is written as <name>.part and renamed when complete and verified,
so a cancelled, failed or mismatching download never appears in a model
list; an existing file with the same name is never overwritten. Downloads run
as background jobs (DownloadManager); /api/models/download(s) in main.py.

Resuming (F26): when the connection drops, <name>.part stays, next to a
<name>.part.dl-meta sidecar (format version, URL, source, size, SHA-256 and
the server's ETag/Last-Modified). Starting the same download again -- even
after a restart -- re-hashes the bytes already there and asks for the rest
with Range + If-Range; a server that answers 200 instead of 206 (the file
changed, or no range support) restarts it from zero. Cancelling, a hash
mismatch or a refused download discard the partial file.

Gated models (F26): a Hugging Face 401/403 carries the repository page as the
error's help_url, so the UI can open it for the user to accept the licence."""

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

from . import paths

logger = logging.getLogger(__name__)

MODEL_FOLDERS = {
    "checkpoints": lambda: paths.CHECKPOINT_DIR,
    "loras": lambda: paths.LORA_DIR,
    "embeddings": lambda: paths.EMBEDDING_DIR,
    "controlnet": lambda: paths.CONTROLNET_DIR,
    "upscale_models": lambda: paths.UPSCALE_MODEL_DIR,
    "diffusion_models": lambda: paths.DIFFUSION_MODEL_DIR,
    "text_encoders": lambda: paths.TEXT_ENCODER_DIR,
    "vae": lambda: paths.VAE_DIR,
}
CIVITAI_TYPES = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "locon": "loras",
    "dora": "loras",
    "textualinversion": "embeddings",
    "controlnet": "controlnet",
    "upscaler": "upscale_models",
}
ALLOWED_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")
CHUNK_BYTES = 1024 * 1024
TIMEOUT = (15, 120)  # connect, read

_HF_FILE = re.compile(r"^/(?P<repo>[^/]+/[^/]+)/(?:resolve|blob)/(?P<rev>[^/]+)/(?P<path>.+)$")
_CIVITAI_MODEL = re.compile(r"^/models/(?P<model>\d+)(?:/[^/]*)?/?$")
_CIVITAI_DOWNLOAD = re.compile(r"^/api/download/models/(?P<version>\d+)/?$")


class DownloadError(ValueError):
    """A link that can't be downloaded, or a download that failed its check.
    `help_url` is a page that fixes it (a gated model's licence page)."""

    def __init__(self, message: str, help_url: Optional[str] = None):
        super().__init__(message)
        self.help_url = help_url


class DownloadCancelled(Exception):
    pass


@dataclass
class Plan:
    """What resolve() found: one file to fetch, where it goes, its hash."""

    url: str
    filename: str
    kind: str
    sha256: Optional[str] = None
    size: Optional[int] = None
    source: str = ""
    headers: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------- resolving


def _guess_kind(path: str) -> str:
    lowered = path.lower()
    if "lora" in lowered or "locon" in lowered:
        return "loras"
    if "controlnet" in lowered or "control_" in lowered:
        return "controlnet"
    if "embedding" in lowered or "textual" in lowered:
        return "embeddings"
    if "esrgan" in lowered or "upscale" in lowered or re.search(r"(^|[^a-z])x[248]([^0-9]|$)", lowered):
        return "upscale_models"
    return "checkpoints"


def _hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001 - no hub package or no saved login
        return None


def _civitai_token() -> Optional[str]:
    token = os.environ.get("CIVITAI_TOKEN") or paths.load_settings(paths.ROOT_DIR).get("civitai_token")
    return token if isinstance(token, str) and token.strip() else None


def _resolve_huggingface(path: str, head: Callable) -> Plan:
    match = _HF_FILE.match(urllib.parse.unquote(path))
    if not match:
        raise DownloadError("for Hugging Face, paste the link of one file: https://huggingface.co/<owner>/<repo>/"
                            "blob/main/<file> (open the file on the model's 'Files' tab)")
    repo, revision, file_path = match.group("repo"), match.group("rev"), match.group("path")
    url = "https://huggingface.co/{}/resolve/{}/{}".format(repo, revision, urllib.parse.quote(file_path))
    headers = {}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = head(url, headers)
    if response.status_code in (401, 403):
        raise DownloadError(
            f"{repo} is gated or private: open its page, log in and accept the licence, then make sure PixelFlow "
            "has your Hugging Face token (huggingface-cli login, or HF_TOKEN)",
            help_url=f"https://huggingface.co/{repo}",
        )
    if response.status_code == 404:
        raise DownloadError(f"{file_path} is not in {repo} at {revision}")
    if response.status_code >= 400:
        raise DownloadError(f"Hugging Face answered {response.status_code} for {url}")
    etag = (response.headers.get("X-Linked-Etag") or "").strip('"').lower()
    size = response.headers.get("X-Linked-Size") or response.headers.get("Content-Length")
    return Plan(
        url=url,
        filename=Path(file_path).name,
        kind=_guess_kind(f"{repo}/{file_path}"),
        sha256=etag if re.fullmatch(r"[0-9a-f]{64}", etag) else None,
        size=int(size) if size and str(size).isdigit() else None,
        source=f"huggingface:{repo}@{revision}/{file_path}",
        headers=headers,
    )


def _resolve_civitai(path: str, query: Dict[str, List[str]], fetch_json: Callable) -> Plan:
    api = "https://civitai.com/api/v1"
    download = _CIVITAI_DOWNLOAD.match(path)
    model = _CIVITAI_MODEL.match(path)
    if download:
        version = fetch_json(f"{api}/model-versions/{download.group('version')}")
    elif model:
        wanted = (query.get("modelVersionId") or [None])[0]
        if wanted and wanted.isdigit():
            version = fetch_json(f"{api}/model-versions/{wanted}")
        else:
            versions = fetch_json(f"{api}/models/{model.group('model')}").get("modelVersions") or []
            if not versions:
                raise DownloadError("this Civitai model has no downloadable version")
            version = versions[0]
            version.setdefault("model", {})
    else:
        raise DownloadError("for Civitai, paste a model page (https://civitai.com/models/<id>...) or a download link")
    files = version.get("files") or []
    chosen = next((f for f in files if f.get("primary")), None) or next(
        (f for f in files if f.get("type") == "Model"), None) or (files[0] if files else None)
    if chosen is None:
        raise DownloadError("this Civitai version lists no files")
    model_type = str((version.get("model") or {}).get("type") or "").replace(" ", "").lower()
    sha = str((chosen.get("hashes") or {}).get("SHA256") or "").lower()
    url = chosen.get("downloadUrl") or f"https://civitai.com/api/download/models/{version.get('id')}"
    headers = {}
    token = _civitai_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    size_kb = chosen.get("sizeKB")
    return Plan(
        url=url,
        filename=chosen.get("name") or f"civitai_{version.get('id')}.safetensors",
        kind=CIVITAI_TYPES.get(model_type, "checkpoints"),
        sha256=sha if re.fullmatch(r"[0-9a-f]{64}", sha) else None,
        size=int(size_kb * 1024) if isinstance(size_kb, (int, float)) else None,
        source=f"civitai:version {version.get('id')}",
        headers=headers,
    )


def _http_head(url, headers):
    return requests.head(url, headers=headers, allow_redirects=False, timeout=TIMEOUT[0])


def _http_json(url):
    response = requests.get(url, timeout=TIMEOUT, headers={"Accept": "application/json"})
    if response.status_code == 404:
        raise DownloadError(f"not found on Civitai ({url})")
    if response.status_code >= 400:
        raise DownloadError(f"Civitai answered {response.status_code} for {url}")
    return response.json()


def resolve(link: str, kind: str = "auto", head: Callable = _http_head, fetch_json: Callable = _http_json) -> Plan:
    """The Plan for a pasted link. `kind` is a MODEL_FOLDERS key or "auto"."""
    if kind != "auto" and kind not in MODEL_FOLDERS:
        raise DownloadError(f"unknown model folder '{kind}' (use auto or one of {', '.join(MODEL_FOLDERS)})")
    parsed = urllib.parse.urlsplit((link or "").strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in ("huggingface.co", "civitai.com", "www.civitai.com", "hf.co"):
        raise DownloadError("paste an https link to a file on huggingface.co or a model on civitai.com")
    if host in ("huggingface.co", "hf.co"):
        plan = _resolve_huggingface(parsed.path, head)
    else:
        plan = _resolve_civitai(parsed.path, urllib.parse.parse_qs(parsed.query), fetch_json)
    if kind != "auto":
        plan.kind = kind
    stem, extension = os.path.splitext(re.split(r"[\\/]", plan.filename)[-1])
    if extension.lower() not in ALLOWED_EXTENSIONS:
        raise DownloadError(f"{plan.filename} is not a model file ({', '.join(ALLOWED_EXTENSIONS)})")
    plan.filename = paths.sanitize_filename(stem, default="model") + extension.lower()
    return plan


# -------------------------------------------------------------- downloading


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


META_VERSION = 1


def _meta_path(part: Path) -> Path:
    return part.with_name(part.name + ".dl-meta")


def read_resume_meta(part: Path, plan: Plan) -> Optional[dict]:
    """The sidecar of a .part this plan can resume, or None (none, unreadable,
    another format version, or a different file/hash/size)."""
    try:
        meta = json.loads(_meta_path(part).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or meta.get("version") != META_VERSION:
        return None
    if meta.get("url") != plan.url or meta.get("sha256") != plan.sha256 or meta.get("size") != plan.size:
        return None
    return meta


def _write_meta(part: Path, plan: Plan, response_headers) -> None:
    meta = {
        "version": META_VERSION, "url": plan.url, "source": plan.source, "size": plan.size, "sha256": plan.sha256,
        "etag": response_headers.get("ETag") or response_headers.get("X-Linked-Etag"),
        "last_modified": response_headers.get("Last-Modified"),
    }
    try:
        _meta_path(part).write_text(json.dumps(meta), encoding="utf-8")
    except OSError:
        logger.warning("could not write the resume sidecar for %s", part, exc_info=True)


def _discard(part: Path) -> None:
    for path in (part, _meta_path(part)):
        try:
            path.unlink()
        except OSError:
            pass


def _gated_help(plan: Plan) -> Optional[str]:
    match = re.match(r"^huggingface:(?P<repo>[^@]+)@", plan.source or "")
    return f"https://huggingface.co/{match.group('repo')}" if match else None


def download(plan: Plan, on_progress: Optional[Callable[[int, Optional[int]], None]] = None,
             cancelled: Callable[[], bool] = lambda: False, session=None) -> Path:
    """Fetch the planned file into its models/ folder and return its path,
    resuming an interrupted .part when its sidecar matches (F26)."""
    folder = MODEL_FOLDERS[plan.kind]()
    folder.mkdir(parents=True, exist_ok=True)
    target = paths.safe_path(folder, plan.filename)
    if target.exists():
        if plan.sha256 and sha256_of(target) == plan.sha256:
            return target  # already downloaded
        raise DownloadError(f"{plan.kind}/{plan.filename} already exists; rename or remove it first")
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    received = 0
    headers = dict(plan.headers)
    meta = read_resume_meta(part, plan) if part.is_file() else None
    if meta is not None:
        with open(part, "rb") as handle:
            for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
                digest.update(block)
                received += len(block)
        validator = meta.get("etag") or meta.get("last_modified")
        headers["Range"] = f"bytes={received}-"
        if validator:
            headers["If-Range"] = validator
        logger.info("resuming %s at %d bytes", plan.filename, received)
    elif part.exists():
        _discard(part)  # a leftover from another file or version: start over
    complete = False
    keep_partial = meta is not None  # a resume that fails to connect keeps what it had
    try:
        with (session or requests).get(plan.url, headers=headers, stream=True, timeout=TIMEOUT) as response:
            if response.status_code in (401, 403):
                raise DownloadError("the site refused the download: this model needs a login token "
                                    "(HF_TOKEN / huggingface-cli login, or civitai_token in settings.json) "
                                    "and, for a gated model, its licence accepted on the model page",
                                    help_url=_gated_help(plan))
            if response.status_code == 416 and received and (plan.size is None or received >= plan.size):
                response = None  # the .part already holds the whole file
            elif response.status_code >= 400:
                raise DownloadError(f"the download failed with HTTP {response.status_code}")
            mode = "ab"
            if response is not None:
                content_type = response.headers.get("Content-Type", "")
                if content_type.startswith("text/html"):
                    raise DownloadError("the site sent a web page instead of the file (a login or consent page?)")
                if received and response.status_code != 206:
                    logger.info("%s: the server sent the whole file again; restarting", plan.filename)
                    digest, received = hashlib.sha256(), 0
                if not received:
                    mode = "wb"
            total = plan.size or (received + int(response.headers["Content-Length"])
                                  if response is not None and response.headers.get("Content-Length", "").isdigit() else None)
            if response is not None:
                _write_meta(part, plan, response.headers)
                keep_partial = True
            with open(part, mode) as handle:
                for chunk in (response.iter_content(CHUNK_BYTES) if response is not None else ()):
                    if cancelled():
                        keep_partial = False
                        raise DownloadCancelled()
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if on_progress is not None:
                        on_progress(received, total)
        if plan.sha256 and digest.hexdigest() != plan.sha256:
            keep_partial = False
            raise DownloadError(f"{plan.filename}: SHA-256 doesn't match what the site lists "
                                f"({digest.hexdigest()[:12]}... instead of {plan.sha256[:12]}...); file discarded")
        os.replace(part, target)
        complete = True
        _discard(part)  # the sidecar
        logger.info("downloaded %s (%d bytes) from %s into %s", plan.filename, received, plan.source, folder)
        return target
    except requests.RequestException as exc:
        if keep_partial:
            raise DownloadError(f"the download was interrupted ({exc}); start it again to resume") from exc
        raise DownloadError(f"the download failed: {exc}") from exc
    finally:
        # Cancelled or mismatching: never leave a partial file. Anything else (a
        # dropped connection, a server error, a licence still to accept) keeps it
        # with its sidecar once data has arrived, so the next attempt resumes.
        if not complete and not keep_partial:
            _discard(part)


# --------------------------------------------------------------------- jobs


@dataclass
class DownloadJob:
    id: str
    url: str
    kind: str
    status: str = "resolving"  # resolving, downloading, done, error, cancelled
    filename: Optional[str] = None
    folder: Optional[str] = None
    bytes: int = 0
    total: Optional[int] = None
    verified: Optional[bool] = None
    error: Optional[str] = None
    help_url: Optional[str] = None  # F26: e.g. a gated model's page, to accept its licence
    started_at: float = field(default_factory=time.time)


class DownloadManager:
    """Background downloads, one thread each; at most `keep` finished jobs listed."""

    def __init__(self, resolver: Callable = resolve, downloader: Callable = download, keep: int = 20):
        self._resolve = resolver
        self._download = downloader
        self._keep = keep
        self._jobs: Dict[str, DownloadJob] = {}
        self._cancel: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start(self, url: str, kind: str = "auto") -> dict:
        job = DownloadJob(id=uuid.uuid4().hex[:12], url=url, kind=kind)
        event = threading.Event()
        with self._lock:
            self._jobs[job.id] = job
            self._cancel[job.id] = event
            finished = [j for j in self._jobs.values() if j.status in ("done", "error", "cancelled")]
            for old in finished[: max(0, len(finished) - self._keep)]:
                self._jobs.pop(old.id, None)
                self._cancel.pop(old.id, None)
        threading.Thread(target=self._run, args=(job, event), name=f"download-{job.id}", daemon=True).start()
        return asdict(job)

    def _run(self, job: DownloadJob, event: threading.Event) -> None:
        try:
            plan = self._resolve(job.url, job.kind)
            job.kind, job.filename, job.total = plan.kind, plan.filename, plan.size
            job.folder = plan.kind
            job.verified = bool(plan.sha256)
            job.status = "downloading"

            def progress(received, total):
                job.bytes, job.total = received, total

            self._download(plan, progress, event.is_set)
            job.status = "done"
        except DownloadCancelled:
            job.status = "cancelled"
        except (DownloadError, requests.RequestException, OSError, ValueError) as exc:
            job.status, job.error = "error", str(exc)
            job.help_url = getattr(exc, "help_url", None)
            logger.warning("download of %s failed: %s", job.url, exc)
        except Exception as exc:  # noqa: BLE001 - a job must never die silently
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            logger.exception("download of %s failed", job.url)

    def list(self) -> List[dict]:
        with self._lock:
            return [asdict(job) for job in self._jobs.values()]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job, event = self._jobs.get(job_id), self._cancel.get(job_id)
        if job is None or event is None or job.status not in ("resolving", "downloading"):
            return False
        event.set()
        return True


MANAGER = DownloadManager()
