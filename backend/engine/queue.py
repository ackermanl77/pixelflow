import asyncio
import json
import logging
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

from .graph import ExecutionInterrupted, NodeExecutionError, OutputCache, execute_prompt

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                logger.debug("dropping dead websocket connection", exc_info=True)
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


DEFAULT_MAX_HISTORY = 200


class PendingQueue:
    """The prompts waiting to run: an asyncio-friendly deque that can also
    take an item at the front (Queue Front / Ctrl+Shift+Enter). Items are
    (prompt_id, prompt, extra_pnginfo) tuples."""

    def __init__(self) -> None:
        self._items: deque = deque()
        self._event = asyncio.Event()

    async def put(self, item, front: bool = False) -> None:
        if front:
            self._items.appendleft(item)
        else:
            self._items.append(item)
        self._event.set()

    async def get(self):
        while not self._items:
            self._event.clear()
            await self._event.wait()
        return self._items.popleft()

    def get_nowait(self):
        if not self._items:
            raise asyncio.QueueEmpty
        return self._items.popleft()

    def task_done(self) -> None:  # asyncio.Queue compatibility
        pass

    def ids(self) -> List[str]:
        return [item[0] for item in self._items]

    def qsize(self) -> int:
        return len(self._items)

# A history entry in one of these states is finished: it's what gets
# persisted, and the only kind _record_history() ever evicts.
TERMINAL_STATUSES = frozenset({"completed", "error", "interrupted", "cancelled"})


class ExecutionQueue:
    """A single-worker async job queue, mirroring ComfyUI's queue/history model:
    prompts are submitted, run one at a time in a background worker, and
    progress/results are broadcast over the websocket as they happen.

    Each history entry records the prompt, the frontend workflow it came
    from, its status, outputs or error details, and queued/started/finished
    timestamps. When `history_path` is set, every entry that reaches a
    terminal status is appended to that JSON-lines file, and load_history()
    restores the most recent ones after a restart. Node outputs are reused
    between runs through `output_cache` (see engine.graph.OutputCache).
    """

    def __init__(
        self,
        manager: ConnectionManager,
        max_history: int = DEFAULT_MAX_HISTORY,
        history_path: Optional[Path] = None,
    ) -> None:
        self.manager = manager
        self.pending = PendingQueue()
        self.history: Dict[str, Any] = {}
        self.max_history = max_history
        self.history_path = history_path
        self.output_cache = OutputCache()
        self._cancelled: set = set()
        self._worker_task: Any = None
        self.running_prompt_id: Optional[str] = None
        # Set by interrupt(), polled by execute_prompt() between nodes and on
        # every KSampler diffusion step (via its on_progress callback) from
        # the worker thread - hence a threading.Event, not an asyncio one.
        self._interrupt_event = threading.Event()

    def interrupt(self, prompt_id: Optional[str] = None) -> Optional[str]:
        """Ask the currently running prompt to stop at its next checkpoint
        (the next node, or the next diffusion step). With `prompt_id`, only
        when that is the prompt running - so a Cancel click on a row can't
        stop a different prompt that started in the meantime. Returns the
        interrupted prompt's id, or None if nothing (matching) is running."""
        running = self.running_prompt_id
        if running is None or (prompt_id is not None and prompt_id != running):
            return None
        self._interrupt_event.set()
        return running

    def cancel(self, prompt_id: str) -> bool:
        """Cancel a prompt that is still queued. Returns False for an unknown
        id or one that's already running/finished - use interrupt() to stop
        the running one."""
        entry = self.history.get(prompt_id)
        if entry is None or entry.get("status") != "queued":
            return False
        self._cancelled.add(prompt_id)
        self._record_history(prompt_id, {"status": "cancelled", "finished_at": time.time()})
        return True

    def cancel_all(self) -> List[str]:
        """Cancel every still-queued prompt; returns their ids."""
        cancelled = [pid for pid, e in list(self.history.items()) if e.get("status") == "queued"]
        for pid in cancelled:
            self.cancel(pid)
        return cancelled

    def clear_finished(self) -> List[str]:
        """Delete every finished history entry; returns their ids."""
        finished = [pid for pid, e in list(self.history.items()) if e.get("status") in TERMINAL_STATUSES]
        for pid in finished:
            self.delete_history(pid)
        return finished

    def delete_history(self, prompt_id: str) -> bool:
        """Remove a finished (completed/error/cancelled) history entry. Returns
        False for an unknown id or one that's still queued/running."""
        entry = self.history.get(prompt_id)
        if entry is None or entry.get("status") in ("queued", "running"):
            return False
        del self.history[prompt_id]
        self._append_history_line({"prompt_id": prompt_id, "deleted": True})
        return True

    def queue_state(self) -> Dict[str, Any]:
        """What GET /api/queue reports: the running prompt's id (or None) and
        the still-pending ids in the order they'll run."""
        queued = [pid for pid, entry in self.history.items() if entry.get("status") == "queued"]
        order = self.pending.ids()
        # Run order (a Queue Front prompt jumps ahead); anything queued but
        # not in the deque (not expected) still shows, at the end.
        pending = [pid for pid in order if pid in queued] + [pid for pid in queued if pid not in order]
        return {"running": self.running_prompt_id, "pending": pending}

    def history_page(self, limit: Optional[int] = None, before: Optional[str] = None) -> Dict[str, Any]:
        """A slice of the history, oldest first like the full dict: only
        entries older than `before` (a prompt id; an unknown id gives an
        empty page), and at most the newest `limit` of those."""
        items = list(self.history.items())
        if before is not None:
            ids = [pid for pid, _ in items]
            if before not in ids:
                return {}
            items = items[: ids.index(before)]
        if limit is not None:
            items = items[-limit:] if limit > 0 else []
        return dict(items)

    def _record_history(self, prompt_id: str, fields: Dict[str, Any]) -> None:
        """Merge `fields` into the prompt's history entry (creating it if
        needed), persist it once it's terminal, and evict the oldest
        *terminal* entries past max_history. Queued/running entries are
        never evicted, so cancel() keeps working however long the queue is."""
        entry = {**self.history.get(prompt_id, {}), **fields}
        # dict preserves insertion order, and updating an existing key keeps
        # its original position, so the front is always the oldest prompt.
        self.history[prompt_id] = entry
        if fields.get("status") in TERMINAL_STATUSES:
            self._append_history_line({"prompt_id": prompt_id, "entry": entry})
        while len(self.history) > self.max_history:
            oldest_done = next(
                (pid for pid, e in self.history.items() if e.get("status") in TERMINAL_STATUSES), None
            )
            if oldest_done is None:
                break
            del self.history[oldest_done]

    def _append_history_line(self, record: Dict[str, Any]) -> None:
        if self.history_path is None:
            return
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.warning("could not append to history file %s", self.history_path, exc_info=True)

    def load_history(self) -> int:
        """Restore the most recent `max_history` finished entries from
        `history_path` (ahead of anything already in memory), compacting the
        file when it has grown well past that. Unreadable lines (e.g. a
        write cut short by a crash) are skipped. Returns the number loaded."""
        if self.history_path is None or not self.history_path.is_file():
            return 0
        restored: Dict[str, Any] = {}
        line_count = 0
        try:
            with open(self.history_path, encoding="utf-8") as handle:
                for line in handle:
                    line_count += 1
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    prompt_id = record.get("prompt_id") if isinstance(record, dict) else None
                    if not isinstance(prompt_id, str):
                        continue
                    restored.pop(prompt_id, None)
                    if not record.get("deleted") and isinstance(record.get("entry"), dict):
                        restored[prompt_id] = record["entry"]
        except OSError:
            logger.warning("could not read history file %s", self.history_path, exc_info=True)
            return 0

        kept = dict(list(restored.items())[-self.max_history :]) if self.max_history > 0 else {}
        self.history = {**kept, **{pid: e for pid, e in self.history.items() if pid not in kept}}
        if line_count > 2 * max(self.max_history, 1):
            self._rewrite_history_file(kept)
        return len(kept)

    def _rewrite_history_file(self, entries: Dict[str, Any]) -> None:
        tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                for prompt_id, entry in entries.items():
                    handle.write(json.dumps({"prompt_id": prompt_id, "entry": entry}, default=str) + "\n")
            tmp.replace(self.history_path)
        except OSError:
            logger.warning("could not compact history file %s", self.history_path, exc_info=True)

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def submit(
        self, prompt: Dict[str, Any], extra_pnginfo: Optional[Dict[str, Any]] = None, front: bool = False
    ) -> str:
        """Queue a prompt (at the front with `front`). `extra_pnginfo` (e.g.
        {"workflow": <litegraph graph>}) is embedded into saved PNGs
        alongside the prompt itself."""
        prompt_id = uuid.uuid4().hex[:12]
        extra_pnginfo = extra_pnginfo or {}
        entry: Dict[str, Any] = {"status": "queued", "prompt": prompt, "queued_at": time.time()}
        if "workflow" in extra_pnginfo:
            entry["workflow"] = extra_pnginfo["workflow"]
        self._record_history(prompt_id, entry)
        await self.pending.put((prompt_id, prompt, extra_pnginfo), front=front)
        await self.manager.broadcast({"type": "queued", "prompt_id": prompt_id})
        return prompt_id

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            prompt_id, prompt, extra_pnginfo = await self.pending.get()
            if prompt_id in self._cancelled:
                self._cancelled.discard(prompt_id)
                self.pending.task_done()
                continue
            self._interrupt_event.clear()
            self.running_prompt_id = prompt_id
            self._record_history(prompt_id, {"status": "running", "started_at": time.time()})
            await self.manager.broadcast({"type": "execution_start", "prompt_id": prompt_id})

            def _send(message: Dict[str, Any]) -> None:
                asyncio.run_coroutine_threadsafe(self.manager.broadcast(message), loop)

            # The prompt holds exactly the nodes that will run or be served
            # from the cache (validate_prompt() pruned the rest), so the
            # UI can show "node 3 / 7" overall progress.
            node_count = len(prompt)
            done = {"n": 0}

            def on_node_start(node_id: str, class_type: str, _pid: str = prompt_id) -> None:
                done["n"] += 1
                _send(
                    {
                        "type": "executing",
                        "prompt_id": _pid,
                        "node_id": node_id,
                        "class_type": class_type,
                        "node_index": done["n"],
                        "node_count": node_count,
                    }
                )

            def on_node_cached(node_id: str, class_type: str, _pid: str = prompt_id) -> None:
                done["n"] += 1
                _send(
                    {
                        "type": "execution_cached",
                        "prompt_id": _pid,
                        "nodes": [node_id],
                        "node_index": done["n"],
                        "node_count": node_count,
                    }
                )

            def on_progress(node_id: str, step: int, total: int, info: Dict[str, Any], _pid: str = prompt_id) -> None:
                message = {"type": "progress", "prompt_id": _pid, "node_id": node_id, "step": step, "total": total}
                message.update({k: v for k, v in info.items() if k != "preview"})
                _send(message)
                if info.get("preview"):
                    _send({"type": "preview", "prompt_id": _pid, "node_id": node_id, "image": info["preview"]})

            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: execute_prompt(
                        prompt,
                        on_node_start,
                        should_interrupt=self._interrupt_event.is_set,
                        on_progress_detail=on_progress,
                        extra_pnginfo=extra_pnginfo,
                        output_cache=self.output_cache,
                        on_node_cached=on_node_cached,
                    ),
                )
                self._record_history(
                    prompt_id, {"status": "completed", "outputs": result, "finished_at": time.time()}
                )
                await self.manager.broadcast(
                    {"type": "execution_success", "prompt_id": prompt_id, "outputs": result}
                )
            except ExecutionInterrupted:
                logger.info("prompt %s was interrupted", prompt_id)
                self._record_history(prompt_id, {"status": "interrupted", "finished_at": time.time()})
                await self.manager.broadcast({"type": "execution_interrupted", "prompt_id": prompt_id})
            except Exception as exc:  # noqa: BLE001 - surface any node error to the client
                logger.exception("prompt %s failed during execution", prompt_id)
                details = _error_details(exc)
                self._record_history(prompt_id, {"status": "error", **details, "finished_at": time.time()})
                await self.manager.broadcast({"type": "execution_error", "prompt_id": prompt_id, **details})
            finally:
                self.running_prompt_id = None
                self._interrupt_event.clear()
                self.pending.task_done()


def _error_details(exc: BaseException) -> Dict[str, Any]:
    """The fields an execution_error message / error history entry carries:
    `error` (message), plus `node_id`, `class_type`, `exception_type` and a
    trimmed `traceback` when a specific node failed."""
    if isinstance(exc, NodeExecutionError):
        return exc.to_dict()
    return {"error": str(exc), "node_id": None, "class_type": None, "exception_type": type(exc).__name__}
