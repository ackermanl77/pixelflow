import asyncio
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

from .graph import ExecutionInterrupted, execute_prompt

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


class ExecutionQueue:
    """A single-worker async job queue, mirroring ComfyUI's queue/history model:
    prompts are submitted, run one at a time in a background worker, and
    progress/results are broadcast over the websocket as they happen.
    """

    def __init__(self, manager: ConnectionManager, max_history: int = DEFAULT_MAX_HISTORY) -> None:
        self.manager = manager
        self.pending: "asyncio.Queue[tuple[str, dict, dict]]" = asyncio.Queue()
        self.history: Dict[str, Any] = {}
        self.max_history = max_history
        self._cancelled: set = set()
        self._worker_task: Any = None
        self.running_prompt_id: Optional[str] = None
        # Set by interrupt(), polled by execute_prompt() between nodes and on
        # every KSampler diffusion step (via its on_progress callback) from
        # the worker thread - hence a threading.Event, not an asyncio one.
        self._interrupt_event = threading.Event()

    def interrupt(self) -> Optional[str]:
        """Ask the currently running prompt to stop at its next checkpoint
        (the next node, or the next diffusion step). Returns the running
        prompt's id, or None if nothing is running."""
        prompt_id = self.running_prompt_id
        if prompt_id is not None:
            self._interrupt_event.set()
        return prompt_id

    def cancel(self, prompt_id: str) -> bool:
        """Cancel a prompt that is still queued. Returns False for an unknown
        id or one that's already running/finished - use interrupt() to stop
        the running one."""
        entry = self.history.get(prompt_id)
        if entry is None or entry.get("status") != "queued":
            return False
        self._cancelled.add(prompt_id)
        self._record_history(prompt_id, {"status": "cancelled"})
        return True

    def delete_history(self, prompt_id: str) -> bool:
        """Remove a finished (completed/error/cancelled) history entry. Returns
        False for an unknown id or one that's still queued/running."""
        entry = self.history.get(prompt_id)
        if entry is None or entry.get("status") in ("queued", "running"):
            return False
        del self.history[prompt_id]
        return True

    def _record_history(self, prompt_id: str, entry: Dict[str, Any]) -> None:
        self.history[prompt_id] = entry
        # dict preserves insertion order, and updating an existing key keeps
        # its original position, so the front is always the oldest prompt.
        while len(self.history) > self.max_history:
            oldest_id = next(iter(self.history))
            del self.history[oldest_id]

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def submit(self, prompt: Dict[str, Any], extra_pnginfo: Optional[Dict[str, Any]] = None) -> str:
        """Queue a prompt. `extra_pnginfo` (e.g. {"workflow": <litegraph
        graph>}) is embedded into saved PNGs alongside the prompt itself."""
        prompt_id = uuid.uuid4().hex[:12]
        self._record_history(prompt_id, {"status": "queued"})
        await self.pending.put((prompt_id, prompt, extra_pnginfo or {}))
        await self.manager.broadcast({"type": "queued", "prompt_id": prompt_id})
        return prompt_id

    async def _worker(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            prompt_id, prompt, extra_pnginfo = await self.pending.get()
            if prompt_id in self._cancelled:
                self._cancelled.discard(prompt_id)
                self.pending.task_done()
                continue
            self._interrupt_event.clear()
            self.running_prompt_id = prompt_id
            self._record_history(prompt_id, {"status": "running"})
            await self.manager.broadcast({"type": "execution_start", "prompt_id": prompt_id})

            def on_node_start(node_id: str, class_type: str, _pid: str = prompt_id) -> None:
                asyncio.run_coroutine_threadsafe(
                    self.manager.broadcast(
                        {
                            "type": "executing",
                            "prompt_id": _pid,
                            "node_id": node_id,
                            "class_type": class_type,
                        }
                    ),
                    loop,
                )

            def on_progress(node_id: str, step: int, total: int, _pid: str = prompt_id) -> None:
                asyncio.run_coroutine_threadsafe(
                    self.manager.broadcast(
                        {
                            "type": "progress",
                            "prompt_id": _pid,
                            "node_id": node_id,
                            "step": step,
                            "total": total,
                        }
                    ),
                    loop,
                )

            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: execute_prompt(
                        prompt,
                        on_node_start,
                        on_progress,
                        should_interrupt=self._interrupt_event.is_set,
                        extra_pnginfo=extra_pnginfo,
                    ),
                )
                self._record_history(prompt_id, {"status": "completed", "outputs": result})
                await self.manager.broadcast(
                    {"type": "execution_success", "prompt_id": prompt_id, "outputs": result}
                )
            except ExecutionInterrupted:
                logger.info("prompt %s was interrupted", prompt_id)
                self._record_history(prompt_id, {"status": "interrupted"})
                await self.manager.broadcast({"type": "execution_interrupted", "prompt_id": prompt_id})
            except Exception as exc:  # noqa: BLE001 - surface any node error to the client
                logger.exception("prompt %s failed during execution", prompt_id)
                self._record_history(prompt_id, {"status": "error", "error": str(exc)})
                await self.manager.broadcast(
                    {"type": "execution_error", "prompt_id": prompt_id, "error": str(exc)}
                )
            finally:
                self.running_prompt_id = None
                self._interrupt_event.clear()
                self.pending.task_done()
