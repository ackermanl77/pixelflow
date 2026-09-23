import asyncio
import logging
import uuid
from typing import Any, Dict, List

from fastapi import WebSocket

from .graph import execute_prompt

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


class ExecutionQueue:
    """A single-worker async job queue, mirroring ComfyUI's queue/history model:
    prompts are submitted, run one at a time in a background worker, and
    progress/results are broadcast over the websocket as they happen.
    """

    def __init__(self, manager: ConnectionManager) -> None:
        self.manager = manager
        self.pending: "asyncio.Queue[tuple[str, dict]]" = asyncio.Queue()
        self.history: Dict[str, Any] = {}
        self._worker_task: Any = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def submit(self, prompt: Dict[str, Any]) -> str:
        prompt_id = uuid.uuid4().hex[:12]
        self.history[prompt_id] = {"status": "queued"}
        await self.pending.put((prompt_id, prompt))
        await self.manager.broadcast({"type": "queued", "prompt_id": prompt_id})
        return prompt_id

    async def _worker(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            prompt_id, prompt = await self.pending.get()
            self.history[prompt_id] = {"status": "running"}
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

            try:
                result = await loop.run_in_executor(None, lambda: execute_prompt(prompt, on_node_start))
                self.history[prompt_id] = {"status": "completed", "outputs": result}
                await self.manager.broadcast(
                    {"type": "execution_success", "prompt_id": prompt_id, "outputs": result}
                )
            except Exception as exc:  # noqa: BLE001 - surface any node error to the client
                logger.exception("prompt %s failed during execution", prompt_id)
                self.history[prompt_id] = {"status": "error", "error": str(exc)}
                await self.manager.broadcast(
                    {"type": "execution_error", "prompt_id": prompt_id, "error": str(exc)}
                )
            finally:
                self.pending.task_done()
