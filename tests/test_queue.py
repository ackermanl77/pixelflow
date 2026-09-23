import asyncio

import pytest
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import ConnectionError as RequestsConnectionError

from backend.engine.graph import execute_prompt
from backend.engine.queue import ConnectionManager, ExecutionQueue

# Same tiny, randomly-initialized-weights checkpoint test_graph.py uses to
# exercise the real diffusers pipeline without downloading a production model.
TINY_SD_CHECKPOINT = "hf-internal-testing/tiny-stable-diffusion-torch"


class _RecordingManager(ConnectionManager):
    """A ConnectionManager that records broadcasts instead of sending them
    over a real websocket, so queue behavior can be asserted on directly."""

    def __init__(self) -> None:
        super().__init__()
        self.messages = []

    async def broadcast(self, message):
        self.messages.append(message)


def test_record_history_caps_at_max_history():
    queue = ExecutionQueue(ConnectionManager(), max_history=3)
    for i in range(5):
        queue._record_history(f"id{i}", {"status": "completed"})
    assert list(queue.history.keys()) == ["id2", "id3", "id4"]


def test_record_history_updating_existing_key_keeps_its_slot():
    queue = ExecutionQueue(ConnectionManager(), max_history=3)
    queue._record_history("a", {"status": "queued"})
    queue._record_history("b", {"status": "queued"})
    queue._record_history("c", {"status": "queued"})
    queue._record_history("a", {"status": "completed"})
    assert list(queue.history.keys()) == ["a", "b", "c"]
    assert queue.history["a"]["status"] == "completed"


def test_cancel_marks_a_queued_prompt_cancelled():
    queue = ExecutionQueue(ConnectionManager())
    queue._record_history("a", {"status": "queued"})
    assert queue.cancel("a") is True
    assert queue.history["a"]["status"] == "cancelled"
    assert "a" in queue._cancelled


def test_cancel_refuses_an_unknown_id():
    queue = ExecutionQueue(ConnectionManager())
    assert queue.cancel("nope") is False


def test_cancel_refuses_a_running_or_finished_prompt():
    queue = ExecutionQueue(ConnectionManager())
    queue._record_history("running", {"status": "running"})
    queue._record_history("done", {"status": "completed"})
    assert queue.cancel("running") is False
    assert queue.cancel("done") is False


def test_delete_history_removes_a_finished_entry():
    queue = ExecutionQueue(ConnectionManager())
    queue._record_history("done", {"status": "completed"})
    assert queue.delete_history("done") is True
    assert "done" not in queue.history


def test_delete_history_refuses_queued_or_running_or_unknown():
    queue = ExecutionQueue(ConnectionManager())
    queue._record_history("queued", {"status": "queued"})
    queue._record_history("running", {"status": "running"})
    assert queue.delete_history("queued") is False
    assert queue.delete_history("running") is False
    assert queue.delete_history("nope") is False


def test_execution_queue_broadcasts_progress_during_a_real_pipeline_run():
    """End-to-end: submitting a real (tiny) diffusion prompt through
    ExecutionQueue must broadcast 'progress' messages (node_id/step/total)
    over its ConnectionManager as KSampler samples, not just the existing
    queued/executing/execution_success messages."""

    # Warm the in-process pipeline cache (see backend/nodes/io_nodes.py) with
    # a direct execute_prompt call first, using the same
    # HfHubHTTPError/RequestsConnectionError skip-if-unreachable pattern
    # test_graph.py uses. This keeps the assertion below about a genuine
    # ExecutionQueue/broadcast bug rather than network flakiness, since any
    # exception raised once the queue is driving it below is swallowed into
    # a generic "error" history entry the test could otherwise not tell apart
    # from a real regression.
    warmup_prompt = {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY_SD_CHECKPOINT}},
    }
    try:
        execute_prompt(warmup_prompt)
    except (HfHubHTTPError, RequestsConnectionError) as exc:
        pytest.skip(f"Hugging Face Hub unreachable: {exc}")

    prompt = {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": TINY_SD_CHECKPOINT}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red circle"}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "color": "#000000"}},
        "sampler": {
            "class_type": "KSamplerStub",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["clip", 0],
                "latent_image": ["latent", 0],
                "seed": 5,
                "steps": 3,
                "cfg": 7.0,
            },
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "queue_progress"}},
    }

    manager = _RecordingManager()
    queue = ExecutionQueue(manager)

    async def _run():
        queue.start()
        prompt_id = await queue.submit(prompt)
        for _ in range(300):  # up to ~30s of polling
            if queue.history[prompt_id]["status"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.1)
        return prompt_id

    prompt_id = asyncio.run(_run())

    entry = queue.history[prompt_id]
    assert entry["status"] == "completed", entry.get("error")

    progress_messages = [m for m in manager.messages if m["type"] == "progress"]
    assert progress_messages, "expected at least one 'progress' broadcast"
    assert all(m["prompt_id"] == prompt_id for m in progress_messages)
    assert all(m["node_id"] == "sampler" for m in progress_messages)
    assert all(m["total"] == 3 for m in progress_messages)
    assert all(1 <= m["step"] <= 3 for m in progress_messages)

    # The existing message types must still show up around it.
    message_types = [m["type"] for m in manager.messages]
    assert "queued" in message_types
    assert "execution_start" in message_types
    assert "executing" in message_types
    assert "execution_success" in message_types
