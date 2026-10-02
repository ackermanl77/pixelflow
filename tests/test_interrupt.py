import asyncio
import threading

import pytest

from backend.engine.graph import ExecutionInterrupted, execute_prompt
from backend.engine.queue import ConnectionManager, ExecutionQueue
from backend.nodes import NODE_REGISTRY
from backend.nodes.base import NodeBase, register_node

from _hub import skip_if_hub_unreachable

TINY_SD_CHECKPOINT = "hf-internal-testing/tiny-stable-diffusion-torch"


class _RecordingManager(ConnectionManager):
    def __init__(self) -> None:
        super().__init__()
        self.messages = []

    async def broadcast(self, message):
        self.messages.append(message)


@pytest.fixture
def slow_node():
    """A fake many-step node that reports progress like KSampler does, with
    no model behind it. `started` is set once its first step has run."""
    NODE_REGISTRY.pop("_SlowSteps", None)
    started = threading.Event()
    steps_run = []

    @register_node()
    class _SlowSteps(NodeBase):
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"steps": ("INT", {"default": 50, "min": 1, "max": 1000})}}

        def run(self, steps, on_progress=None):
            for i in range(steps):
                steps_run.append(i)
                started.set()
                threading.Event().wait(0.01)
                if on_progress is not None:
                    on_progress(i + 1, steps)
            return (None,)

    yield {"started": started, "steps_run": steps_run}
    NODE_REGISTRY.pop("_SlowSteps", None)


def test_should_interrupt_aborts_mid_node_via_the_progress_callback(slow_node):
    calls = []

    def should_interrupt():
        calls.append(1)
        return len(slow_node["steps_run"]) >= 3

    prompt = {"n": {"class_type": "_SlowSteps", "inputs": {"steps": 50}}}
    with pytest.raises(ExecutionInterrupted):
        execute_prompt(prompt, should_interrupt=should_interrupt)
    assert len(slow_node["steps_run"]) == 3, "the node must stop at the next step, not run to completion"


def test_should_interrupt_is_checked_before_each_node(output_dir):
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "color": "#000000"}},
        "2": {"class_type": "SaveImage", "inputs": {"image": ["1", 0], "filename_prefix": "never"}},
    }
    with pytest.raises(ExecutionInterrupted):
        execute_prompt(prompt, should_interrupt=lambda: True)
    assert list(output_dir.iterdir()) == []


def test_queue_interrupt_stops_the_running_prompt_and_runs_the_next(slow_node):
    manager = _RecordingManager()
    queue = ExecutionQueue(manager)

    async def _run():
        queue.start()
        first = await queue.submit({"n": {"class_type": "_SlowSteps", "inputs": {"steps": 1000}}})
        second = await queue.submit({"n": {"class_type": "_SlowSteps", "inputs": {"steps": 2}}})
        for _ in range(200):
            if slow_node["started"].is_set() and queue.running_prompt_id == first:
                break
            await asyncio.sleep(0.01)
        assert queue.interrupt() == first
        for _ in range(500):
            if queue.history[second]["status"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.01)
        return first, second

    first, second = asyncio.run(_run())
    assert queue.history[first]["status"] == "interrupted"
    assert len(slow_node["steps_run"]) < 1000
    # The flag is cleared per prompt, so the next one isn't interrupted too.
    assert queue.history[second]["status"] == "completed"
    assert {"type": "execution_interrupted", "prompt_id": first} in manager.messages
    assert queue.interrupt() is None


def test_interrupt_reaches_a_real_diffusers_step_callback():
    """KSampler's _report_progress (diffusers' callback_on_step_end) is where
    the interrupt check fires during sampling."""
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
                "seed": 1,
                "steps": 10,
                "cfg": 7.0,
            },
        },
    }
    progress = []
    try:
        execute_prompt({"ckpt": prompt["ckpt"]})  # warm the cache / skip if offline
    except Exception as exc:
        skip_if_hub_unreachable(exc)
        raise

    with pytest.raises(ExecutionInterrupted):
        execute_prompt(
            prompt,
            on_progress=lambda node_id, step, total: progress.append(step),
            should_interrupt=lambda: len(progress) >= 2,
        )
    assert progress == [1, 2]
