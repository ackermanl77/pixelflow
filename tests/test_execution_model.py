"""Output-driven execution (only what the outputs need), the output cache
between runs, and structured node errors (report items B1, B2, B3)."""

import asyncio

import pytest

from backend.engine.graph import (
    NodeExecutionError,
    OutputCache,
    PromptValidationError,
    execute_prompt,
    validate_prompt,
)
from backend.engine.queue import ConnectionManager, ExecutionQueue
from backend.nodes import NODE_REGISTRY
from backend.nodes.base import NodeBase, register_node
from backend.nodes.generation import KSamplerImage, KSamplerInpaint, KSamplerStub


class _RecordingManager(ConnectionManager):
    def __init__(self) -> None:
        super().__init__()
        self.messages = []

    async def broadcast(self, message):
        self.messages.append(message)


@pytest.fixture
def counting_nodes():
    """_Counter produces an IMAGE-typed value and counts its executions;
    _Boom always raises."""
    calls = []
    for name in ("_Counter", "_Boom"):
        NODE_REGISTRY.pop(name, None)

    @register_node()
    class _Counter(NodeBase):
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("INT", {"default": 0})}}

        def run(self, value):
            calls.append(value)
            from PIL import Image

            return (Image.new("RGB", (8, 8), (value % 256, 0, 0)),)

    @register_node()
    class _Boom(NodeBase):
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        def run(self):
            raise ValueError("kaboom")

    yield calls
    for name in ("_Counter", "_Boom"):
        NODE_REGISTRY.pop(name, None)


def _counter_graph(value=1, radius=1.0):
    return {
        "src": {"class_type": "_Counter", "inputs": {"value": value}},
        "blur": {"class_type": "ImageBlur", "inputs": {"image": ["src", 0], "radius": radius}},
        "out": {"class_type": "PreviewImage", "inputs": {"image": ["blur", 0]}},
    }


# ---- B1: output-driven validation and execution ----


def test_validation_drops_nodes_no_output_needs(counting_nodes):
    graph = _counter_graph()
    graph["stray"] = {"class_type": "ImageBlur", "inputs": {"radius": 1.0}}  # unconnected 'image'
    graph["unused_ckpt"] = {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": "x"}}
    normalized = validate_prompt(graph)
    assert set(normalized) == {"src", "blur", "out"}


def test_validation_requires_an_output_node():
    with pytest.raises(PromptValidationError, match="no output nodes") as exc_info:
        validate_prompt({"a": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "color": "#000"}}})
    assert exc_info.value.node_id is None


def test_execution_skips_nodes_the_outputs_do_not_need(counting_nodes, output_dir):
    graph = _counter_graph(value=7)
    graph["unused"] = {"class_type": "_Counter", "inputs": {"value": 99}}
    execute_prompt(graph)
    assert counting_nodes == [7]


# ---- B3: structured errors ----


def test_validation_error_names_class_and_input(counting_nodes):
    graph = _counter_graph()
    graph["blur"]["inputs"]["radius"] = 1e9
    with pytest.raises(PromptValidationError) as exc_info:
        validate_prompt(graph)
    assert exc_info.value.to_dict() == {
        "message": str(exc_info.value),
        "node_id": "blur",
        "class_type": "ImageBlur",
        "input": "radius",
    }


def test_node_exception_is_wrapped_with_node_details(counting_nodes):
    graph = {"b": {"class_type": "_Boom", "inputs": {}}, "out": {"class_type": "PreviewImage", "inputs": {"image": ["b", 0]}}}
    with pytest.raises(NodeExecutionError) as exc_info:
        execute_prompt(graph)
    err = exc_info.value
    assert (err.node_id, err.class_type, err.exception_type) == ("b", "_Boom", "ValueError")
    assert isinstance(err.__cause__, ValueError)
    assert "kaboom" in str(err) and "kaboom" in err.traceback


def test_queue_reports_the_failing_node_in_history_and_broadcast(counting_nodes):
    manager = _RecordingManager()
    queue = ExecutionQueue(manager)
    graph = {"b": {"class_type": "_Boom", "inputs": {}}, "out": {"class_type": "PreviewImage", "inputs": {"image": ["b", 0]}}}

    async def _run():
        queue.start()
        prompt_id = await queue.submit(graph)
        for _ in range(200):
            if queue.history[prompt_id]["status"] not in ("queued", "running"):
                break
            await asyncio.sleep(0.01)
        return prompt_id

    prompt_id = asyncio.run(_run())
    entry = queue.history[prompt_id]
    assert entry["status"] == "error"
    assert entry["node_id"] == "b" and entry["class_type"] == "_Boom"
    assert entry["exception_type"] == "ValueError" and "kaboom" in entry["traceback"]
    assert entry["started_at"] <= entry["finished_at"]
    (message,) = [m for m in manager.messages if m["type"] == "execution_error"]
    assert message["node_id"] == "b" and "kaboom" in message["error"]


# ---- B2: output cache between runs ----


def test_unchanged_upstream_is_reused_and_changed_downstream_reruns(counting_nodes, temp_dir):
    cache = OutputCache()
    cached = []
    execute_prompt(_counter_graph(radius=1.0), output_cache=cache)
    execute_prompt(
        _counter_graph(radius=3.0), output_cache=cache, on_node_cached=lambda nid, cls: cached.append(nid)
    )
    assert counting_nodes == [1], "the unchanged source node must not run again"
    assert cached == ["src"]
    # Output nodes always run, so every run still produces a preview.
    assert len(list(temp_dir.iterdir())) == 2


def test_changed_widget_invalidates_the_node_and_its_dependents(counting_nodes, output_dir):
    cache = OutputCache()
    execute_prompt(_counter_graph(value=1), output_cache=cache)
    execute_prompt(_counter_graph(value=2), output_cache=cache)
    assert counting_nodes == [1, 2]


def test_whole_upstream_chain_is_served_from_cache(counting_nodes, output_dir):
    cache = OutputCache()
    cached = []
    execute_prompt(_counter_graph(), output_cache=cache)
    execute_prompt(_counter_graph(), output_cache=cache, on_node_cached=lambda nid, cls: cached.append(nid))
    # blur is a hit, so src never even needs resolving.
    assert cached == ["blur"]
    assert counting_nodes == [1]


def test_without_a_cache_everything_runs(counting_nodes, output_dir):
    execute_prompt(_counter_graph())
    execute_prompt(_counter_graph())
    assert counting_nodes == [1, 1]


def test_cache_keeps_only_the_last_run(counting_nodes, output_dir):
    cache = OutputCache()
    execute_prompt(_counter_graph(), output_cache=cache)
    assert "src" in cache and "blur" in cache
    other = {
        "e": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "color": "#000"}},
        "out": {"class_type": "PreviewImage", "inputs": {"image": ["e", 0]}},
    }
    execute_prompt(other, output_cache=cache)
    assert "src" not in cache and "e" in cache


def test_samplers_rerun_with_a_random_seed_but_not_a_fixed_one():
    for cls in (KSamplerStub, KSamplerImage, KSamplerInpaint):
        random_token = cls.IS_CHANGED(seed=-1)
        assert random_token != random_token  # NaN: never equal, always re-run
        assert cls.IS_CHANGED(seed=5) == cls.IS_CHANGED(seed=5)


def test_nan_is_changed_forces_a_rerun(counting_nodes, output_dir, monkeypatch):
    monkeypatch.setattr(NODE_REGISTRY["_Counter"], "IS_CHANGED", classmethod(lambda cls, **_: float("nan")), raising=False)
    cache = OutputCache()
    execute_prompt(_counter_graph(), output_cache=cache)
    execute_prompt(_counter_graph(), output_cache=cache)
    assert counting_nodes == [1, 1]


def test_failed_run_keeps_earlier_cache_entries(counting_nodes, output_dir):
    cache = OutputCache()
    execute_prompt(_counter_graph(), output_cache=cache)
    broken = {"b": {"class_type": "_Boom", "inputs": {}}, "out": {"class_type": "PreviewImage", "inputs": {"image": ["b", 0]}}}
    with pytest.raises(NodeExecutionError):
        execute_prompt(broken, output_cache=cache)
    assert "blur" in cache


def test_queue_broadcasts_execution_cached(counting_nodes, output_dir):
    manager = _RecordingManager()
    queue = ExecutionQueue(manager)

    async def _run():
        queue.start()
        ids = [await queue.submit(_counter_graph()), await queue.submit(_counter_graph())]
        for _ in range(300):
            if all(queue.history[i]["status"] not in ("queued", "running") for i in ids):
                break
            await asyncio.sleep(0.01)
        return ids

    first, second = asyncio.run(_run())
    assert queue.history[second]["status"] == "completed"
    cached = [m for m in manager.messages if m["type"] == "execution_cached"]
    assert cached == [
        {"type": "execution_cached", "prompt_id": second, "nodes": ["blur"], "node_index": 1, "node_count": 3}
    ]
    executing = [m for m in manager.messages if m["type"] == "executing" and m["prompt_id"] == second]
    # "blur" came from the cache, so "src" (only its input) never had to run.
    assert [(m["node_index"], m["node_count"]) for m in executing] == [(2, 3)]
    assert counting_nodes == [1]
