import inspect
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from ..nodes import NODE_REGISTRY

logger = logging.getLogger(__name__)


class GraphError(Exception):
    """Raised for structural problems in a prompt graph: an unknown node id,
    an unknown class_type, or a cycle between node inputs."""


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)


def execute_prompt(
    prompt: Dict[str, Dict[str, Any]],
    on_node_start: Optional[Callable[[str, str], None]] = None,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """Execute a graph given in ComfyUI-style "prompt" form:

        {node_id: {"class_type": str, "inputs": {name: value | [node_id, output_index]}}}

    Returns a dict of {node_id: ui_dict} collected from OUTPUT_NODE results
    (e.g. {"images": [filename, ...]}).

    `on_progress`, if given, is called as `on_progress(node_id, step, total)`
    for in-progress work within a single node (e.g. KSampler's diffusion
    steps). It's only wired into a node's FUNCTION call when that function
    declares an `on_progress` parameter (checked via introspection here rather
    than by changing every node's call signature) - most nodes finish in one
    shot and have no sub-steps to report.
    """

    cache: Dict[str, Tuple[Any, ...]] = {}
    ui_outputs: Dict[str, Any] = {}
    in_progress: set = set()

    def resolve(node_id: str) -> Tuple[Any, ...]:
        if node_id in cache:
            return cache[node_id]
        if node_id in in_progress:
            raise GraphError(f"Cycle detected in node graph at node '{node_id}'")

        try:
            spec = prompt[node_id]
        except KeyError:
            raise GraphError(f"Node '{node_id}' is referenced but not present in the prompt") from None

        class_type = spec.get("class_type")
        try:
            node_cls = NODE_REGISTRY[class_type]
        except KeyError:
            raise GraphError(f"Unknown node class '{class_type}' (node '{node_id}')") from None
        node = node_cls()

        in_progress.add(node_id)
        try:
            kwargs = {}
            for name, value in spec.get("inputs", {}).items():
                if _is_link(value):
                    src_id, out_idx = value
                    src_outputs = resolve(str(src_id))
                    kwargs[name] = src_outputs[out_idx]
                else:
                    kwargs[name] = value

            if on_node_start:
                on_node_start(node_id, class_type)

            logger.debug("executing node %s (%s)", node_id, class_type)
            func = getattr(node, node_cls.FUNCTION)
            if on_progress is not None and "on_progress" in inspect.signature(func).parameters:
                kwargs["on_progress"] = lambda step, total, _id=node_id: on_progress(_id, step, total)
            try:
                result = func(**kwargs)
            except Exception:
                logger.exception("node %s (%s) raised during execution", node_id, class_type)
                raise
        finally:
            in_progress.discard(node_id)

        if isinstance(result, dict):
            ui_outputs[node_id] = result
            outputs: Tuple[Any, ...] = tuple(result.get("result", ()))
        else:
            outputs = result if isinstance(result, tuple) else (result,)

        cache[node_id] = outputs
        return outputs

    for node_id in list(prompt.keys()):
        resolve(node_id)

    return ui_outputs
