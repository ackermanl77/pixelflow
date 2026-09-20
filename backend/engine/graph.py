from typing import Any, Callable, Dict, Optional, Tuple

from ..nodes import NODE_REGISTRY


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)


def execute_prompt(
    prompt: Dict[str, Dict[str, Any]],
    on_node_start: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Execute a graph given in ComfyUI-style "prompt" form:

        {node_id: {"class_type": str, "inputs": {name: value | [node_id, output_index]}}}

    Returns a dict of {node_id: ui_dict} collected from OUTPUT_NODE results
    (e.g. {"images": [filename, ...]}).
    """

    cache: Dict[str, Tuple[Any, ...]] = {}
    ui_outputs: Dict[str, Any] = {}

    def resolve(node_id: str) -> Tuple[Any, ...]:
        if node_id in cache:
            return cache[node_id]

        spec = prompt[node_id]
        node_cls = NODE_REGISTRY[spec["class_type"]]
        node = node_cls()

        kwargs = {}
        for name, value in spec.get("inputs", {}).items():
            if _is_link(value):
                src_id, out_idx = value
                src_outputs = resolve(str(src_id))
                kwargs[name] = src_outputs[out_idx]
            else:
                kwargs[name] = value

        if on_node_start:
            on_node_start(node_id, spec["class_type"])

        func = getattr(node, node_cls.FUNCTION)
        result = func(**kwargs)

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
