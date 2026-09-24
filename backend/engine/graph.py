import copy
import inspect
import logging
import math
from typing import Any, Callable, Dict, Optional, Tuple

from ..nodes import NODE_REGISTRY

logger = logging.getLogger(__name__)


class GraphError(Exception):
    """Raised for structural problems in a prompt graph: an unknown node id,
    an unknown class_type, or a cycle between node inputs."""


class PromptValidationError(GraphError):
    """Raised by validate_prompt() for a graph that can't run as submitted.
    `node_id` names the offending node (None for graph-wide problems)."""

    def __init__(self, node_id: Optional[str], message: str) -> None:
        self.node_id = node_id
        super().__init__(f"Node '{node_id}': {message}" if node_id is not None else message)


class ExecutionInterrupted(Exception):
    """Raised inside execute_prompt() when `should_interrupt()` turns true -
    between nodes, or from inside a node's per-step `on_progress` callback
    (e.g. between KSampler's diffusion steps)."""


# Widget (non-socket) value types. Anything else declared as a plain string
# type in INPUT_TYPES (IMAGE, MODEL, CONDITIONING, ...) is a socket that must
# be fed by a link.
PRIMITIVE_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}

# Hidden input kinds execute_prompt() knows how to inject (ComfyUI's names).
HIDDEN_PROMPT = "PROMPT"
HIDDEN_EXTRA_PNGINFO = "EXTRA_PNGINFO"
HIDDEN_UNIQUE_ID = "UNIQUE_ID"


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)


def _declared_inputs(node_cls) -> Dict[str, Any]:
    input_types = node_cls.INPUT_TYPES()
    declared = {}
    declared.update(input_types.get("required", {}))
    declared.update(input_types.get("optional", {}))
    return declared


def _validate_value(node_id: str, name: str, spec: Tuple, value: Any) -> Any:
    """Check a literal (non-link) widget value against its INPUT_TYPES spec
    and return it normalized (e.g. an integral float coerced to int)."""
    input_type = spec[0]
    config = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}

    if isinstance(input_type, (list, tuple)):
        if value not in input_type:
            raise PromptValidationError(
                node_id, f"input '{name}' must be one of {list(input_type)}, got {value!r}"
            )
        return value

    if input_type in ("INT", "FLOAT"):
        # bool is an int subclass, but True/False is never a meaningful number here.
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PromptValidationError(node_id, f"input '{name}' must be a number, got {value!r}")
        if input_type == "INT":
            if value != int(value):
                raise PromptValidationError(node_id, f"input '{name}' must be an integer, got {value!r}")
            value = int(value)
        else:
            value = float(value)
        minimum, maximum = config.get("min"), config.get("max")
        if minimum is not None and value < minimum:
            raise PromptValidationError(node_id, f"input '{name}' is {value}, below the minimum of {minimum}")
        if maximum is not None and value > maximum:
            raise PromptValidationError(node_id, f"input '{name}' is {value}, above the maximum of {maximum}")
        return value

    if input_type == "STRING":
        if not isinstance(value, str):
            raise PromptValidationError(node_id, f"input '{name}' must be a string, got {value!r}")
        return value

    if input_type == "BOOLEAN":
        if not isinstance(value, bool):
            raise PromptValidationError(node_id, f"input '{name}' must be true or false, got {value!r}")
        return value

    raise PromptValidationError(node_id, f"input '{name}' ({input_type}) must be connected to another node's output")


def validate_prompt(prompt: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Check a whole prompt graph against each node's INPUT_TYPES/RETURN_TYPES
    before it's queued, so a bad graph fails fast with a message naming the
    node instead of partway through execution. Checks: known class_type,
    required inputs present, no undeclared inputs, links pointing at an
    existing node and an in-range output whose type matches the input's,
    numbers within INPUT_TYPES min/max, combo values in their list, and no
    cycles. Returns a normalized deep copy (e.g. INT values coerced to int).
    Raises PromptValidationError."""
    if not prompt:
        raise PromptValidationError(None, "The prompt contains no nodes")

    normalized: Dict[str, Dict[str, Any]] = {}
    for node_id, spec in prompt.items():
        node_id = str(node_id)
        class_type = spec.get("class_type")
        node_cls = NODE_REGISTRY.get(class_type)
        if node_cls is None:
            raise PromptValidationError(node_id, f"unknown node class '{class_type}'")

        input_types = node_cls.INPUT_TYPES()
        required = input_types.get("required", {})
        declared = _declared_inputs(node_cls)
        inputs = spec.get("inputs") or {}

        for name in required:
            if name not in inputs:
                raise PromptValidationError(node_id, f"required input '{name}' is missing (not connected or not set)")

        normalized_inputs: Dict[str, Any] = {}
        for name, value in inputs.items():
            if name not in declared:
                raise PromptValidationError(node_id, f"unknown input '{name}' for {class_type}")
            input_spec = declared[name]
            if _is_link(value):
                src_id, out_idx = str(value[0]), value[1]
                src_spec = prompt.get(src_id)
                if src_spec is None:
                    raise PromptValidationError(node_id, f"input '{name}' is linked to missing node '{src_id}'")
                src_cls = NODE_REGISTRY.get(src_spec.get("class_type"))
                if src_cls is None:
                    raise PromptValidationError(src_id, f"unknown node class '{src_spec.get('class_type')}'")
                return_types = tuple(src_cls.RETURN_TYPES)
                if not 0 <= out_idx < len(return_types):
                    raise PromptValidationError(
                        node_id,
                        f"input '{name}' is linked to output {out_idx} of node '{src_id}', which has "
                        f"{len(return_types)} output(s)",
                    )
                expected = input_spec[0]
                actual = return_types[out_idx]
                if isinstance(expected, (list, tuple)) or expected != actual:
                    raise PromptValidationError(
                        node_id,
                        f"input '{name}' expects {expected if isinstance(expected, str) else 'a choice'} "
                        f"but is linked to a {actual} output of node '{src_id}'",
                    )
                normalized_inputs[name] = [src_id, out_idx]
            else:
                normalized_inputs[name] = _validate_value(node_id, name, input_spec, value)

        normalized[node_id] = {"class_type": class_type, "inputs": normalized_inputs}

    # Cycle check (iterative DFS; the engine would also catch this, but only
    # once execution has already started).
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node_id: WHITE for node_id in normalized}
    for root in normalized:
        if color[root] != WHITE:
            continue
        stack = [(root, iter(normalized[root]["inputs"].values()))]
        color[root] = GRAY
        while stack:
            node_id, it = stack[-1]
            advanced = False
            for value in it:
                if _is_link(value):
                    dep = value[0]
                    if color[dep] == GRAY:
                        raise PromptValidationError(dep, "cycle detected in node graph")
                    if color[dep] == WHITE:
                        color[dep] = GRAY
                        stack.append((dep, iter(normalized[dep]["inputs"].values())))
                        advanced = True
                        break
            if not advanced:
                color[node_id] = BLACK
                stack.pop()

    return normalized


def execute_prompt(
    prompt: Dict[str, Dict[str, Any]],
    on_node_start: Optional[Callable[[str, str], None]] = None,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
    should_interrupt: Optional[Callable[[], bool]] = None,
    extra_pnginfo: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a graph given in ComfyUI-style "prompt" form:

        {node_id: {"class_type": str, "inputs": {name: value | [node_id, output_index]}}}

    Returns a dict of {node_id: ui_dict} collected from nodes that return a
    dict (e.g. SaveImage's {"images": [filename, ...]}, KSampler's
    {"seed": N}); a dict's "result" key, if any, is the node's output tuple
    and is not part of the ui_dict.

    `on_progress`, if given, is called as `on_progress(node_id, step, total)`
    for in-progress work within a single node (e.g. KSampler's diffusion
    steps). It's only wired into a node's FUNCTION call when that function
    declares an `on_progress` parameter (checked via introspection here rather
    than by changing every node's call signature) - most nodes finish in one
    shot and have no sub-steps to report.

    `should_interrupt`, if given, is polled before every node and on every
    `on_progress` step; once it returns True, ExecutionInterrupted is raised
    (from inside the node for a per-step check, e.g. out of diffusers'
    step callback), abandoning the rest of the prompt.

    Nodes can declare ComfyUI-style hidden inputs in INPUT_TYPES()["hidden"]
    ({name: "PROMPT" | "EXTRA_PNGINFO" | "UNIQUE_ID"}); they're injected here
    rather than coming from the graph. PROMPT is the prompt as executed so
    far, with any node's reported {"seed": N} written back into its `seed`
    input so the embedded graph reproduces the result.
    """

    cache: Dict[str, Tuple[Any, ...]] = {}
    ui_outputs: Dict[str, Any] = {}
    in_progress: set = set()
    resolved_prompt = copy.deepcopy(prompt)

    def check_interrupt() -> None:
        if should_interrupt is not None and should_interrupt():
            raise ExecutionInterrupted("Prompt execution was interrupted")

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

            hidden_values = {
                HIDDEN_PROMPT: resolved_prompt,
                HIDDEN_EXTRA_PNGINFO: extra_pnginfo,
                HIDDEN_UNIQUE_ID: node_id,
            }
            for name, kind in node_cls.INPUT_TYPES().get("hidden", {}).items():
                kwargs[name] = hidden_values.get(kind)

            check_interrupt()
            if on_node_start:
                on_node_start(node_id, class_type)

            logger.debug("executing node %s (%s)", node_id, class_type)
            func = getattr(node, node_cls.FUNCTION)
            if (on_progress is not None or should_interrupt is not None) and (
                "on_progress" in inspect.signature(func).parameters
            ):

                def _node_progress(step, total, _id=node_id):
                    check_interrupt()
                    if on_progress is not None:
                        on_progress(_id, step, total)

                kwargs["on_progress"] = _node_progress
            try:
                result = func(**kwargs)
            except ExecutionInterrupted:
                raise
            except Exception:
                logger.exception("node %s (%s) raised during execution", node_id, class_type)
                raise
        finally:
            in_progress.discard(node_id)

        if isinstance(result, dict):
            ui = {key: value for key, value in result.items() if key != "result"}
            if ui:
                ui_outputs[node_id] = ui
            if "seed" in ui and "seed" in resolved_prompt.get(node_id, {}).get("inputs", {}):
                resolved_prompt[node_id]["inputs"]["seed"] = ui["seed"]
            outputs: Tuple[Any, ...] = tuple(result.get("result", ()))
        else:
            outputs = result if isinstance(result, tuple) else (result,)

        cache[node_id] = outputs
        return outputs

    for node_id in list(prompt.keys()):
        resolve(node_id)

    return ui_outputs
