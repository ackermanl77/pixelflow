import copy
import hashlib
import inspect
import json
import logging
import math
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..nodes import NODE_REGISTRY, resolve_class_type

logger = logging.getLogger(__name__)


class GraphError(Exception):
    """Raised for structural problems in a prompt graph: an unknown node id,
    an unknown class_type, or a cycle between node inputs."""


class PromptValidationError(GraphError):
    """Raised by validate_prompt() for a graph that can't run as submitted.
    `node_id` names the offending node (None for graph-wide problems),
    `class_type` its node class and `input_name` the offending input, when
    known, so the API can hand the frontend enough to highlight the exact
    node/input (see to_dict())."""

    def __init__(
        self,
        node_id: Optional[str],
        message: str,
        *,
        class_type: Optional[str] = None,
        input_name: Optional[str] = None,
    ) -> None:
        self.node_id = node_id
        self.class_type = class_type
        self.input_name = input_name
        self.reason = message
        super().__init__(f"Node '{node_id}': {message}" if node_id is not None else message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": str(self),
            "node_id": self.node_id,
            "class_type": self.class_type,
            "input": self.input_name,
        }


# How much of a failing node's traceback is kept in NodeExecutionError /
# history entries / execution_error messages (the tail is the useful part).
MAX_TRACEBACK_CHARS = 4000


class NodeExecutionError(GraphError):
    """Raised by execute_prompt() when a node's FUNCTION raises. Wraps the
    original exception (kept as __cause__) with the node it came from, so
    the queue can report *which* node failed instead of a bare message."""

    def __init__(self, node_id: str, class_type: str, exc: BaseException) -> None:
        self.node_id = node_id
        self.class_type = class_type
        self.exception_type = type(exc).__name__
        self.reason = str(exc) or self.exception_type
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.traceback = tb[-MAX_TRACEBACK_CHARS:]
        super().__init__(f"{class_type} (node '{node_id}'): {self.reason}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": str(self),
            "node_id": self.node_id,
            "class_type": self.class_type,
            "exception_type": self.exception_type,
            "traceback": self.traceback,
        }


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

    def fail(message: str) -> PromptValidationError:
        return PromptValidationError(node_id, message, input_name=name)

    if isinstance(input_type, (list, tuple)):
        if config.get("allow_custom") and isinstance(value, str):
            # A combo that also accepts values outside its list (e.g. Load
            # Checkpoint's Hugging Face Hub ids); the node's own
            # VALIDATE_INPUTS() checks those further.
            return value
        if value not in input_type:
            raise fail(f"input '{name}' must be one of {list(input_type)}, got {value!r}")
        return value

    if input_type in ("INT", "FLOAT"):
        # bool is an int subclass, but True/False is never a meaningful number here.
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise fail(f"input '{name}' must be a number, got {value!r}")
        if input_type == "INT":
            if value != int(value):
                raise fail(f"input '{name}' must be an integer, got {value!r}")
            value = int(value)
        else:
            value = float(value)
        minimum, maximum = config.get("min"), config.get("max")
        if minimum is not None and value < minimum:
            raise fail(f"input '{name}' is {value}, below the minimum of {minimum}")
        if maximum is not None and value > maximum:
            raise fail(f"input '{name}' is {value}, above the maximum of {maximum}")
        return value

    if input_type == "STRING":
        if not isinstance(value, str):
            raise fail(f"input '{name}' must be a string, got {value!r}")
        return value

    if input_type == "BOOLEAN":
        if not isinstance(value, bool):
            raise fail(f"input '{name}' must be true or false, got {value!r}")
        return value

    raise fail(f"input '{name}' ({input_type}) must be connected to another node's output")


def _is_output_node(spec: Any) -> bool:
    if not isinstance(spec, dict):
        return False
    node_cls = NODE_REGISTRY.get(resolve_class_type(spec.get("class_type")))
    return bool(getattr(node_cls, "OUTPUT_NODE", False))


def output_node_ids(prompt: Dict[str, Dict[str, Any]]) -> List[str]:
    """Ids (as strings, in prompt order) of the nodes whose class declares
    OUTPUT_NODE = True - the roots execution is driven from."""
    return [str(node_id) for node_id, spec in prompt.items() if _is_output_node(spec)]


def _ancestors_of(prompt: Dict[str, Dict[str, Any]], roots: List[str]) -> Set[str]:
    """`roots` plus every node they (transitively) take a linked input from.
    Links to ids that aren't in the prompt are skipped here; the consumer's
    own validation reports them."""
    needed: Set[str] = set()
    stack = list(roots)
    while stack:
        node_id = stack.pop()
        if node_id in needed:
            continue
        needed.add(node_id)
        spec = prompt.get(node_id)
        if not isinstance(spec, dict):
            continue
        for value in (spec.get("inputs") or {}).values():
            if _is_link(value) and str(value[0]) in prompt:
                stack.append(str(value[0]))
    return needed


def validate_prompt(prompt: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Check a prompt graph against each node's INPUT_TYPES/RETURN_TYPES
    before it's queued, so a bad graph fails fast with a message naming the
    node instead of partway through execution.

    Like ComfyUI, only the part of the graph the output nodes (OUTPUT_NODE =
    True, e.g. Save/Preview Image) actually need is validated and returned:
    a half-built side branch no output depends on is dropped instead of
    blocking the run, and a prompt with no output node at all is rejected.
    Checks on that subgraph: known class_type, required inputs present, no
    undeclared inputs, links pointing at an existing node and an in-range
    output whose type matches the input's, numbers within INPUT_TYPES
    min/max, combo values in their list, and no cycles. Returns a normalized
    deep copy of the needed nodes (e.g. INT values coerced to int). Raises
    PromptValidationError (with node_id/class_type/input_name filled in
    where they apply)."""
    if not prompt:
        raise PromptValidationError(None, "The prompt contains no nodes")

    by_id = {str(node_id): spec for node_id, spec in prompt.items()}
    outputs = output_node_ids(by_id)
    if not outputs:
        raise PromptValidationError(None, "Prompt has no output nodes - add a Save Image or Preview Image node")
    needed = _ancestors_of(by_id, outputs)

    normalized: Dict[str, Dict[str, Any]] = {}
    for node_id, spec in by_id.items():
        if node_id not in needed:
            continue
        # Old class names (NODE_ALIASES) are rewritten to the current ones.
        class_type = resolve_class_type(spec.get("class_type"))
        try:
            normalized[node_id] = {
                "class_type": class_type,
                "inputs": _validate_node_inputs(by_id, node_id, spec),
            }
        except PromptValidationError as exc:
            if exc.node_id == node_id and exc.class_type is None:
                exc.class_type = class_type
            raise

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
                        raise PromptValidationError(
                            dep, "cycle detected in node graph", class_type=normalized[dep]["class_type"]
                        )
                    if color[dep] == WHITE:
                        color[dep] = GRAY
                        stack.append((dep, iter(normalized[dep]["inputs"].values())))
                        advanced = True
                        break
            if not advanced:
                color[node_id] = BLACK
                stack.pop()

    return normalized


def _validate_node_inputs(by_id: Dict[str, Dict[str, Any]], node_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    class_type = resolve_class_type(spec.get("class_type"))
    node_cls = NODE_REGISTRY.get(class_type)
    if node_cls is None:
        raise PromptValidationError(node_id, f"unknown node class '{class_type}'")

    input_types = node_cls.INPUT_TYPES()
    required = input_types.get("required", {})
    declared = _declared_inputs(node_cls)
    inputs = spec.get("inputs") or {}

    for name in required:
        if name not in inputs:
            raise PromptValidationError(
                node_id, f"required input '{name}' is missing (not connected or not set)", input_name=name
            )

    normalized_inputs: Dict[str, Any] = {}
    for name, value in inputs.items():
        if name not in declared:
            raise PromptValidationError(node_id, f"unknown input '{name}' for {class_type}", input_name=name)
        input_spec = declared[name]
        if _is_link(value):
            src_id, out_idx = str(value[0]), value[1]
            src_spec = by_id.get(src_id)
            if src_spec is None:
                raise PromptValidationError(
                    node_id, f"input '{name}' is linked to missing node '{src_id}'", input_name=name
                )
            src_cls = NODE_REGISTRY.get(resolve_class_type(src_spec.get("class_type")))
            if src_cls is None:
                raise PromptValidationError(
                    src_id,
                    f"unknown node class '{src_spec.get('class_type')}'",
                    class_type=src_spec.get("class_type"),
                )
            return_types = tuple(src_cls.RETURN_TYPES)
            if not 0 <= out_idx < len(return_types):
                raise PromptValidationError(
                    node_id,
                    f"input '{name}' is linked to output {out_idx} of node '{src_id}', which has "
                    f"{len(return_types)} output(s)",
                    input_name=name,
                )
            expected = input_spec[0]
            actual = return_types[out_idx]
            if isinstance(expected, (list, tuple)) or expected != actual:
                raise PromptValidationError(
                    node_id,
                    f"input '{name}' expects {expected if isinstance(expected, str) else 'a choice'} "
                    f"but is linked to a {actual} output of node '{src_id}'",
                    input_name=name,
                )
            normalized_inputs[name] = [src_id, out_idx]
        else:
            normalized_inputs[name] = _validate_value(node_id, name, input_spec, value)
    _run_validate_inputs(node_cls, node_id, normalized_inputs)
    return normalized_inputs


def _run_validate_inputs(node_cls, node_id: str, inputs: Dict[str, Any]) -> None:
    """Call the class's optional VALIDATE_INPUTS(**widget_values) hook. It
    returns True, or a message; a message (or an exception) is a
    PromptValidationError for this node."""
    hook = getattr(node_cls, "VALIDATE_INPUTS", None)
    if hook is None:
        return
    literal = {name: value for name, value in inputs.items() if not _is_link(value)}
    try:
        result = hook(**literal)
    except Exception as exc:  # noqa: BLE001 - a broken hook is reported, not a 500
        result = f"{type(exc).__name__}: {exc}"
    if result is True or result is None:
        return
    message = str(result)
    # Name the input when the message is about one (e.g. "'x' is not a colour").
    input_name = next((name for name in literal if f"'{literal[name]}'" in message), None)
    raise PromptValidationError(node_id, message, input_name=input_name)


CacheEntry = Tuple[str, Tuple[Any, ...], Optional[Dict[str, Any]]]


class OutputCache:
    """Node outputs kept between runs (like ComfyUI), so re-queueing a graph
    after tweaking e.g. a Blur radius downstream of a fixed-seed KSampler
    reuses the sampled image instead of re-running diffusion.

    Entries are keyed by node id and tagged with a *signature*: a hash of
    the node's class, its literal inputs, the signatures of the nodes
    feeding its linked inputs, and its class's optional IS_CHANGED() result
    (see execute_prompt). A node is reused only when its signature matches.
    Only the most recent run's entries are kept (plus, after a failed or
    interrupted run, the older entries it didn't get to), which bounds
    memory. Output nodes are never cached, and classes with
    CACHE_OUTPUTS = False (LoadCheckpoint, whose pipelines live in their
    own LRU) aren't stored.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, CacheEntry] = {}

    def get(self, node_id: str, signature: str) -> Optional[Tuple[Tuple[Any, ...], Optional[Dict[str, Any]]]]:
        entry = self._entries.get(node_id)
        if entry is not None and entry[0] == signature:
            return entry[1], entry[2]
        return None

    def commit(self, entries: Dict[str, CacheEntry], complete: bool) -> None:
        """Replace the cache with a finished run's entries (`complete`), or
        merge a failed/interrupted run's entries over the existing ones."""
        if complete:
            self._entries = dict(entries)
        else:
            self._entries.update(entries)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._entries


def _is_changed_token(node_cls, literal_inputs: Dict[str, Any]) -> Any:
    """Call a node class's optional IS_CHANGED(**literal_inputs) hook. A NaN
    result (or an exception) means "always re-run": it's replaced by a
    fresh random token so the signature never matches a cached one, and
    neither does any downstream node's."""
    hook = getattr(node_cls, "IS_CHANGED", None)
    if hook is None:
        return None
    try:
        token = hook(**literal_inputs)
    except Exception:  # noqa: BLE001 - a broken hook only disables caching, never the run
        logger.debug("IS_CHANGED of %s raised; not caching it", node_cls.__name__, exc_info=True)
        token = float("nan")
    if isinstance(token, float) and math.isnan(token):
        return uuid.uuid4().hex
    return token


def execute_prompt(
    prompt: Dict[str, Dict[str, Any]],
    on_node_start: Optional[Callable[[str, str], None]] = None,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
    should_interrupt: Optional[Callable[[], bool]] = None,
    extra_pnginfo: Optional[Dict[str, Any]] = None,
    output_cache: Optional[OutputCache] = None,
    on_node_cached: Optional[Callable[[str, str], None]] = None,
    on_progress_detail: Optional[Callable[[str, int, int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Execute a graph given in ComfyUI-style "prompt" form:

        {node_id: {"class_type": str, "inputs": {name: value | [node_id, output_index]}}}

    Execution is driven from the output nodes (OUTPUT_NODE = True): only
    they and the nodes they depend on run. A prompt with no output node at
    all (only possible when calling this directly - validate_prompt()
    rejects it) runs every node.

    Returns a dict of {node_id: ui_dict} collected from nodes that return a
    dict (e.g. SaveImage's {"images": [filename, ...]}, KSampler's
    {"seed": N}); a dict's "result" key, if any, is the node's output tuple
    and is not part of the ui_dict.

    An exception raised by a node's FUNCTION is re-raised as
    NodeExecutionError (node id, class, exception type and a trimmed
    traceback; the original exception is its __cause__).

    `on_progress`, if given, is called as `on_progress(node_id, step, total)`
    for in-progress work within a single node (e.g. KSampler's diffusion
    steps). It's only wired into a node's FUNCTION call when that function
    declares an `on_progress` parameter (checked via introspection here rather
    than by changing every node's call signature) - most nodes finish in one
    shot and have no sub-steps to report.

    `on_progress_detail`, if given, is called instead of on_progress as
    `on_progress_detail(node_id, step, total, info)`, where `info` may hold
    `it_per_s` and `eta_s` (from the step timing), `unit` ("bytes" for a
    download) and `preview` (a live latent preview as a JPEG data URL; a
    node only computes previews when this callback is used).

    `should_interrupt`, if given, is polled before every node and on every
    `on_progress` step; once it returns True, ExecutionInterrupted is raised
    (from inside the node for a per-step check, e.g. out of diffusers'
    step callback), abandoning the rest of the prompt.

    `output_cache`, if given, serves nodes whose signature is unchanged
    since an earlier run from it (see OutputCache) instead of executing
    them, and is updated with this run's outputs; `on_node_cached(node_id,
    class_type)` is called for each node served that way. Without it every
    needed node runs.

    Nodes can declare ComfyUI-style hidden inputs in INPUT_TYPES()["hidden"]
    ({name: "PROMPT" | "EXTRA_PNGINFO" | "UNIQUE_ID"}); they're injected here
    rather than coming from the graph. PROMPT is the prompt as executed so
    far, with any node's reported {"seed": N} written back into its `seed`
    input so the embedded graph reproduces the result.
    """

    results: Dict[str, Tuple[Any, ...]] = {}
    ui_outputs: Dict[str, Any] = {}
    in_progress: set = set()
    resolved_prompt = copy.deepcopy(prompt)
    signatures: Dict[str, Optional[str]] = {}
    new_cache_entries: Dict[str, CacheEntry] = {}

    def check_interrupt() -> None:
        if should_interrupt is not None and should_interrupt():
            raise ExecutionInterrupted("Prompt execution was interrupted")

    def signature(node_id: str, visiting: Optional[set] = None) -> Optional[str]:
        """Bottom-up cache signature, or None when the node can't be cached
        (a missing/unknown node or a cycle upstream - resolve() reports
        those as errors)."""
        if node_id in signatures:
            return signatures[node_id]
        visiting = visiting if visiting is not None else set()
        spec = prompt.get(node_id)
        node_cls = NODE_REGISTRY.get(resolve_class_type(spec.get("class_type"))) if isinstance(spec, dict) else None
        if node_cls is None or node_id in visiting:
            return None
        visiting.add(node_id)
        literal: Dict[str, Any] = {}
        links: Dict[str, Any] = {}
        try:
            for name, value in (spec.get("inputs") or {}).items():
                if _is_link(value):
                    dep = signature(str(value[0]), visiting)
                    if dep is None:
                        return None
                    links[name] = [dep, value[1]]
                else:
                    literal[name] = value
        finally:
            visiting.discard(node_id)
        token = _is_changed_token(node_cls, literal)
        payload = json.dumps([spec.get("class_type"), literal, links, token], sort_keys=True, default=repr)
        signatures[node_id] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return signatures[node_id]

    def make_progress(node_id: str):
        timing: Dict[str, float] = {}

        def _node_progress(step, total, **info):
            check_interrupt()
            if info.get("unit") is None:
                now = time.monotonic()
                if "start" not in timing or step <= timing.get("first_step", step):
                    timing.update(start=now, first_step=step)
                elif now > timing["start"]:
                    rate = (step - timing["first_step"]) / (now - timing["start"])
                    if rate > 0:
                        info["it_per_s"] = round(rate, 3)
                        info["eta_s"] = round(max(0, total - step) / rate, 1)
            if on_progress_detail is not None:
                on_progress_detail(node_id, step, total, info)
            elif on_progress is not None:
                on_progress(node_id, step, total)

        _node_progress.wants_previews = on_progress_detail is not None
        return _node_progress

    def record_ui(node_id: str, ui: Optional[Dict[str, Any]]) -> None:
        if not ui:
            return
        ui_outputs[node_id] = ui
        if "seed" in ui and "seed" in resolved_prompt.get(node_id, {}).get("inputs", {}):
            resolved_prompt[node_id]["inputs"]["seed"] = ui["seed"]

    def resolve(node_id: str) -> Tuple[Any, ...]:
        if node_id in results:
            return results[node_id]
        if node_id in in_progress:
            raise GraphError(f"Cycle detected in node graph at node '{node_id}'")

        try:
            spec = prompt[node_id]
        except KeyError:
            raise GraphError(f"Node '{node_id}' is referenced but not present in the prompt") from None

        class_type = resolve_class_type(spec.get("class_type"))
        try:
            node_cls = NODE_REGISTRY[class_type]
        except KeyError:
            raise GraphError(f"Unknown node class '{class_type}' (node '{node_id}')") from None

        cacheable = (
            output_cache is not None
            and not getattr(node_cls, "OUTPUT_NODE", False)
            and getattr(node_cls, "CACHE_OUTPUTS", True)
        )
        sig = signature(node_id) if cacheable else None
        if sig is not None:
            hit = output_cache.get(node_id, sig)
            if hit is not None:
                outputs, ui = hit
                record_ui(node_id, ui)
                results[node_id] = outputs
                new_cache_entries[node_id] = (sig, outputs, ui)
                logger.debug("reusing cached outputs of node %s (%s)", node_id, class_type)
                if on_node_cached:
                    on_node_cached(node_id, class_type)
                return outputs

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
            if (on_progress is not None or on_progress_detail is not None or should_interrupt is not None) and (
                "on_progress" in inspect.signature(func).parameters
            ):
                kwargs["on_progress"] = make_progress(node_id)
            try:
                result = func(**kwargs)
            except ExecutionInterrupted:
                raise
            except Exception as exc:
                logger.exception("node %s (%s) raised during execution", node_id, class_type)
                raise NodeExecutionError(node_id, class_type, exc) from exc
        finally:
            in_progress.discard(node_id)

        ui: Optional[Dict[str, Any]] = None
        if isinstance(result, dict):
            ui = {key: value for key, value in result.items() if key != "result"} or None
            record_ui(node_id, ui)
            outputs: Tuple[Any, ...] = tuple(result.get("result", ()))
        else:
            outputs = result if isinstance(result, tuple) else (result,)

        results[node_id] = outputs
        if sig is not None:
            new_cache_entries[node_id] = (sig, outputs, ui)
        return outputs

    # Keys are used as given (not str()-ed) so a caller's own id type keeps working.
    targets = [node_id for node_id, spec in prompt.items() if _is_output_node(spec)] or list(prompt.keys())
    complete = False
    try:
        for node_id in targets:
            resolve(node_id)
        complete = True
    finally:
        if output_cache is not None:
            output_cache.commit(new_cache_entries, complete)

    return ui_outputs
