from typing import Any, Dict, Tuple, Type

NODE_REGISTRY: Dict[str, Type["NodeBase"]] = {}


class NodeBase:
    """Base class for a node. Mirrors ComfyUI's node contract:
    INPUT_TYPES describes widgets/sockets, RETURN_TYPES describes outputs,
    FUNCTION names the method the engine calls to execute the node.
    """

    CATEGORY = "misc"
    RETURN_TYPES: Tuple[str, ...] = ()
    RETURN_NAMES: Tuple[str, ...] = ()
    FUNCTION: str = "execute"
    OUTPUT_NODE: bool = False
    DISPLAY_NAME: str = ""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {"required": {}}


def register_node(display_name: str = ""):
    """Class decorator that registers a node under its class name."""

    def _wrap(node_cls: Type[NodeBase]):
        node_cls.DISPLAY_NAME = display_name or node_cls.__name__
        if not node_cls.RETURN_NAMES:
            node_cls.RETURN_NAMES = node_cls.RETURN_TYPES
        NODE_REGISTRY[node_cls.__name__] = node_cls
        return node_cls

    return _wrap
