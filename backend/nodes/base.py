import logging
from typing import Any, Dict, Tuple, Type

logger = logging.getLogger(__name__)

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
        existing = NODE_REGISTRY.get(node_cls.__name__)
        if existing is not None and existing is not node_cls:
            logger.warning(
                "node class name '%s' is already registered (by %s.%s); "
                "%s.%s will overwrite it in NODE_REGISTRY",
                node_cls.__name__,
                existing.__module__,
                existing.__qualname__,
                node_cls.__module__,
                node_cls.__qualname__,
            )
        NODE_REGISTRY[node_cls.__name__] = node_cls
        return node_cls

    return _wrap


# ---- IMAGE socket helpers ----
#
# An IMAGE value flowing between nodes is either a single PIL.Image (EmptyImage,
# LoadImage) or a list of them (KSampler always outputs a list, one entry per
# `batch_size` image). Every node that consumes IMAGE should go through these
# helpers rather than assuming one shape, so any IMAGE output can be wired into
# any IMAGE input.


def as_image_list(image) -> list:
    """Normalize an IMAGE input to a list of PIL Images."""
    if isinstance(image, (list, tuple)):
        if not image:
            raise ValueError("received an empty image batch")
        return list(image)
    return [image]


def first_image(image):
    """The first image of an IMAGE input (for nodes that only need one, e.g.
    KSampler reading the output size from `latent_image`)."""
    return as_image_list(image)[0]


def map_images(image, fn):
    """Apply `fn` to every image in an IMAGE input, preserving its shape: a
    single image in -> a single image out, a batch (list) in -> a list out."""
    if isinstance(image, (list, tuple)):
        return [fn(img) for img in as_image_list(image)]
    return fn(image)


def map_image_pairs(first, second, fn):
    """Like map_images() for a node with two IMAGE inputs (e.g. Blend). A
    single image on either side is broadcast against every image of a batch
    on the other side; two batches are paired up element-wise and must be
    the same length. Returns a list if either input was a batch."""
    is_batch = isinstance(first, (list, tuple)) or isinstance(second, (list, tuple))
    firsts, seconds = as_image_list(first), as_image_list(second)
    if len(firsts) != len(seconds):
        if len(firsts) == 1:
            firsts = firsts * len(seconds)
        elif len(seconds) == 1:
            seconds = seconds * len(firsts)
        else:
            raise ValueError(
                f"cannot pair image batches of different sizes ({len(firsts)} vs {len(seconds)})"
            )
    results = [fn(a, b) for a, b in zip(firsts, seconds)]
    return results if is_batch else results[0]
