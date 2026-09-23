import logging

from backend.nodes.base import NODE_REGISTRY, NodeBase, register_node


def test_register_node_warns_on_name_collision(caplog):
    NODE_REGISTRY.pop("DuplicateNode", None)
    try:

        @register_node()
        class DuplicateNode(NodeBase):
            pass

        with caplog.at_level(logging.WARNING, logger="backend.nodes.base"):

            @register_node()
            class DuplicateNode(NodeBase):  # noqa: F811 - intentional redefinition
                pass

        assert any("already registered" in record.message for record in caplog.records)
        assert NODE_REGISTRY["DuplicateNode"] is DuplicateNode
    finally:
        NODE_REGISTRY.pop("DuplicateNode", None)


def test_register_node_does_not_warn_for_a_new_name(caplog):
    NODE_REGISTRY.pop("BrandNewNode", None)
    try:
        with caplog.at_level(logging.WARNING, logger="backend.nodes.base"):

            @register_node()
            class BrandNewNode(NodeBase):
                pass

        assert caplog.records == []
    finally:
        NODE_REGISTRY.pop("BrandNewNode", None)
