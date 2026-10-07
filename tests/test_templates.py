"""F18: the shipped workflow templates only use real nodes, inputs and types."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import backend.nodes.generation  # noqa: F401 -- registers the nodes
import backend.nodes.image_ops  # noqa: F401
import backend.nodes.io_nodes  # noqa: F401
from backend.nodes import NODE_REGISTRY

TEMPLATES = Path(__file__).resolve().parents[1] / "frontend" / "templates"
INDEX = json.loads((TEMPLATES / "index.json").read_text(encoding="utf-8"))["templates"]


def _inputs(node_cls) -> dict:
    types = node_cls.INPUT_TYPES()
    return {**types.get("required", {}), **types.get("optional", {})}


def test_the_index_lists_every_template_file():
    assert sorted(entry["file"] for entry in INDEX) == sorted(p.name for p in TEMPLATES.glob("*.json") if p.name != "index.json")
    assert all(entry["title"] and entry["description"] for entry in INDEX)


@pytest.mark.parametrize("file", [entry["file"] for entry in INDEX])
def test_a_template_wires_real_nodes_with_matching_types(file):
    spec = json.loads((TEMPLATES / file).read_text(encoding="utf-8"))
    nodes = {entry["id"]: entry for entry in spec["nodes"]}
    assert len(nodes) == len(spec["nodes"]), "duplicate node id"
    for entry in spec["nodes"]:
        assert entry["type"] in NODE_REGISTRY, entry["type"]
        inputs = _inputs(NODE_REGISTRY[entry["type"]])
        for name in entry.get("values", {}):
            assert name in inputs, f"{entry['id']}: no widget {name!r}"
    connected = set()
    for source, slot, target, input_name in spec["links"]:
        output_type = NODE_REGISTRY[nodes[source]["type"]].RETURN_TYPES[slot]
        input_spec = _inputs(NODE_REGISTRY[nodes[target]["type"]])[input_name]
        assert input_spec[0] == output_type, f"{source}[{slot}] {output_type} -> {target}.{input_name} {input_spec[0]}"
        connected.add((target, input_name))
    for entry in spec["nodes"]:
        required = NODE_REGISTRY[entry["type"]].INPUT_TYPES().get("required", {})
        sockets = [name for name, value in required.items() if isinstance(value[0], str) and value[0].isupper() and value[0] not in ("INT", "FLOAT", "STRING", "BOOLEAN")]
        missing = [name for name in sockets if (entry["id"], name) not in connected]
        assert not missing, f"{file}: {entry['id']} leaves required input(s) {missing} unconnected"
