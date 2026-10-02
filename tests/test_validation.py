import pytest

from backend.engine.graph import PromptValidationError, validate_prompt


def _valid_graph():
    return {
        "ckpt": {"class_type": "LoadCheckpoint", "inputs": {"ckpt_name": "stabilityai/sd-turbo"}},
        "clip": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}},
        "latent": {"class_type": "EmptyImage", "inputs": {"width": 512, "height": 512, "color": "#000000"}},
        "sampler": {
            "class_type": "KSamplerStub",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["clip", 0],
                "latent_image": ["latent", 0],
                "sampler": "euler",
                "seed": -1,
                "steps": 4,
                "cfg": 0,
                "batch_size": 1,
            },
        },
        "save": {"class_type": "SaveImage", "inputs": {"image": ["sampler", 0], "filename_prefix": "x"}},
    }


def _expect_error(graph, node_id, match):
    with pytest.raises(PromptValidationError, match=match) as exc_info:
        validate_prompt(graph)
    assert exc_info.value.node_id == node_id
    assert f"Node '{node_id}'" in str(exc_info.value)


def test_a_valid_graph_passes_and_is_normalized():
    graph = _valid_graph()
    graph["latent"]["inputs"]["width"] = 512.0  # JSON clients may send integral floats
    normalized = validate_prompt(graph)
    assert normalized["latent"]["inputs"]["width"] == 512
    assert isinstance(normalized["latent"]["inputs"]["width"], int)
    assert isinstance(normalized["sampler"]["inputs"]["cfg"], float)


def test_missing_required_input():
    graph = _valid_graph()
    del graph["sampler"]["inputs"]["model"]
    _expect_error(graph, "sampler", "required input 'model' is missing")


def test_optional_input_may_be_omitted_but_is_checked_when_present():
    graph = _valid_graph()
    validate_prompt(graph)  # no `negative`
    graph["sampler"]["inputs"]["negative"] = ["latent", 0]
    _expect_error(graph, "sampler", "expects CONDITIONING but is linked to a IMAGE")


def test_link_to_missing_node():
    graph = _valid_graph()
    graph["save"]["inputs"]["image"] = ["nope", 0]
    _expect_error(graph, "save", "missing node 'nope'")


def test_link_output_index_out_of_range():
    graph = _valid_graph()
    graph["save"]["inputs"]["image"] = ["sampler", 1]
    _expect_error(graph, "save", "has 1 output")


def test_link_type_mismatch():
    graph = _valid_graph()
    graph["sampler"]["inputs"]["positive"] = ["ckpt", 0]
    _expect_error(graph, "sampler", "expects CONDITIONING but is linked to a MODEL")


def test_socket_input_given_a_literal_value():
    graph = _valid_graph()
    graph["save"]["inputs"]["image"] = "not a link"
    _expect_error(graph, "save", "must be connected")


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("steps", 0, "below the minimum of 1"),
        ("steps", 151, "above the maximum of 150"),
        ("steps", 2.5, "must be an integer"),
        ("steps", "4", "must be a number"),
        ("steps", True, "must be a number"),
        ("cfg", 31.0, "above the maximum"),
        ("sampler", "not_a_sampler", "must be one of"),
    ],
)
def test_widget_values_are_checked_against_input_types(name, value, match):
    graph = _valid_graph()
    graph["sampler"]["inputs"][name] = value
    _expect_error(graph, "sampler", match)


def test_string_input_must_be_a_string():
    graph = _valid_graph()
    graph["clip"]["inputs"]["text"] = 123
    _expect_error(graph, "clip", "must be a string")


def test_unknown_input_and_unknown_class():
    graph = _valid_graph()
    graph["clip"]["inputs"]["bogus"] = 1
    _expect_error(graph, "clip", "unknown input 'bogus'")
    graph = _valid_graph()
    graph["clip"]["class_type"] = "NotARealNode"
    _expect_error(graph, "clip", "unknown node class")


def test_cycle_is_rejected_before_execution():
    graph = {
        "a": {"class_type": "ImageBlur", "inputs": {"image": ["b", 0], "radius": 1.0}},
        "b": {"class_type": "ImageBlur", "inputs": {"image": ["a", 0], "radius": 1.0}},
        "out": {"class_type": "PreviewImage", "inputs": {"image": ["b", 0]}},
    }
    with pytest.raises(PromptValidationError, match="cycle"):
        validate_prompt(graph)


def test_empty_prompt_is_rejected():
    with pytest.raises(PromptValidationError):
        validate_prompt({})
