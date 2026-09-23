import pytest
from pydantic import ValidationError

from backend.main import NodeSpec, QueueRequest, WorkflowPayload


def test_node_spec_requires_class_type():
    with pytest.raises(ValidationError):
        NodeSpec(inputs={})


def test_node_spec_defaults_inputs_to_empty_dict():
    spec = NodeSpec(class_type="EmptyImage")
    assert spec.inputs == {}


def test_queue_request_rejects_node_missing_class_type():
    with pytest.raises(ValidationError):
        QueueRequest(prompt={"1": {"inputs": {}}})


def test_queue_request_accepts_a_valid_prompt():
    request = QueueRequest(prompt={"1": {"class_type": "EmptyImage", "inputs": {"width": 64}}})
    assert request.prompt["1"].class_type == "EmptyImage"


def test_workflow_payload_allows_arbitrary_fields():
    payload = WorkflowPayload(nodes=[{"id": 1}], links=[], extra_field="ok")
    assert payload.model_dump()["extra_field"] == "ok"
