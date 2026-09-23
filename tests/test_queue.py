from backend.engine.queue import ConnectionManager, ExecutionQueue


def test_record_history_caps_at_max_history():
    queue = ExecutionQueue(ConnectionManager(), max_history=3)
    for i in range(5):
        queue._record_history(f"id{i}", {"status": "completed"})
    assert list(queue.history.keys()) == ["id2", "id3", "id4"]


def test_record_history_updating_existing_key_keeps_its_slot():
    queue = ExecutionQueue(ConnectionManager(), max_history=3)
    queue._record_history("a", {"status": "queued"})
    queue._record_history("b", {"status": "queued"})
    queue._record_history("c", {"status": "queued"})
    queue._record_history("a", {"status": "completed"})
    assert list(queue.history.keys()) == ["a", "b", "c"]
    assert queue.history["a"]["status"] == "completed"


def test_cancel_marks_a_queued_prompt_cancelled():
    queue = ExecutionQueue(ConnectionManager())
    queue._record_history("a", {"status": "queued"})
    assert queue.cancel("a") is True
    assert queue.history["a"]["status"] == "cancelled"
    assert "a" in queue._cancelled


def test_cancel_refuses_an_unknown_id():
    queue = ExecutionQueue(ConnectionManager())
    assert queue.cancel("nope") is False


def test_cancel_refuses_a_running_or_finished_prompt():
    queue = ExecutionQueue(ConnectionManager())
    queue._record_history("running", {"status": "running"})
    queue._record_history("done", {"status": "completed"})
    assert queue.cancel("running") is False
    assert queue.cancel("done") is False


def test_delete_history_removes_a_finished_entry():
    queue = ExecutionQueue(ConnectionManager())
    queue._record_history("done", {"status": "completed"})
    assert queue.delete_history("done") is True
    assert "done" not in queue.history


def test_delete_history_refuses_queued_or_running_or_unknown():
    queue = ExecutionQueue(ConnectionManager())
    queue._record_history("queued", {"status": "queued"})
    queue._record_history("running", {"status": "running"})
    assert queue.delete_history("queued") is False
    assert queue.delete_history("running") is False
    assert queue.delete_history("nope") is False
