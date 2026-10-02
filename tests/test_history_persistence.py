"""Persistent history and queue bookkeeping (report item B10)."""

import json

from backend.engine.queue import ConnectionManager, ExecutionQueue


def _queue(path=None, max_history=200):
    return ExecutionQueue(ConnectionManager(), max_history=max_history, history_path=path)


def test_terminal_entries_are_appended_and_restored(tmp_path):
    path = tmp_path / "history.jsonl"
    q = _queue(path)
    q._record_history("a", {"status": "queued", "prompt": {"1": {}}, "queued_at": 1.0})
    q._record_history("a", {"status": "running", "started_at": 2.0})
    assert not path.exists(), "only finished entries are persisted"
    q._record_history("a", {"status": "completed", "outputs": {"2": {"images": ["x.png"]}}, "finished_at": 3.0})
    q._record_history("b", {"status": "queued"})
    q.cancel("b")

    restored = _queue(path)
    assert restored.load_history() == 2
    assert list(restored.history) == ["a", "b"]
    a = restored.history["a"]
    assert a["status"] == "completed" and a["prompt"] == {"1": {}}
    assert (a["queued_at"], a["started_at"], a["finished_at"]) == (1.0, 2.0, 3.0)
    assert restored.history["b"]["status"] == "cancelled"


def test_deleted_entries_stay_deleted_after_restart(tmp_path):
    path = tmp_path / "history.jsonl"
    q = _queue(path)
    q._record_history("a", {"status": "completed"})
    q._record_history("b", {"status": "error", "error": "x"})
    assert q.delete_history("a")
    restored = _queue(path)
    restored.load_history()
    assert list(restored.history) == ["b"]


def test_load_keeps_the_newest_and_compacts_the_file(tmp_path):
    path = tmp_path / "history.jsonl"
    q = _queue(path, max_history=3)
    for i in range(10):
        q._record_history(f"p{i}", {"status": "completed"})
    restored = _queue(path, max_history=3)
    assert restored.load_history() == 3
    assert list(restored.history) == ["p7", "p8", "p9"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_corrupt_lines_are_skipped(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps({"prompt_id": "ok", "entry": {"status": "completed"}}) + "\n{truncated by a cra\n", encoding="utf-8"
    )
    q = _queue(path)
    assert q.load_history() == 1
    assert "ok" in q.history


def test_queued_entries_are_never_evicted(tmp_path):
    q = _queue(max_history=2)
    for i in range(3):
        q._record_history(f"q{i}", {"status": "queued"})
    assert list(q.history) == ["q0", "q1", "q2"]
    assert q.cancel("q0"), "a long queue must still be cancellable from the front"
    q._record_history("done", {"status": "completed"})
    # Over the cap: the oldest *finished* entries go first.
    assert "q1" in q.history and "q2" in q.history
    assert len(q.history) <= 3


def test_queue_state_lists_pending_in_order():
    q = _queue()
    for pid in ("a", "b", "c"):
        q._record_history(pid, {"status": "queued"})
    q.cancel("b")
    q.running_prompt_id = "r"
    assert q.queue_state() == {"running": "r", "pending": ["a", "c"]}


def test_history_page():
    q = _queue()
    for i in range(4):
        q._record_history(f"p{i}", {"status": "completed"})
    assert list(q.history_page(limit=2)) == ["p2", "p3"]
    assert list(q.history_page(before="p2")) == ["p0", "p1"]
    assert q.history_page(before="unknown") == {}
