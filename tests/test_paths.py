import pytest

from backend.paths import PathTraversalError, safe_path


def test_safe_path_allows_a_plain_filename(tmp_path):
    result = safe_path(tmp_path, "image.png")
    assert result == (tmp_path / "image.png").resolve()


def test_safe_path_rejects_dotdot_traversal(tmp_path):
    with pytest.raises(PathTraversalError):
        safe_path(tmp_path, "../../etc/passwd")


def test_safe_path_rejects_absolute_path(tmp_path):
    with pytest.raises(PathTraversalError):
        safe_path(tmp_path, "/etc/passwd")
