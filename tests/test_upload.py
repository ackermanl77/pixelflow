import asyncio
import io

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from backend import main

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01"
    b"\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _upload(filename: str, content: bytes, content_type: str) -> UploadFile:
    return UploadFile(io.BytesIO(content), filename=filename, headers=Headers({"content-type": content_type}))


def test_upload_rejects_unsupported_content_type():
    file = _upload("payload.exe", b"not an image", "application/octet-stream")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.upload_image(file))
    assert exc_info.value.status_code == 400


def test_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    file = _upload("big.png", PNG_BYTES, "image/png")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.upload_image(file))
    assert exc_info.value.status_code == 413


def test_upload_accepts_a_valid_png(upload_dir):
    file = _upload("sample.png", PNG_BYTES, "image/png")
    result = asyncio.run(main.upload_image(file))
    assert result == {"filename": "sample.png"}
    assert (upload_dir / "sample.png").read_bytes() == PNG_BYTES
