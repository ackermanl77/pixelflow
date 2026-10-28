import asyncio
import io

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import Headers, UploadFile

from backend import main


def _png_bytes(size=(1, 1)) -> bytes:
    # Generated with Pillow rather than hand-written: the upload endpoint now
    # verifies the bytes decode as a real image (the previous hand-written
    # 1x1 PNG literal had a bad IDAT checksum and fails that check).
    buf = io.BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


PNG_BYTES = _png_bytes()


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


@pytest.mark.parametrize("size", [(4, 1), (3, 2)])
def test_upload_rejects_images_over_dimension_or_pixel_limits(upload_dir, monkeypatch, size):
    monkeypatch.setattr(main, "MAX_UPLOAD_DIMENSION", 3)
    monkeypatch.setattr(main, "MAX_UPLOAD_PIXELS", 5)
    file = _upload("large.png", _png_bytes(size), "image/png")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.upload_image(file))
    assert exc_info.value.status_code == 413
    assert list(upload_dir.iterdir()) == []


def test_upload_accepts_a_valid_png(upload_dir):
    file = _upload("sample.png", PNG_BYTES, "image/png")
    result = asyncio.run(main.upload_image(file))
    assert result == {"filename": "sample.png"}
    assert (upload_dir / "sample.png").read_bytes() == PNG_BYTES


def test_upload_streams_from_spooled_file_without_reading_into_memory(upload_dir):
    file = _upload("sample.png", PNG_BYTES, "image/png")

    async def unexpected_read(*_args, **_kwargs):
        raise AssertionError("upload endpoint should stream from the spooled file")

    file.read = unexpected_read
    result = asyncio.run(main.upload_image(file))
    assert result == {"filename": "sample.png"}
    assert (upload_dir / "sample.png").read_bytes() == PNG_BYTES


def test_upload_rejects_non_image_bytes_with_an_image_content_type(upload_dir):
    file = _upload("fake.png", b"<script>alert(1)</script>", "image/png")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.upload_image(file))
    assert exc_info.value.status_code == 400
    assert list(upload_dir.iterdir()) == []


def test_upload_rejects_empty_filename(upload_dir):
    file = _upload("", PNG_BYTES, "image/png")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.upload_image(file))
    assert exc_info.value.status_code == 400


def test_upload_rejects_empty_file(upload_dir):
    file = _upload("empty.png", b"", "image/png")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.upload_image(file))
    assert exc_info.value.status_code == 400


def test_upload_gives_a_unique_name_instead_of_overwriting(upload_dir):
    first = asyncio.run(main.upload_image(_upload("photo.png", PNG_BYTES, "image/png")))
    second = asyncio.run(main.upload_image(_upload("photo.png", PNG_BYTES, "image/png")))
    third = asyncio.run(main.upload_image(_upload("photo.png", PNG_BYTES, "image/png")))
    assert first == {"filename": "photo.png"}
    assert second == {"filename": "photo_1.png"}
    assert third == {"filename": "photo_2.png"}


def test_upload_sanitizes_characters_that_would_create_an_ntfs_stream(upload_dir):
    result = asyncio.run(main.upload_image(_upload("ab:c?.png", PNG_BYTES, "image/png")))
    assert result == {"filename": "ab_c.png"}
    assert (upload_dir / "ab_c.png").read_bytes() == PNG_BYTES
