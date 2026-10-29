from io import BytesIO

from PIL import Image

from sortara.config import MAX_FILE_SIZE_BYTES
from sortara.validator import validate_uploaded_image


def _make_image_bytes(fmt: str = "JPEG", size: tuple[int, int] = (300, 300), mode: str = "RGB") -> bytes:
    buf = BytesIO()
    img = Image.new(mode, size, color=(40, 160, 80))
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_accepts_valid_jpg_image_at01():
    """AT-01: Mengunggah gambar JPG yang valid -> dapat dibaca tanpa error."""
    raw = _make_image_bytes("JPEG")
    res = validate_uploaded_image("botol_plastik.jpg", raw)
    assert res.is_valid is True
    assert res.error_message is None
    assert res.image is not None
    assert res.image.mode == "RGB"


def test_accepts_valid_png_rgba_image_and_converts_to_rgb_at02():
    """AT-02: Mengunggah gambar PNG (termasuk RGBA) -> dapat dibaca dan dikonversi ke RGB."""
    raw = _make_image_bytes("PNG", mode="RGBA")
    res = validate_uploaded_image("kardus.PNG", raw)
    assert res.is_valid is True
    assert res.error_message is None
    assert res.image is not None
    assert res.image.mode == "RGB"


def test_rejects_unsupported_extension_at03():
    """AT-03: Mengunggah format yang tidak didukung (mis. PDF/GIF/TXT) -> ditolak dengan pesan jelas."""
    raw = _make_image_bytes("JPEG")
    res = validate_uploaded_image("dokumen.pdf", raw)
    assert res.is_valid is False
    assert res.image is None
    assert res.error_message is not None
    assert "format" in res.error_message.lower()


def test_rejects_file_exceeding_max_size_at04():
    """AT-04: Mengunggah file yang melebihi batas 10 MB -> sistem menolak file dan memberi tahu pengguna."""
    oversized = b"x" * (MAX_FILE_SIZE_BYTES + 1)
    res = validate_uploaded_image("besar.jpg", oversized)
    assert res.is_valid is False
    assert res.image is None
    assert res.error_message is not None
    assert "10 mb" in res.error_message.lower()


def test_rejects_empty_or_corrupted_image_bytes_fr03():
    """FR-03: Sistem menolak file yang rusak / tidak dapat dibaca dan menampilkan pesan yang jelas."""
    res_empty = validate_uploaded_image("kosong.jpg", b"")
    assert res_empty.is_valid is False
    assert res_empty.error_message is not None

    res_corrupt = validate_uploaded_image("rusak.png", b"bukan_isi_gambar_valid_12345")
    assert res_corrupt.is_valid is False
    assert res_corrupt.image is None
    assert res_corrupt.error_message is not None
    assert "rusak" in res_corrupt.error_message.lower() or "dibaca" in res_corrupt.error_message.lower()
