from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from sortara.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    error_message: str | None = None
    image: Image.Image | None = None
    file_size_kb: float = 0.0
    image_dimensions: tuple[int, int] | None = None


def validate_uploaded_image(file_name: str, file_bytes: bytes) -> ValidationResult:
    """Memvalidasi format ekstensi, ukuran file, dan integritas gambar unggahan (FR-02, FR-03, FR-04)."""
    ext = file_name.rsplit(".", maxsplit=1)[-1].lower().strip() if "." in file_name else ""
    if ext not in ALLOWED_EXTENSIONS:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"Format file '.{ext or 'tidak dikenal'}' tidak didukung. "
                "Silakan unggah foto dengan format JPG, JPEG, atau PNG."
            ),
        )

    size_bytes = len(file_bytes)
    if size_bytes == 0:
        return ValidationResult(
            is_valid=False,
            error_message="File gambar kosong atau tidak dapat dibaca. Silakan pilih foto lain.",
        )

    if size_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"Ukuran file ({size_mb:.1f} MB) melebihi batas maksimal {MAX_FILE_SIZE_MB} MB. "
                "Silakan gunakan foto dengan ukuran lebih kecil."
            ),
        )

    try:
        with Image.open(BytesIO(file_bytes)) as probe:
            probe.verify()

        with Image.open(BytesIO(file_bytes)) as opened:
            rgb_image = opened.convert("RGB")
            dimensions = rgb_image.size

        return ValidationResult(
            is_valid=True,
            image=rgb_image,
            file_size_kb=round(size_bytes / 1024, 1),
            image_dimensions=dimensions,
        )
    except (UnidentifiedImageError, OSError, ValueError):
        return ValidationResult(
            is_valid=False,
            error_message=(
                "File gambar rusak atau tidak dapat dibaca. "
                "Pastikan file merupakan gambar JPG, JPEG, atau PNG yang valid."
            ),
        )
