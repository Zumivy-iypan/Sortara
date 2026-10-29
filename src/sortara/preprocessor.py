from dataclasses import dataclass

import numpy as np
from PIL import Image

from sortara.config import INPUT_IMAGE_SIZE


@dataclass(frozen=True)
class ImageQualityMetrics:
    brightness_mean: float
    contrast_std: float
    edge_strength: float
    is_low_quality: bool
    quality_note: str | None = None


def preprocess_for_mobilenetv2(image: Image.Image) -> np.ndarray:
    """Mengubah gambar PIL ke tensor (1, 224, 224, 3) float32 pada rentang [-1.0, 1.0] (FR-06).

    Fungsi ini identik secara matematis dengan `tf.keras.applications.mobilenet_v2.preprocess_input`:
    `(pixel / 127.5) - 1.0`.
    """
    rgb_img = image.convert("RGB").resize(INPUT_IMAGE_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(rgb_img, dtype=np.float32)
    normalized = (arr / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0).astype(np.float32)


def compute_image_quality_metrics(image: Image.Image) -> ImageQualityMetrics:
    """Menganalisis kecerahan, kontras, dan ketajaman tepi gambar untuk mendeteksi foto buram/gelap (AT-06)."""
    gray = np.asarray(image.convert("L").resize(INPUT_IMAGE_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_strength = float((np.mean(gx) + np.mean(gy)) / 2.0)

    if contrast_std < 8.0 or brightness_mean < 15.0 or brightness_mean > 248.0:
        return ImageQualityMetrics(
            brightness_mean=round(brightness_mean, 2),
            contrast_std=round(contrast_std, 2),
            edge_strength=round(edge_strength, 2),
            is_low_quality=True,
            quality_note=("Foto tampak terlalu gelap, terlalu terang, atau kurang memiliki kontras objek yang jelas."),
        )

    return ImageQualityMetrics(
        brightness_mean=round(brightness_mean, 2),
        contrast_std=round(contrast_std, 2),
        edge_strength=round(edge_strength, 2),
        is_low_quality=False,
    )
