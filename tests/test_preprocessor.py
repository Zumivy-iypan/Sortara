import numpy as np
from PIL import Image

from sortara.config import INPUT_IMAGE_SIZE
from sortara.preprocessor import compute_image_quality_metrics, preprocess_for_mobilenetv2


def test_preprocess_for_mobilenetv2_returns_expected_shape_and_dtype_fr06():
    """FR-06: Gambar diubah ke (1, 224, 224, 3) float32 dengan skala [-1.0, 1.0] sesuai MobileNetV2."""
    img = Image.new("RGB", (640, 480), color=(255, 0, 127))
    tensor = preprocess_for_mobilenetv2(img)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1], 3)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= -1.0001
    assert float(tensor.max()) <= 1.0001
    # Nilai 255 pada kanal R harus dipetakan mendekati +1.0, dan nilai 0 pada kanal G mendekati -1.0
    assert np.isclose(tensor[0, 0, 0, 0], 1.0, atol=1e-2)
    assert np.isclose(tensor[0, 0, 0, 1], -1.0, atol=1e-2)


def test_compute_image_quality_metrics_identifies_low_contrast_image_at06():
    """AT-06: Mendeteksi gambar seragam/buram ekstrem yang tidak memiliki kontras objek."""
    flat_img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    metrics = compute_image_quality_metrics(flat_img)
    assert metrics.is_low_quality is True
    assert metrics.contrast_std == 0.0
