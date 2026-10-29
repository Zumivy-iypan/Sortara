from pathlib import Path

from PIL import Image

from sortara.config import CLASS_LABELS, DEFAULT_MODEL_PATH
from sortara.predictor import WastePredictor


def test_predictor_loads_real_keras_mobilenetv2_model_when_available():
    """Memastikan model MobileNetV2 (.keras) hasil training dapat dimuat dan melakukan inferensi nyata."""
    if not DEFAULT_MODEL_PATH.exists():
        return
    predictor = WastePredictor(model_path=DEFAULT_MODEL_PATH, strict_model_loading=True)
    assert predictor.model_mode == "keras_mobilenetv2"

    img = Image.new("RGB", (224, 224), color=(120, 180, 90))
    res = predictor.predict(img)
    assert res.is_success is True
    assert res.model_mode == "keras_mobilenetv2"
    assert res.predicted_category in CLASS_LABELS
    assert abs(sum(res.class_probabilities.values()) - 1.0) < 1e-3


def test_predictor_fallback_mode_returns_valid_six_class_probabilities_fr07_fr08():
    """FR-07 & FR-08: Menghasilkan 1 kategori dari 6 kelas beserta skor keyakinan dan distribusi probabilitas."""
    predictor = WastePredictor(model_path=Path("models/non_existent_model.keras"))
    img = Image.new("RGB", (300, 300), color=(45, 175, 65))
    result = predictor.predict(img)

    assert result.is_success is True
    assert result.predicted_category in CLASS_LABELS
    assert 0.0 <= result.confidence_score <= 1.0
    assert set(result.class_probabilities.keys()) == set(CLASS_LABELS)
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-4


def test_predictor_handles_corrupt_model_file_gracefully_fr13_at08(tmp_path: Path):
    """FR-13 & AT-08: Ketika file model rusak/tidak dapat dimuat, kembalikan pesan kegagalan yang jelas."""
    bad_model_file = tmp_path / "corrupt_model.keras"
    bad_model_file.write_bytes(b"corrupted_keras_archive")

    predictor = WastePredictor(model_path=bad_model_file, strict_model_loading=True)
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    result = predictor.predict(img)

    assert result.is_success is False
    assert result.error_message is not None
    assert "model" in result.error_message.lower()
