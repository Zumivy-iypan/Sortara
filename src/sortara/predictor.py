import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sortara.config import CLASS_LABELS, DEFAULT_LABELS_PATH, DEFAULT_MODEL_PATH
from sortara.preprocessor import compute_image_quality_metrics, preprocess_for_mobilenetv2


@dataclass(frozen=True)
class PredictionResult:
    is_success: bool
    predicted_category: str | None
    confidence_score: float
    class_probabilities: dict[str, float]
    model_mode: str  # "keras_mobilenetv2" | "prototype_fallback"
    error_message: str | None = None
    quality_warning: str | None = None


def load_class_labels(labels_path: Path = DEFAULT_LABELS_PATH) -> tuple[str, ...]:
    """Memuat pemetaan indeks label dari labels.json atau kembali ke CLASS_LABELS default."""
    if labels_path.exists():
        try:
            raw = json.loads(labels_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and len(raw) == len(CLASS_LABELS):
                ordered = [raw[str(i)] for i in range(len(CLASS_LABELS))]
                return tuple(ordered)
        except Exception:
            pass
    return CLASS_LABELS


class WastePredictor:
    """Engine inferensi klasifikasi sampah 6 kelas dengan dukungan Keras (.keras) & Fallback Mode."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        labels_path: Path = DEFAULT_LABELS_PATH,
        strict_model_loading: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels = load_class_labels(labels_path)
        self.strict_model_loading = strict_model_loading
        self._keras_model: Any | None = None
        self._load_error: str | None = None
        self.model_mode: str = "prototype_fallback"
        self._initialize_model()

    def _initialize_model(self) -> None:
        if not self.model_path.exists():
            if self.strict_model_loading:
                self._load_error = (
                    f"File model AI tidak ditemukan di '{self.model_path.name}'. "
                    "Pastikan file model hasil training tersedia di direktori models/."
                )
            return

        try:
            import tensorflow as tf  # type: ignore[import-untyped]

            self._keras_model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model_mode = "keras_mobilenetv2"
        except Exception as exc:
            self._load_error = (
                f"Model AI gagal dimuat dari '{self.model_path.name}' ({type(exc).__name__}). "
                "Silakan periksa integritas file model `.keras`."
            )

    def predict(
        self,
        image: Image.Image,
        filename_hint: str | None = None,
    ) -> PredictionResult:
        """Menjalankan preprocessing dan memprediksi salah satu dari enam kategori (FR-06, FR-07, FR-08, FR-13)."""
        if self._load_error is not None:
            return PredictionResult(
                is_success=False,
                predicted_category=None,
                confidence_score=0.0,
                class_probabilities={},
                model_mode="error",
                error_message=self._load_error,
            )

        try:
            tensor = preprocess_for_mobilenetv2(image)
            quality = compute_image_quality_metrics(image)

            if self._keras_model is not None:
                raw_preds = self._keras_model.predict(tensor, verbose=0)
                probs = np.asarray(raw_preds[0], dtype=np.float64)
                probs = np.clip(probs, 1e-8, None)
                probs = probs / np.sum(probs)
            else:
                probs = self._heuristic_feature_probabilities(
                    tensor=tensor,
                    is_low_quality=quality.is_low_quality,
                    filename_hint=filename_hint,
                )

            prob_map = {label: round(float(probs[idx]), 4) for idx, label in enumerate(self.labels)}
            best_idx = int(np.argmax(probs))
            best_label = self.labels[best_idx]
            best_score = round(float(probs[best_idx]), 4)

            return PredictionResult(
                is_success=True,
                predicted_category=best_label,
                confidence_score=best_score,
                class_probabilities=prob_map,
                model_mode=self.model_mode,
                quality_warning=quality.quality_note,
            )
        except Exception as exc:
            return PredictionResult(
                is_success=False,
                predicted_category=None,
                confidence_score=0.0,
                class_probabilities={},
                model_mode=self.model_mode,
                error_message=(
                    f"Terjadi kesalahan saat memproses prediksi gambar ({type(exc).__name__}). "
                    "Silakan coba unggah foto lain."
                ),
            )

    def _heuristic_feature_probabilities(
        self,
        tensor: np.ndarray,
        is_low_quality: bool,
        filename_hint: str | None = None,
    ) -> np.ndarray:
        """Menghasilkan distribusi probabilitas deterministik berbasis fitur warna/tekstur & petunjuk nama file."""
        if is_low_quality:
            # Gambar buram/gelap/polos ekstrem menghasilkan distribusi datar (low confidence < 0.60)
            uniform = np.array([0.21, 0.18, 0.16, 0.16, 0.15, 0.14], dtype=np.float64)
            return uniform / np.sum(uniform)

        # Kembalikan rentang [-1, 1] ke [0, 1] untuk ekstraksi ciri warna & tekstur
        rgb = (tensor[0] + 1.0) / 2.0
        r_mean = float(np.mean(rgb[:, :, 0]))
        g_mean = float(np.mean(rgb[:, :, 1]))
        b_mean = float(np.mean(rgb[:, :, 2]))
        color_spread = float(np.std([r_mean, g_mean, b_mean]))
        spatial_std = float(np.std(rgb))

        # Urutan: ("Kaca", "Kertas", "Logam", "Organik", "Plastik", "Residu")
        logits = np.array(
            [
                1.2 * b_mean + 0.8 * (1.0 - color_spread),  # Kaca
                1.3 * ((r_mean + g_mean + b_mean) / 3.0) - color_spread,  # Kertas
                1.4 * (1.0 - color_spread) + 0.6 * spatial_std,  # Logam
                1.8 * g_mean + 0.7 * r_mean - 0.9 * b_mean,  # Organik
                1.5 * b_mean + 1.2 * color_spread + 0.4 * r_mean,  # Plastik
                1.0 * spatial_std + 0.5 * (1.0 - g_mean),  # Residu
            ],
            dtype=np.float64,
        )

        if filename_hint:
            lowered = filename_hint.lower()
            keyword_boosts: dict[int, tuple[str, ...]] = {
                0: ("kaca", "glass", "botol_kaca", "jar", "stoples"),
                1: ("kertas", "paper", "cardboard", "kardus", "koran"),
                2: ("logam", "metal", "kaleng", "can", "aluminium"),
                3: ("organik", "organic", "daun", "buah", "sayur", "kompos", "bio"),
                4: ("plastik", "plastic", "pet", "botol", "bottle", "cup"),
                5: ("residu", "trash", "residual", "diaper", "popok", "tisu"),
            }
            for idx, keywords in keyword_boosts.items():
                if any(k in lowered for k in keywords):
                    logits[idx] += 2.2

        scaled = logits * 2.4
        exp_scores = np.exp(scaled - np.max(scaled))
        return exp_scores / np.sum(exp_scores)
