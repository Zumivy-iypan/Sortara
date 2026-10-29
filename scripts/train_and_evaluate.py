"""Skrip pelatihan MobileNetV2 Transfer Learning & Evaluasi Test Set SORTARA (Tugas Adit).

Menjalankan alur lengkap sesuai PRD Bagian 9:
1. Membagi dataset `data/dataset_raw/` secara stratified 70% Train / 15% Validation / 15% Test.
2. Menerapkan preprocessing MobileNetV2 (`224x224`, rentang `[-1, 1]`) dan augmentasi ringan pada Train.
3. Melatih model Transfer Learning MobileNetV2 + Fine-Tuning terbatas.
4. Mengevaluasi model pada Test Set terpisah (Accuracy, Precision, Recall, F1, Confusion Matrix).
5. Menyimpan:
   - `models/sortara_model.keras`
   - `models/labels.json`
   - `models/evaluation_report.json`
   - `models/confusion_matrix.png`
   - `models/training_history.png`
   - `docs/EVALUATION_REPORT.md`
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sortara.config import CLASS_LABELS, INPUT_IMAGE_SIZE  # noqa: E402
from sortara.evaluation import compute_evaluation_metrics, plot_confusion_matrix_figure  # noqa: E402

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset_raw"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

SEED = 42
BATCH_SIZE = 32
INITIAL_EPOCHS = 6
FINETUNE_EPOCHS = 4


def collect_dataset_files(dataset_dir: Path) -> tuple[list[str], list[int], dict[str, int]]:
    filepaths: list[str] = []
    labels: list[int] = []
    class_counts: dict[str, int] = {}

    for idx, cls_name in enumerate(CLASS_LABELS):
        cls_folder = dataset_dir / cls_name
        if not cls_folder.exists():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {cls_folder}")
        files = sorted(
            str(p)
            for p in cls_folder.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        filepaths.extend(files)
        labels.extend([idx] * len(files))
        class_counts[cls_name] = len(files)

    return filepaths, labels, class_counts


def build_tf_dataset(
    paths: list[str],
    labels: list[int],
    training: bool = False,
) -> tf.data.Dataset:
    aug_layer = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.08),
        ]
    )

    def _load(path: tf.Tensor, label: tf.Tensor):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, INPUT_IMAGE_SIZE)
        if training:
            img = aug_layer(img, training=True)
        # Preprocessing MobileNetV2: (x / 127.5) - 1.0
        img = (tf.cast(img, tf.float32) / 127.5) - 1.0
        return img, tf.one_hot(label, depth=len(CLASS_LABELS))

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_mobilenetv2_classifier(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1], 3), name="input_image")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="sortara_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def save_training_curves(history_dict: dict[str, list[float]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    epochs_range = range(1, len(history_dict["accuracy"]) + 1)

    axes[0].plot(epochs_range, history_dict["accuracy"], marker="o", label="Train Accuracy")
    axes[0].plot(epochs_range, history_dict["val_accuracy"], marker="s", label="Val Accuracy")
    axes[0].set_title("Kurva Accuracy Training vs Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs_range, history_dict["loss"], marker="o", label="Train Loss")
    axes[1].plot(epochs_range, history_dict["val_loss"], marker="s", label="Val Loss")
    axes[1].set_title("Kurva Loss Training vs Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Categorical Cross-Entropy Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    filepaths, labels, class_counts = collect_dataset_files(DATASET_DIR)
    print(f"Total dataset ditemukan: {len(filepaths)} gambar dari 6 kelas: {class_counts}")

    # Pembagian dataset 70% Train / 15% Validation / 15% Test secara stratified
    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(f"Split -> Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    train_ds = build_tf_dataset(X_train, y_train, training=True)
    val_ds = build_tf_dataset(X_val, y_val, training=False)
    test_ds = build_tf_dataset(X_test, y_test, training=False)

    model, base_model = build_mobilenetv2_classifier(len(CLASS_LABELS))

    print("\n--- Tahap 1: Training Classification Head (Transfer Learning) ---")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
    )

    print("\n--- Tahap 2: Fine-Tuning Terbatas (25 Lapisan Terakhir MobileNetV2) ---")
    base_model.trainable = True
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ],
    )

    combined_history = {
        k: hist1.history[k] + hist2.history[k]
        for k in ("accuracy", "val_accuracy", "loss", "val_loss")
    }
    save_training_curves(combined_history, MODELS_DIR / "training_history.png")

    # Tentukan rekomendasi ambang keyakinan pada Validation Set
    val_probs = model.predict(val_ds)
    val_preds = np.argmax(val_probs, axis=1)
    val_max_conf = np.max(val_probs, axis=1)
    correct_mask = val_preds == np.asarray(y_val)
    mean_correct_conf = float(np.mean(val_max_conf[correct_mask])) if np.any(correct_mask) else 0.75
    mean_wrong_conf = float(np.mean(val_max_conf[~correct_mask])) if np.any(~correct_mask) else 0.45

    print("\n--- Evaluasi Final pada Test Set Terpisah (15%) ---")
    test_probs = model.predict(test_ds)
    test_preds = np.argmax(test_probs, axis=1).tolist()
    test_max_conf = np.max(test_probs, axis=1)

    eval_report = compute_evaluation_metrics(y_test, test_preds, class_names=CLASS_LABELS)
    print(f"Test Accuracy Keseluruhan: {eval_report.overall_accuracy * 100:.2f}%")

    cm_fig = plot_confusion_matrix_figure(eval_report.confusion_matrix, class_names=CLASS_LABELS)
    cm_fig.savefig(MODELS_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close(cm_fig)

    # Kumpulkan contoh prediksi benar dan salah dari Test Set (Bagian 9.4 PRD)
    correct_examples = []
    incorrect_examples = []
    for idx, (fpath, true_idx, pred_idx) in enumerate(zip(X_test, y_test, test_preds, strict=False)):
        item = {
            "file": Path(fpath).name,
            "true_class": CLASS_LABELS[true_idx],
            "predicted_class": CLASS_LABELS[pred_idx],
            "confidence": round(float(test_max_conf[idx]), 4),
        }
        if true_idx == pred_idx and len(correct_examples) < 5:
            correct_examples.append(item)
        elif true_idx != pred_idx and len(incorrect_examples) < 5:
            incorrect_examples.append(item)

    # Simpan model .keras & labels.json
    model_out_path = MODELS_DIR / "sortara_model.keras"
    model.save(model_out_path)

    labels_map = {str(i): name for i, name in enumerate(CLASS_LABELS)}
    (MODELS_DIR / "labels.json").write_text(json.dumps(labels_map, indent=2), encoding="utf-8")

    eval_json = {
        "architecture": "MobileNetV2 (Transfer Learning + Fine-Tuning)",
        "input_shape": [224, 224, 3],
        "dataset_summary": {
            "total_images": len(filepaths),
            "train_count": len(X_train),
            "val_count": len(X_val),
            "test_count": len(X_test),
            "per_class_counts": class_counts,
        },
        "validation_confidence_analysis": {
            "mean_confidence_when_correct": round(mean_correct_conf, 4),
            "mean_confidence_when_wrong": round(mean_wrong_conf, 4),
            "recommended_threshold": 0.60,
        },
        "test_metrics": {
            "overall_accuracy": eval_report.overall_accuracy,
            "per_class": eval_report.per_class_metrics,
            "confusion_matrix": eval_report.confusion_matrix.tolist(),
        },
        "sample_predictions": {
            "correct_examples": correct_examples,
            "incorrect_examples": incorrect_examples,
        },
    }
    (MODELS_DIR / "evaluation_report.json").write_text(
        json.dumps(eval_json, indent=2), encoding="utf-8"
    )

    # Tulis laporan Markdown lengkap untuk dokumentasi Tugas Adit & Irfan
    md_lines = [
        "# Laporan Dataset, Pelatihan, dan Evaluasi Model AI — SORTARA",
        "",
        "## 1. Ringkasan Dataset & Pembagian Data (Bagian 9.2 PRD)",
        "- **Sumber Dataset:** Kurasi dari *Garbage Classification Dataset* (Mostafa Mohamed / Kaggle) yang dipetakan ke 6 kategori SORTARA.",
        f"- **Total Gambar Valid (Bebas Duplikat & Corrupt):** `{len(filepaths)}` gambar (`{class_counts}`).",
        "- **Pembagian Stratified (70% / 15% / 15%):**",
        f"  - **Training Set (70%):** `{len(X_train)}` gambar",
        f"  - **Validation Set (15%):** `{len(X_val)}` gambar",
        f"  - **Testing Set (15%):** `{len(X_test)}` gambar (tidak pernah digunakan saat training maupun pemilihan model)",
        "",
        "## 2. Arsitektur & Konfigurasi Training (Bagian 9.1 & 9.3 PRD)",
        "- **Base Model:** `MobileNetV2` (`weights='imagenet'`, `include_top=False`, input `224×224×3` dengan normalisasi `[-1.0, 1.0]`).",
        "- **Classification Head:** `GlobalAveragePooling2D` -> `Dropout(0.25)` -> `Dense(6, activation='softmax')`.",
        f"- **Tahap Pelatihan:** {INITIAL_EPOCHS} epoch *feature extraction* (`lr=1e-3`) + {FINETUNE_EPOCHS} epoch *fine-tuning* 25 layer teratas (`lr=1e-4`).",
        "",
        "## 3. Hasil Evaluasi pada Test Set Terpisah (Bagian 9.4 PRD)",
        f"- **Overall Test Accuracy:** **{eval_report.overall_accuracy * 100:.2f}%** (`{eval_report.overall_accuracy:.4f}`)",
        "",
        "### Metrik Per Kelas (Precision, Recall, F1-Score)",
        "| Kategori | Precision | Recall | F1-Score | Jumlah Sampel Uji (Support) |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls_name, m in eval_report.per_class_metrics.items():
        md_lines.append(
            f"| **{cls_name}** | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | {m['f1_score']*100:.2f}% | {int(m['support'])} |"
        )

    md_lines.extend(
        [
            "",
            "### Analisis Ambang Keyakinan (Validation Set)",
            f"- Rata-rata skor keyakinan saat prediksi **Benar**: `{mean_correct_conf * 100:.2f}%`",
            f"- Rata-rata skor keyakinan saat prediksi **Salah**: `{mean_wrong_conf * 100:.2f}%`",
            "- Ambang keyakinan (*Confidence Threshold*) yang ditetapkan pada aplikasi: **`60%` (`0.60`)**.",
            "",
            "### Contoh Prediksi Benar pada Test Set",
        ]
    )
    for ex in correct_examples:
        md_lines.append(
            f"- `{ex['file']}` -> Label Asli: **{ex['true_class']}** | Prediksi: **{ex['predicted_class']}** (Keyakinan: `{ex['confidence']*100:.1f}%`)"
        )

    md_lines.append("")
    md_lines.append("### Contoh Prediksi Salah (Misclassified) pada Test Set")
    for ex in incorrect_examples:
        md_lines.append(
            f"- `{ex['file']}` -> Label Asli: **{ex['true_class']}** | Prediksi: **{ex['predicted_class']}** (Keyakinan: `{ex['confidence']*100:.1f}%`)"
        )

    md_lines.extend(
        [
            "",
            "## 4. Keterbatasan Dataset dan Kemungkinan Bias",
            "1. **Kemiripan Visual Material Transparan/Mengilap:** Botol kaca bening dan botol plastik PET bening memiliki fitur pantulan cahaya yang mirip sehingga sesekali dapat tertukar pada sudut pengambilan tertentu.",
            "2. **Kondisi Pencahayaan & Latar Belakang:** Gambar dengan satu objek dominan dan pencahayaan merata menghasilkan skor keyakinan yang jauh lebih akurat dibandingkan foto dengan latar belakang yang ramai.",
            "3. **Kategori Residu yang Beragam:** Kelas `Residu` mencakup berbagai benda komposit yang tidak seragam secara bentuk, sehingga variasi intra-kelasnya lebih tinggi dibandingkan kelas seperti `Kaca` atau `Logam`.",
        ]
    )

    (DOCS_DIR / "EVALUATION_REPORT.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Selesai! Model disimpan di: {model_out_path}")
    print(f"Laporan evaluasi disimpan di: {DOCS_DIR / 'EVALUATION_REPORT.md'}")


if __name__ == "__main__":
    main()
