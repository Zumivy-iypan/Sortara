from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from sortara.config import CLASS_LABELS

matplotlib.use("Agg")


@dataclass(frozen=True)
class EvaluationReport:
    overall_accuracy: float
    per_class_metrics: dict[str, dict[str, float]]
    confusion_matrix: np.ndarray
    class_names: tuple[str, ...]


def compute_evaluation_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] = CLASS_LABELS,
) -> EvaluationReport:
    """Menghitung accuracy, precision, recall, f1-score per kelas, dan confusion matrix (Bagian 9.4 & AT-09)."""
    labels_idx = list(range(len(class_names)))
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_idx,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    per_class: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1_score": round(float(f1[idx]), 4),
            "support": float(support[idx]),
        }

    return EvaluationReport(
        overall_accuracy=round(acc, 4),
        per_class_metrics=per_class,
        confusion_matrix=cm,
        class_names=tuple(class_names),
    )


def plot_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: Sequence[str] = CLASS_LABELS,
    title: str = "Confusion Matrix — Evaluasi Model SORTARA (Test Set)",
) -> Figure:
    """Membuat visualisasi Confusion Matrix menggunakan Object-Oriented API Matplotlib."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)
    cmap = matplotlib.colormaps["Greens"]
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, labels=list(class_names), rotation=30, ha="right")
    ax.set_yticks(ticks, labels=list(class_names))
    ax.set_xlabel("Prediksi Model")
    ax.set_ylabel("Label Sebenarnya (Ground Truth)")
    ax.set_title(title, pad=12)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "#0f172a",
                fontweight="bold",
            )

    fig.tight_layout()
    return fig
