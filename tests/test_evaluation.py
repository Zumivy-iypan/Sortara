from matplotlib.figure import Figure

from sortara.config import CLASS_LABELS
from sortara.evaluation import compute_evaluation_metrics, plot_confusion_matrix_figure


def test_compute_evaluation_metrics_and_confusion_matrix_at09():
    """AT-09: Menghitung accuracy, precision, recall per kelas, dan confusion matrix."""
    y_true = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 4, 0, 1, 2, 3, 4, 5]

    report = compute_evaluation_metrics(y_true, y_pred, class_names=list(CLASS_LABELS))
    assert 0.80 <= report.overall_accuracy <= 1.0
    assert report.confusion_matrix.shape == (6, 6)
    assert set(report.per_class_metrics.keys()) == set(CLASS_LABELS)
    for _cls_name, m in report.per_class_metrics.items():
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1_score"] <= 1.0

    fig = plot_confusion_matrix_figure(report.confusion_matrix, class_names=list(CLASS_LABELS))
    assert isinstance(fig, Figure)
