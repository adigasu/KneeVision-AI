import numpy as np
import pytest
from src.metrics.auc_metrics import TARGET_COLUMNS, compute_competition_metric


def test_perfect_prediction():
    y_true = np.array([
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    ], dtype=np.float32)

    y_pred = np.array([
        [0.9, 0.1, 0.8, 0.2, 0.1, 0.95, 0.05, 0.85, 0.15, 0.1, 0.99, 0.01],
        [0.1, 0.9, 0.2, 0.8, 0.9, 0.05, 0.95, 0.15, 0.85, 0.9, 0.01, 0.99],
    ], dtype=np.float32)

    macro_auc, per_class = compute_competition_metric(y_true, y_pred)
    assert pytest.approx(macro_auc, 1e-4) == 1.0
    for col in TARGET_COLUMNS:
        assert pytest.approx(per_class[col], 1e-4) == 1.0


def test_nan_handling():
    # Only 2 targets labeled for row 1, other targets NaN
    y_true = np.array([
        [1, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [0, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ], dtype=np.float32)

    y_pred = np.array([
        [0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.2, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ], dtype=np.float32)

    macro_auc, per_class = compute_competition_metric(y_true, y_pred)
    assert pytest.approx(macro_auc, 1e-4) == 1.0
    assert pytest.approx(per_class["ACL"], 1e-4) == 1.0
    assert pytest.approx(per_class["MCL"], 1e-4) == 1.0
    assert np.isnan(per_class["Medial Meniscus"])
