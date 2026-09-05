"""
RSNA Knee Abnormality Detection - Evaluation Metrics
Exact Macro ROC-AUC evaluation across all 12 target classes.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# The 12 official competition target abnormality classes
TARGET_COLUMNS: List[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def compute_competition_metric(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    target_columns: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes the Kaggle competition evaluation metric: Macro-Averaged AUC-ROC across 12 targets.

    Args:
        y_true: Ground truth binary labels of shape (N, 12). Can contain NaNs (unlabeled instances).
        y_pred: Predicted probability scores of shape (N, 12) between [0.0, 1.0].
        target_columns: Optional list of target column names. Defaults to TARGET_COLUMNS.

    Returns:
        macro_auc: Average AUC across all evaluable target classes (float).
        per_class_auc: Dictionary mapping class name to its individual AUC score.
    """
    if target_columns is None:
        target_columns = TARGET_COLUMNS

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true[target_columns].to_numpy()
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred[target_columns].to_numpy()

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    assert y_true.shape == y_pred.shape, f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
    num_classes = y_true.shape[1]
    assert num_classes == len(target_columns), f"Expected {len(target_columns)} columns, got {num_classes}"

    per_class_auc: Dict[str, float] = {}
    valid_aucs: List[float] = []

    for i, col_name in enumerate(target_columns):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]

        # Filter out NaN / missing values (for partially labeled training sets)
        valid_mask = ~np.isnan(true_col)
        valid_true = true_col[valid_mask]
        valid_pred = pred_col[valid_mask]

        # AUC is only well-defined if both positive and negative samples exist in ground truth
        unique_classes = np.unique(valid_true)
        if len(unique_classes) < 2:
            per_class_auc[col_name] = np.nan
            continue

        try:
            score = float(roc_auc_score(valid_true, valid_pred))
            per_class_auc[col_name] = score
            valid_aucs.append(score)
        except ValueError:
            per_class_auc[col_name] = np.nan

    macro_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else 0.5
    return macro_auc, per_class_auc


def compute_macro_auc(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    target_columns: Optional[List[str]] = None,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Helper returning a structured dict with 'macro_auc' and 'per_class_auc'."""
    macro_auc, per_class_auc = compute_competition_metric(y_true, y_pred, target_columns)
    return {
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
    }
