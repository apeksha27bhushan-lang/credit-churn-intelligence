"""
evaluation.py
-------------
Calculates and reports all model evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Compute classification metrics.
    y_pred should use the optimised threshold (not raw 0.5).
    """
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_arr, precision_arr)

    return {
        "Accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "ROC-AUC":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "PR-AUC":    round(float(pr_auc), 4),
    }


def compute_confusion(y_true, y_pred) -> np.ndarray:
    """Return the confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def compare_models(results: dict) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a dict of {model_name: metrics_dict}.
    """
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("ROC-AUC", ascending=False)


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 72)
    print(comparison_df.to_string())
    print("=" * 72)


def print_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    print(f"\nConfusion Matrix — {model_name}")
    print(f"               Predicted 0   Predicted 1")
    print(f"  Actual 0     {cm[0,0]:>10,}   {cm[0,1]:>10,}")
    print(f"  Actual 1     {cm[1,0]:>10,}   {cm[1,1]:>10,}")
