"""
VIGIL AI – Evaluation Service
Computes confusion matrix, accuracy, precision, recall, F1, and ROC-AUC
from predictions stored in the database.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics.

    Args:
        y_true  : ground-truth labels (0=NonViolent, 1=Violent)
        y_pred  : predicted labels
        y_scores: predicted violence probabilities (for ROC-AUC)

    Returns:
        dict with accuracy, precision, recall, f1, roc_auc, confusion_matrix
    """
    if len(y_true) == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "roc_auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "support": 0,
        }

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    roc_auc = None
    if y_scores is not None and len(set(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except Exception:
            pass

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
        "support": len(y_true),
    }


def label_to_int(label: str) -> int:
    """Convert string label to int (Violent=1, Non-Violent=0)."""
    return 1 if label.strip().lower() == "violent" else 0


def predictions_to_arrays(predictions):
    """
    Convert a list of Prediction ORM objects (with ground_truth) to
    y_true, y_pred, y_scores arrays for metric computation.
    Only includes records that have a ground_truth label set.
    """
    y_true, y_pred, y_scores = [], [], []
    for p in predictions:
        if p.ground_truth is None:
            continue
        y_true.append(label_to_int(p.ground_truth))
        y_pred.append(label_to_int(p.prediction))
        y_scores.append(p.violence_prob)
    return y_true, y_pred, y_scores
