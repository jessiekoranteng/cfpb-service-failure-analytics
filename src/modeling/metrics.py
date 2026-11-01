"""Evaluation metrics and plots for the risk models."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.modeling import config as C


def top_share_threshold(scores, share: float) -> float:
    """Score cut that flags the top `share` of complaints (review-capacity rule)."""
    return float(np.quantile(scores, 1 - share))


def f1_threshold(y, scores) -> float:
    grid = np.unique(np.quantile(scores, np.linspace(0.5, 0.999, 200)))
    return float(grid[int(np.argmax([f1_score(y, scores >= t, zero_division=0) for t in grid]))])


def evaluate(y, scores, threshold: float, capacity: float = C.REVIEW_CAPACITY) -> dict:
    y = np.asarray(y)
    scores = np.asarray(scores)
    pred = scores >= threshold
    cap_t = top_share_threshold(scores, capacity)
    flagged = scores >= cap_t
    prev = float(y.mean())
    pr = float(average_precision_score(y, scores))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y), "prevalence": prev,
        "roc_auc": float(roc_auc_score(y, scores)), "pr_auc": pr, "pr_auc_lift": pr / prev if prev else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(scores, 0, 1))),
        "threshold": float(threshold), "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "recall_at_capacity": float(recall_score(y, flagged, zero_division=0)), "precision_at_capacity": float(precision_score(y, flagged, zero_division=0)),
        "capacity_threshold": cap_t,
    }
