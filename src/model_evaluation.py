
import numpy as np


def evaluate_predictions(y_true, scores, threshold=0.50):
    """Evaluate binary fraud predictions at a threshold."""

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y_true) != len(scores) or len(y_true) == 0:
        raise ValueError("Inputs must have equal, nonzero lengths.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")

    if not np.isfinite(scores).all() or (
        (scores < 0) | (scores > 1)
    ).any():
        raise ValueError("Scores must be finite and between 0 and 1.")

    predictions = scores >= threshold

    tp = int(np.sum((y_true == 1) & predictions))
    fp = int(np.sum((y_true == 0) & predictions))
    fn = int(np.sum((y_true == 1) & ~predictions))
    tn = int(np.sum((y_true == 0) & ~predictions))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "alerts": tp + fp,
    }
