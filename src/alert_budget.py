
import numpy as np


def evaluate_alert_budget(y_true, scores, budget):
    """Evaluate the highest-scoring transactions."""

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y_true) == 0 or len(y_true) != len(scores):
        raise ValueError("Inputs must have equal, nonzero lengths.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be binary.")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")

    if not isinstance(budget, (int, np.integer)):
        raise ValueError("Budget must be an integer.")

    if not 1 <= budget <= len(scores):
        raise ValueError("Budget is outside the valid range.")

    # Sort from highest score to lowest score.
    ranked_indices = np.argsort(
        -scores, kind="stable"
    )

    # Select only the highest-scoring transactions.
    selected_indices = ranked_indices[:budget]

    predictions = np.zeros(len(scores), dtype=int)
    predictions[selected_indices] = 1

    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "budget": int(budget),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "selected_indices": selected_indices,
    }
