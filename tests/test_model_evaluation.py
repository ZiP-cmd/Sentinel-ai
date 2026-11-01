
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_evaluation import evaluate_predictions


def test_confusion_matrix_counts():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.3, 0.8, 0.1]

    result = evaluate_predictions(
        labels, scores, threshold=0.5
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1


def test_precision_and_recall():
    labels = [1, 1, 1, 0]
    scores = [0.9, 0.8, 0.2, 0.7]

    result = evaluate_predictions(
        labels, scores, threshold=0.5
    )

    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)


def test_lower_threshold_catches_more_fraud():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.3, 0.4, 0.1]

    high = evaluate_predictions(labels, scores, 0.5)
    low = evaluate_predictions(labels, scores, 0.2)

    assert low["tp"] >= high["tp"]
    assert low["fp"] >= high["fp"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="Threshold"):
        evaluate_predictions([0, 1], [0.2, 0.8], threshold)
