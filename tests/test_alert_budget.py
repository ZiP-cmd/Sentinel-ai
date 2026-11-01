
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alert_budget import evaluate_alert_budget


def test_budget_selects_highest_scores():
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.4, 0.1]

    result = evaluate_alert_budget(
        labels, scores, budget=2
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1


def test_budget_limits_alerts():
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.4, 0.1]

    result = evaluate_alert_budget(
        labels, scores, budget=2
    )

    assert result["tp"] + result["fp"] == 2


def test_larger_budget_catches_at_least_as_much_fraud():
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.4, 0.1]

    small = evaluate_alert_budget(labels, scores, 1)
    large = evaluate_alert_budget(labels, scores, 3)

    assert large["tp"] >= small["tp"]


@pytest.mark.parametrize("budget", [0, 5, -1])
def test_invalid_budget(budget):
    with pytest.raises(ValueError, match="Budget"):
        evaluate_alert_budget(
            [1, 0, 1, 0],
            [0.9, 0.8, 0.4, 0.1],
            budget,
        )
