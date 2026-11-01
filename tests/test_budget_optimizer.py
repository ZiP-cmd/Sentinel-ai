
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from budget_optimizer import compare_budget_costs


@pytest.fixture
def sample_data():
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.1]
    return labels, scores


def test_high_fraud_loss_favors_more_reviews(sample_data):
    labels, scores = sample_data

    result = compare_budget_costs(
        labels,
        scores,
        budgets=[1, 3],
        cost_per_review=5,
        loss_per_missed_fraud=100,
    )

    assert result["recommended_budget"] == 3


def test_high_review_cost_favors_fewer_reviews(sample_data):
    labels, scores = sample_data

    result = compare_budget_costs(
        labels,
        scores,
        budgets=[1, 3],
        cost_per_review=100,
        loss_per_missed_fraud=5,
    )

    assert result["recommended_budget"] == 1


def test_equal_cost_prefers_smaller_budget(sample_data):
    labels, scores = sample_data

    result = compare_budget_costs(
        labels,
        scores,
        budgets=[1, 3],
        cost_per_review=0,
        loss_per_missed_fraud=0,
    )

    assert result["recommended_budget"] == 1


def test_empty_budgets_rejected(sample_data):
    labels, scores = sample_data

    with pytest.raises(ValueError, match="budget"):
        compare_budget_costs(
            labels,
            scores,
            budgets=[],
            cost_per_review=5,
            loss_per_missed_fraud=100,
        )
