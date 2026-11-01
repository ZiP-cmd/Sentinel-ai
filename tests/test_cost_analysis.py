
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cost_analysis import estimate_cost


def test_total_cost():
    result = estimate_cost(
        alerts=60,
        missed_fraud=15,
        cost_per_review=5,
        loss_per_missed_fraud=200,
    )

    assert result["review_cost"] == 300
    assert result["missed_fraud_cost"] == 3000
    assert result["total_cost"] == 3300


def test_zero_costs():
    result = estimate_cost(10, 5, 0, 0)

    assert result["total_cost"] == 0


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        estimate_cost(10, 5, -1, 200)


def test_fractional_transaction_count_rejected():
    with pytest.raises(ValueError):
        estimate_cost(2.5, 5, 5, 200)
