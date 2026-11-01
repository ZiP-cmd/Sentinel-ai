
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from investigation_evidence import (
    build_investigation_evidence,
)


def test_valid_evidence():
    result = build_investigation_evidence(
        {"Amount": 50.00},
        {
            "fraud_score": 0.2941,
            "threshold": 0.50,
            "flagged": False,
        },
    )

    assert result["transaction_amount"] == 50.00
    assert result["review_required"] is False
    assert len(result["limitations"]) > 0


def test_missing_amount_rejected():
    with pytest.raises(ValueError):
        build_investigation_evidence(
            {},
            {
                "fraud_score": 0.80,
                "threshold": 0.50,
                "flagged": True,
            },
        )


def test_inconsistent_decision_rejected():
    with pytest.raises(ValueError):
        build_investigation_evidence(
            {"Amount": 100.00},
            {
                "fraud_score": 0.90,
                "threshold": 0.50,
                "flagged": False,
            },
        )


def test_invalid_score_rejected():
    with pytest.raises(ValueError):
        build_investigation_evidence(
            {"Amount": 100.00},
            {
                "fraud_score": 1.5,
                "threshold": 0.50,
                "flagged": True,
            },
        )

import math


def test_nan_score_rejected():
    with pytest.raises(ValueError):
        build_investigation_evidence(
            {"Amount": 50.00},
            {
                "fraud_score": float("nan"),
                "threshold": 0.50,
                "flagged": False,
            },
        )


def test_infinite_threshold_rejected():
    with pytest.raises(ValueError):
        build_investigation_evidence(
            {"Amount": 50.00},
            {
                "fraud_score": 0.80,
                "threshold": float("inf"),
                "flagged": False,
            },
        )


def test_nan_amount_rejected():
    with pytest.raises(ValueError):
        build_investigation_evidence(
            {"Amount": float("nan")},
            {
                "fraud_score": 0.80,
                "threshold": 0.50,
                "flagged": True,
            },
        )
def test_boolean_amount_rejected():
    with pytest.raises(ValueError):
        build_investigation_evidence(
            {"Amount": True},
            {
                "fraud_score": 0.80,
                "threshold": 0.50,
                "flagged": True,
            },
        )
