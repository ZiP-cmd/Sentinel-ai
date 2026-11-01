
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from investigation_evidence import build_investigation_evidence
from claim_validator import validate_claims


@pytest.fixture
def evidence():
    return build_investigation_evidence(
        {"Amount": 75.00},
        {
            "fraud_score": 0.91,
            "threshold": 0.50,
            "flagged": True,
        },
    )


def test_verified_score_approved(evidence):
    result = validate_claims(
        evidence,
        [{"field": "model_score", "value": 0.91}],
    )

    assert result["approved"] is True


def test_verified_decision_approved(evidence):
    result = validate_claims(
        evidence,
        [{"field": "review_required", "value": True}],
    )

    assert result["approved"] is True


def test_invented_location_rejected(evidence):
    result = validate_claims(
        evidence,
        [{"field": "country", "value": "France"}],
    )

    assert result["approved"] is False


def test_fraud_accusation_rejected(evidence):
    result = validate_claims(
        evidence,
        [{"field": "confirmed_fraud", "value": True}],
    )

    assert result["approved"] is False


def test_incorrect_score_rejected(evidence):
    result = validate_claims(
        evidence,
        [{"field": "model_score", "value": 0.99}],
    )

    assert result["approved"] is False


def test_boolean_cannot_replace_numeric_score(evidence):
    result = validate_claims(
        evidence,
        [{"field": "model_score", "value": True}],
    )

    assert result["approved"] is False
