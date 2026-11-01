
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from investigation_evidence import build_investigation_evidence
from safe_response_builder import build_safe_response


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


def test_verified_claims_approved(evidence):
    claims = [
        {"field": "model_score", "value": 0.91},
        {"field": "review_required", "value": True},
    ]

    result = build_safe_response(evidence, claims)

    assert result["status"] == "approved"
    assert "0.9100" in result["text"]


def test_unsupported_claim_removed(evidence):
    claims = [
        {"field": "model_score", "value": 0.91},
        {"field": "review_required", "value": True},
        {"field": "confirmed_fraud", "value": True},
    ]

    result = build_safe_response(evidence, claims)

    assert result["status"] == "filtered"
    assert result["rejected_count"] == 1
    assert "does not establish" in result["text"]
    assert "confirmed fraud" not in result["text"].lower()


def test_missing_required_claim_uses_fallback(evidence):
    claims = [
        {"field": "model_score", "value": 0.91},
    ]

    result = build_safe_response(evidence, claims)

    assert result["status"] == "fallback"
    assert "SENTINEL AI INVESTIGATION SUMMARY" in result["text"]


def test_invented_location_not_displayed(evidence):
    claims = [
        {"field": "model_score", "value": 0.91},
        {"field": "review_required", "value": True},
        {"field": "country", "value": "France"},
    ]

    result = build_safe_response(evidence, claims)

    assert result["status"] == "filtered"
    assert "France" not in result["text"]


def test_invalid_claim_format_rejected(evidence):
    with pytest.raises(ValueError, match="Claims must be a list"):
        build_safe_response(evidence, "Fraud confirmed")
