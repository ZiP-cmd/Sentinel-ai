
import sys
from pathlib import Path
from datetime import datetime

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from investigation_evidence import build_investigation_evidence
from investigation_audit import create_investigation_audit


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


def test_approved_claims_create_audit(evidence):
    claims = [
        {"field": "model_score", "value": 0.91},
        {"field": "review_required", "value": True},
    ]

    result = create_investigation_audit(evidence, claims)

    assert result["investigator_view"]["status"] == "approved"
    assert result["audit_record"]["approved_count"] == 2
    assert result["audit_record"]["rejected_count"] == 0


def test_rejected_claim_preserved_in_audit(evidence):
    claims = [
        {"field": "model_score", "value": 0.91},
        {"field": "review_required", "value": True},
        {"field": "confirmed_fraud", "value": True},
    ]

    result = create_investigation_audit(evidence, claims)

    audit = result["audit_record"]

    assert audit["validation_status"] == "filtered"
    assert audit["rejected_count"] == 1
    assert audit["rejected_claims"][0]["claim"]["field"] == (
        "confirmed_fraud"
    )


def test_investigator_view_excludes_rejected_claim(evidence):
    claims = [
        {"field": "model_score", "value": 0.91},
        {"field": "review_required", "value": True},
        {"field": "country", "value": "France"},
    ]

    result = create_investigation_audit(evidence, claims)

    public_view = result["investigator_view"]

    assert public_view["status"] == "filtered"
    assert public_view["excluded_claim_count"] == 1
    assert "France" not in str(public_view)
    assert "rejected_claims" not in public_view


def test_audit_has_event_id_and_timestamp(evidence):
    claims = [
        {"field": "model_score", "value": 0.91},
        {"field": "review_required", "value": True},
    ]

    result = create_investigation_audit(evidence, claims)
    audit = result["audit_record"]

    assert audit["event_id"]
    assert datetime.fromisoformat(audit["timestamp"]).tzinfo is not None


def test_missing_claim_uses_fallback_and_audits(evidence):
    claims = [
        {"field": "model_score", "value": 0.91},
    ]

    result = create_investigation_audit(evidence, claims)

    assert result["investigator_view"]["status"] == "fallback"
    assert result["audit_record"]["validation_status"] == "fallback"
    assert result["audit_record"]["approved_count"] == 1
