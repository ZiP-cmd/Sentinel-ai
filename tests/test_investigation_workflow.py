
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from investigation_evidence import build_investigation_evidence
from investigation_audit import create_investigation_audit
from audit_sessions import (
    create_demo_session,
    end_demo_session,
    get_authenticated_audit_record,
)


def test_complete_investigation_workflow():
    # Simulated prediction from Sentinel's model.
    prediction = {
        "fraud_score": 0.91,
        "threshold": 0.50,
        "flagged": True,
    }

    transaction = {"Amount": 75.00}

    # Step 1: Validate the evidence.
    evidence = build_investigation_evidence(
        transaction, prediction
    )

    # Step 2: Simulate candidate AI claims.
    claims = [
        {"field": "model_score", "value": 0.91},
        {"field": "review_required", "value": True},
        {"field": "confirmed_fraud", "value": True},
    ]

    # Step 3: Filter unsupported claims and audit.
    result = create_investigation_audit(
        evidence, claims
    )

    investigator_view = result["investigator_view"]

    assert investigator_view["status"] == "filtered"
    assert investigator_view["excluded_claim_count"] == 1
    assert "confirmed fraud" not in (
        investigator_view["summary"].lower()
    )

    # Step 4: Verify a regular investigator is denied.
    investigator_token = create_demo_session("james")

    try:
        with pytest.raises(PermissionError):
            get_authenticated_audit_record(
                result, investigator_token
            )
    finally:
        end_demo_session(investigator_token)

    # Step 5: Verify an auditor can inspect the record.
    auditor_token = create_demo_session("sarah")

    try:
        audit = get_authenticated_audit_record(
            result, auditor_token
        )

        assert audit["rejected_count"] == 1
        assert audit["rejected_claims"][0]["claim"]["field"] == (
            "confirmed_fraud"
        )
    finally:
        end_demo_session(auditor_token)
