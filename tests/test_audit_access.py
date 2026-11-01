
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audit_access import get_audit_record


@pytest.fixture
def audit_result():
    return {
        "investigator_view": {
            "status": "filtered",
            "summary": "Verified information only.",
        },
        "audit_record": {
            "event_id": "test-event-001",
            "rejected_claims": [
                {"reason": "Unsupported accusation"}
            ],
        },
    }


def test_security_auditor_allowed(audit_result):
    record = get_audit_record(
        audit_result, "security_auditor"
    )
    assert record["event_id"] == "test-event-001"


def test_compliance_officer_allowed(audit_result):
    record = get_audit_record(
        audit_result, "compliance_officer"
    )
    assert "rejected_claims" in record


def test_investigator_denied(audit_result):
    with pytest.raises(PermissionError):
        get_audit_record(
            audit_result, "investigator"
        )


def test_unknown_role_denied(audit_result):
    with pytest.raises(PermissionError):
        get_audit_record(
            audit_result, "administrator"
        )


def test_missing_role_denied(audit_result):
    with pytest.raises(PermissionError):
        get_audit_record(
            audit_result, None
        )


def test_missing_audit_record_rejected():
    with pytest.raises(ValueError):
        get_audit_record(
            {"investigator_view": {}},
            "security_auditor",
        )
