
from datetime import datetime, timezone
from uuid import uuid4

from claim_validator import validate_claims
from safe_response_builder import build_safe_response


def create_investigation_audit(evidence, candidate_claims):
    """
    Generate a safe investigator response and
    a separate internal audit record.

    Audit information is returned in memory.
    This function does not save it to disk.
    """

    validation = validate_claims(evidence, candidate_claims)

    safe_response = build_safe_response(
        evidence,
        candidate_claims,
    )

    audit_record = {
        "event_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "ai_claim_validation",
        "validation_status": safe_response["status"],
        "approved_count": len(
            validation["supported_claims"]
        ),
        "rejected_count": len(
            validation["rejected_claims"]
        ),
        "rejected_claims": validation["rejected_claims"],
    }

    # The investigator-facing output contains no
    # raw rejected claims or audit details.
    investigator_view = {
        "status": safe_response["status"],
        "summary": safe_response["text"],
        "excluded_claim_count": safe_response["rejected_count"],
    }

    return {
        "investigator_view": investigator_view,
        "audit_record": audit_record,
    }
