
import math


def validate_claims(evidence, claims):
    """Validate structured claims against trusted evidence."""

    if not isinstance(claims, list):
        raise ValueError("Claims must be a list.")

    supported = []
    rejected = []

    allowed_fields = {
        "transaction_amount": "transaction_amount",
        "model_score": "model_score",
        "decision_threshold": "decision_threshold",
        "review_required": "review_required",
    }

    for claim in claims:
        if not isinstance(claim, dict):
            rejected.append({
                "claim": claim,
                "reason": "Invalid claim format",
            })
            continue

        field = claim.get("field")
        value = claim.get("value")

        if field not in allowed_fields:
            rejected.append({
                "claim": claim,
                "reason": "No verified evidence for this field",
            })
            continue

        expected = evidence[allowed_fields[field]]

        # Decisions must be actual booleans.
        if field == "review_required":
            valid = (
                type(value) is bool
                and value is expected
            )
        else:
            valid = (
                type(value) in (int, float)
                and math.isfinite(value)
                and value == expected
            )

        if valid:
            supported.append({
                "field": field,
                "value": expected,
            })
        else:
            rejected.append({
                "claim": claim,
                "reason": "Claim does not match verified evidence",
            })

    return {
        "approved": len(rejected) == 0,
        "supported_claims": supported,
        "rejected_claims": rejected,
    }
