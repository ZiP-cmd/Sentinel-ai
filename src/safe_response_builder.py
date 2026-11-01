
from claim_validator import validate_claims
from investigation_assistant import generate_investigation_summary


def build_safe_response(evidence, candidate_claims):
    """Build a response from approved claims only."""

    # Use the existing reference builder to check
    # the basic structure and consistency of evidence.
    fallback = generate_investigation_summary(evidence)

    review = validate_claims(evidence, candidate_claims)

    approved_fields = {
        claim["field"]
        for claim in review["supported_claims"]
    }

    # Require both the score and review decision.
    required_fields = {"model_score", "review_required"}

    if not required_fields.issubset(approved_fields):
        return {
            "status": "fallback",
            "text": fallback,
            "rejected_count": len(review["rejected_claims"]),
        }

    score = evidence["model_score"]
    flagged = evidence["review_required"]

    if flagged:
        decision = "selected for review"
    else:
        decision = "not selected for review"

    # Construct new wording from trusted evidence.
    # Never copy arbitrary candidate text.
    summary = (
        f"Sentinel reported a model score of {score:.4f}. "
        f"The transaction was {decision}. "
        "The score is not a calibrated fraud probability "
        "and does not establish that fraud occurred."
    )

    return {
        "status": (
            "filtered"
            if review["rejected_claims"]
            else "approved"
        ),
        "text": summary,
        "rejected_count": len(review["rejected_claims"]),
    }
