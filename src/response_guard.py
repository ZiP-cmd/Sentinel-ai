
from investigation_assistant import generate_investigation_summary


def approve_response(evidence, candidate_response):
    """Allow only the verified reference summary."""

    trusted_summary = generate_investigation_summary(evidence)

    if not isinstance(candidate_response, str):
        return {
            "approved": False,
            "reason": "Invalid response format",
            "response": trusted_summary,
        }

    if candidate_response != trusted_summary:
        return {
            "approved": False,
            "reason": "Response differs from verified reference",
            "response": trusted_summary,
        }

    return {
        "approved": True,
        "reason": "Matches verified reference",
        "response": candidate_response,
    }
