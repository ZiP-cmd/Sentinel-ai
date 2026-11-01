
from investigation_assistant import generate_investigation_summary


def generate_structured_summary(evidence, style="standard"):
    """Generate a controlled paraphrase from verified evidence."""

    # Validate the evidence against our existing reference logic.
    reference = generate_investigation_summary(evidence)

    if style == "standard":
        return reference

    if style != "concise":
        raise ValueError("Unsupported summary style.")

    score = evidence["model_score"]
    threshold = evidence["decision_threshold"]
    amount = evidence["transaction_amount"]
    flagged = evidence["review_required"]

    if flagged:
        decision = "selected for human review"
    else:
        decision = "not selected for review"

    return (
        f"Sentinel assessed a ${amount:.2f} transaction. "
        f"The model score was {score:.4f}, compared with "
        f"a review threshold of {threshold:.2f}. "
        f"The transaction was {decision}. "
        "This score is not a calibrated fraud probability "
        "and does not prove fraud occurred."
    )
