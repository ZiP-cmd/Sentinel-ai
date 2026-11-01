
def generate_investigation_summary(evidence):
    """Generate a factual, rule-based investigation summary."""

    required = [
        "transaction_amount",
        "model_score",
        "decision_threshold",
        "review_required",
        "limitations",
    ]

    missing = [key for key in required if key not in evidence]

    if missing:
        raise ValueError(
            f"Missing evidence fields: {missing}"
        )

    amount = evidence["transaction_amount"]
    score = evidence["model_score"]
    threshold = evidence["decision_threshold"]
    flagged = evidence["review_required"]

    if not isinstance(flagged, bool):
        raise ValueError("Review decision must be a boolean.")

    if not (0 <= score <= 1 and 0 <= threshold <= 1):
        raise ValueError("Score and threshold must be between 0 and 1.")

    if flagged != (score >= threshold):
        raise ValueError("Evidence contains an inconsistent decision.")

    if flagged:
        decision = "FLAGGED FOR REVIEW"
        next_step = (
            "A human investigator should review available "
            "transaction records before reaching a conclusion."
        )
    else:
        decision = "NOT FLAGGED"
        next_step = (
            "No review was triggered by this model policy. "
            "Other established fraud controls may still apply."
        )

    summary = (
        "SENTINEL AI INVESTIGATION SUMMARY\n"
        "---------------------------------\n"
        f"Transaction amount: ${amount:,.2f}\n"
        f"Model score: {score:.4f}\n"
        f"Decision threshold: {threshold:.2f}\n"
        f"Decision: {decision}\n\n"
        "ASSESSMENT\n"
        "The review decision is based on whether the "
        "model score meets the configured threshold. "
        "This does not establish whether fraud occurred.\n\n"
        "RECOMMENDED NEXT STEP\n"
        f"{next_step}\n\n"
        "LIMITATIONS\n"
    )

    for limitation in evidence["limitations"]:
        summary += f"- {limitation}\n"

    return summary
