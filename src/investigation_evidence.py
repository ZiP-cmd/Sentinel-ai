
import math


def build_investigation_evidence(transaction, prediction):
    """Prepare validated evidence for an investigator."""

    if "Amount" not in transaction:
        raise ValueError("Transaction amount is required.")

    required_fields = [
        "fraud_score",
        "threshold",
        "flagged",
    ]

    for field in required_fields:
        if field not in prediction:
            raise ValueError(
                f"Missing prediction field: {field}"
            )

    # Reject booleans before numeric conversion.
    if isinstance(transaction["Amount"], bool):
        raise ValueError(
            "Transaction amount must be numeric, not boolean."
        )

    amount = float(transaction["Amount"])
    score = float(prediction["fraud_score"])
    threshold = float(prediction["threshold"])
    flagged = prediction["flagged"]

    # Reject NaN and infinite values.
    if not math.isfinite(amount):
        raise ValueError(
            "Transaction amount must be a finite number."
        )

    if not math.isfinite(score):
        raise ValueError(
            "Fraud score must be a finite number."
        )

    if not math.isfinite(threshold):
        raise ValueError(
            "Threshold must be a finite number."
        )

    # Check allowed score and threshold ranges.
    if not 0 <= score <= 1:
        raise ValueError(
            "Fraud score must be between 0 and 1."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if not isinstance(flagged, bool):
        raise ValueError("Flagged must be a boolean.")

    if flagged != (score >= threshold):
        raise ValueError(
            "Prediction decision is inconsistent."
        )

    return {
        "transaction_amount": amount,
        "model_score": score,
        "decision_threshold": threshold,
        "review_required": flagged,
        "verified_facts": [
            f"Transaction amount: ${amount:.2f}",
            f"Model score: {score:.4f}",
            f"Decision threshold: {threshold:.2f}",
            (
                "Transaction flagged for review."
                if flagged
                else "Transaction not flagged for review."
            ),
        ],
        "limitations": [
            "The model score is not a calibrated fraud probability.",
            "A fraud alert does not prove fraud occurred.",
            "No transaction-specific fraud cause has been established.",
            "Human review is required for investigative conclusions.",
        ],
    }
