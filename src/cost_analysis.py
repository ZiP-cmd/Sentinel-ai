
def estimate_cost(
    alerts,
    missed_fraud,
    cost_per_review,
    loss_per_missed_fraud,
):
    """Estimate cost under simplified business assumptions."""

    values = [
        alerts,
        missed_fraud,
        cost_per_review,
        loss_per_missed_fraud,
    ]

    if any(value < 0 for value in values):
        raise ValueError("Costs and counts cannot be negative.")

    if int(alerts) != alerts or int(missed_fraud) != missed_fraud:
        raise ValueError("Transaction counts must be whole numbers.")

    review_cost = alerts * cost_per_review
    missed_fraud_cost = missed_fraud * loss_per_missed_fraud

    return {
        "review_cost": review_cost,
        "missed_fraud_cost": missed_fraud_cost,
        "total_cost": review_cost + missed_fraud_cost,
    }
