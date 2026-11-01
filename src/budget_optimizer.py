
from alert_budget import evaluate_alert_budget
from cost_analysis import estimate_cost


def compare_budget_costs(
    y_true,
    scores,
    budgets,
    cost_per_review,
    loss_per_missed_fraud,
):
    """Compare financial costs for several alert budgets."""

    if not budgets:
        raise ValueError("At least one budget is required.")

    if len(set(budgets)) != len(budgets):
        raise ValueError("Budgets must be unique.")

    results = []

    for budget in budgets:
        evaluation = evaluate_alert_budget(
            y_true, scores, budget
        )

        costs = estimate_cost(
            alerts=budget,
            missed_fraud=evaluation["fn"],
            cost_per_review=cost_per_review,
            loss_per_missed_fraud=loss_per_missed_fraud,
        )

        results.append({
            "budget": budget,
            "fraud_caught": evaluation["tp"],
            "fraud_missed": evaluation["fn"],
            "false_alarms": evaluation["fp"],
            "total_cost": costs["total_cost"],
        })

    # Choose the least expensive policy.
    # Break ties in favor of fewer investigations.
    best = min(
        results,
        key=lambda row: (row["total_cost"], row["budget"])
    )

    return {
        "results": results,
        "recommended_budget": best["budget"],
        "lowest_total_cost": best["total_cost"],
    }
