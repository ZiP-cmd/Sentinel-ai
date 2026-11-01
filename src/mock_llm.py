
from investigation_assistant import generate_investigation_summary


def mock_generate(evidence, scenario="safe"):
    """Simulate different language-model responses."""

    if scenario == "safe":
        return generate_investigation_summary(evidence)

    if scenario == "unsupported_accusation":
        return (
            "This customer is definitely committing fraud. "
            "Block the transaction immediately."
        )

    if scenario == "fabricated_location":
        return (
            "This transaction was made overseas "
            "from an unfamiliar device."
        )

    raise ValueError(f"Unknown mock scenario: {scenario}")