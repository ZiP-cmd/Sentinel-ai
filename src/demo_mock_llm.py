
from investigation_evidence import build_investigation_evidence
from mock_llm import mock_generate
from response_guard import approve_response


evidence = build_investigation_evidence(
    {"Amount": 50.00},
    {
        "fraud_score": 0.91,
        "threshold": 0.50,
        "flagged": True,
    },
)

scenarios = [
    "safe",
    "unsupported_accusation",
    "fabricated_location",
]

for scenario in scenarios:
    candidate = mock_generate(evidence, scenario)

    result = approve_response(evidence, candidate)

    print(f"\n=== Scenario: {scenario} ===")
    print(f"Approved: {result['approved']}")
    print(f"Reason: {result['reason']}")
    print(f"Candidate: {candidate[:120]}")
    print(
        "Delivered trusted summary: "
        f"{result['response'].startswith('SENTINEL AI')}"
    )
