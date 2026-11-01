
from investigation_evidence import build_investigation_evidence
from investigation_assistant import generate_investigation_summary


transaction = {
    "Amount": 50.00,
}

prediction = {
    "fraud_score": 0.2941,
    "threshold": 0.50,
    "flagged": False,
}

evidence = build_investigation_evidence(
    transaction,
    prediction,
)

summary = generate_investigation_summary(evidence)

print()
print(summary)
