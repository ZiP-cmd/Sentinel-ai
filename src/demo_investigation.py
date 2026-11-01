
import json

from investigation_evidence import (
    build_investigation_evidence,
)

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

print("\n=== SENTINEL AI INVESTIGATION EVIDENCE ===")

print(json.dumps(evidence, indent=2))
