
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from investigation_evidence import build_investigation_evidence
from investigation_assistant import generate_investigation_summary


def make_evidence(score, threshold=0.50):
    return build_investigation_evidence(
        {"Amount": 50.00},
        {
            "fraud_score": score,
            "threshold": threshold,
            "flagged": score >= threshold,
        },
    )


def test_not_flagged_summary():
    evidence = make_evidence(0.2941)
    summary = generate_investigation_summary(evidence)

    assert "NOT FLAGGED" in summary
    assert "0.2941" in summary
    assert "$50.00" in summary


def test_flagged_summary():
    evidence = make_evidence(0.80)
    summary = generate_investigation_summary(evidence)

    assert "FLAGGED FOR REVIEW" in summary
    assert "human investigator" in summary


def test_missing_evidence_rejected():
    with pytest.raises(ValueError, match="Missing evidence"):
        generate_investigation_summary({})


def test_summary_does_not_claim_proven_fraud():
    evidence = make_evidence(0.95)
    summary = generate_investigation_summary(evidence)

    assert "does not establish whether fraud occurred" in summary
    assert "confirmed fraudulent" not in summary.lower()
