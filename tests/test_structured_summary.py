
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from investigation_evidence import build_investigation_evidence
from structured_summary import generate_structured_summary


@pytest.fixture
def flagged_evidence():
    return build_investigation_evidence(
        {"Amount": 75.00},
        {
            "fraud_score": 0.91,
            "threshold": 0.50,
            "flagged": True,
        },
    )


def test_standard_summary(flagged_evidence):
    result = generate_structured_summary(
        flagged_evidence, style="standard"
    )

    assert "FLAGGED FOR REVIEW" in result


def test_concise_summary(flagged_evidence):
    result = generate_structured_summary(
        flagged_evidence, style="concise"
    )

    assert "$75.00" in result
    assert "0.9100" in result
    assert "selected for human review" in result


def test_concise_summary_preserves_limitations(flagged_evidence):
    result = generate_structured_summary(
        flagged_evidence, style="concise"
    )

    assert "not a calibrated fraud probability" in result
    assert "does not prove fraud occurred" in result


def test_unknown_style_rejected(flagged_evidence):
    with pytest.raises(ValueError):
        generate_structured_summary(
            flagged_evidence, style="creative"
        )
