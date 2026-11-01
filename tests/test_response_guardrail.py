
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from investigation_evidence import build_investigation_evidence
from mock_llm import mock_generate
from response_guard import approve_response


@pytest.fixture
def evidence():
    return build_investigation_evidence(
        {"Amount": 50.00},
        {
            "fraud_score": 0.91,
            "threshold": 0.50,
            "flagged": True,
        },
    )


def test_safe_response_approved(evidence):
    candidate = mock_generate(evidence, "safe")

    result = approve_response(evidence, candidate)

    assert result["approved"] is True
    assert result["response"] == candidate


@pytest.mark.parametrize(
    "scenario",
    ["unsupported_accusation", "fabricated_location"],
)
def test_unsafe_response_rejected(evidence, scenario):
    candidate = mock_generate(evidence, scenario)

    result = approve_response(evidence, candidate)

    assert result["approved"] is False
    assert result["response"] != candidate
    assert "does not establish" in result["response"]


def test_invalid_response_format_rejected(evidence):
    result = approve_response(evidence, None)

    assert result["approved"] is False


def test_unknown_scenario_rejected(evidence):
    with pytest.raises(ValueError, match="Unknown mock scenario"):
        mock_generate(evidence, "unknown")
