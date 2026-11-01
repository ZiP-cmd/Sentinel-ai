
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from prediction_service import load_model, predict_transaction


@pytest.fixture(scope="module")
def model_artifact():
    """Load the model once for the entire test module."""
    return load_model()


@pytest.fixture
def valid_transaction(model_artifact):
    """Create a correctly shaped synthetic transaction."""
    features = model_artifact["features"]

    return {feature: 0.0 for feature in features}


def test_valid_transaction(model_artifact, valid_transaction):
    result = predict_transaction(
        valid_transaction,
        model_artifact,
        threshold=0.50,
    )

    assert isinstance(result["fraud_score"], float)
    assert 0.0 <= result["fraud_score"] <= 1.0
    assert result["threshold"] == 0.50
    assert isinstance(result["flagged"], bool)


def test_missing_features(model_artifact, valid_transaction):
    # Remove a required feature.
    valid_transaction.pop("Amount")

    with pytest.raises(ValueError, match="Missing transaction features"):
        predict_transaction(
            valid_transaction,
            model_artifact,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold(
    model_artifact,
    valid_transaction,
    threshold,
):
    with pytest.raises(ValueError, match="Threshold"):
        predict_transaction(
            valid_transaction,
            model_artifact,
            threshold=threshold,
        )
