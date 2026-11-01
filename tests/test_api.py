
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api import app


@pytest.fixture(scope="module")
def client():
    # The context manager runs FastAPI's startup
    # and shutdown lifecycle.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_features(client):
    artifact = client.app.state.model_artifact

    return {
        feature: 0.0
        for feature in artifact["features"]
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_valid_prediction(client, valid_features):
    response = client.post(
        "/predict",
        json={
            "features": valid_features,
            "threshold": 0.50,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["fraud_score"] <= 1
    assert result["threshold"] == 0.50
    assert isinstance(result["flagged"], bool)


def test_missing_feature(client, valid_features):
    valid_features.pop("Amount")

    response = client.post(
        "/predict",
        json={
            "features": valid_features,
            "threshold": 0.50,
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold(client, valid_features, threshold):
    response = client.post(
        "/predict",
        json={
            "features": valid_features,
            "threshold": threshold,
        },
    )

    assert response.status_code == 422
