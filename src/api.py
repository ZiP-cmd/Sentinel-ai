
from contextlib import asynccontextmanager
from math import isfinite

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from prediction_service import load_model, predict_transaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the saved model once when the server starts.
    app.state.model_artifact = load_model()
    yield


app = FastAPI(
    title="Sentinel AI",
    description="Machine Learning Fraud Detection API",
    version="0.2.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    features: dict[str, float]
    threshold: float = Field(default=0.50, ge=0, le=1)


@app.get("/health")
def health_check():
    return {
        "status": "running",
        "service": "Sentinel AI",
    }


@app.post("/predict")
def predict(request: PredictionRequest, http_request: Request):
    # Reject missing or non-finite numeric inputs.
    features = request.features

    if not all(isfinite(value) for value in features.values()):
        raise HTTPException(
            status_code=422,
            detail="All feature values must be finite numbers.",
        )

    if not isfinite(request.threshold):
        raise HTTPException(
            status_code=422,
            detail="Threshold must be a finite number.",
        )

    try:
        return predict_transaction(
            transaction=features,
            artifact=http_request.app.state.model_artifact,
            threshold=request.threshold,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
