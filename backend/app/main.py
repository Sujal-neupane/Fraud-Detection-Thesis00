from fastapi import FastAPI, HTTPException

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_service import predict

app = FastAPI(title="Fraud Detection API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(payload: PredictionRequest) -> PredictionResponse:
    try:
        is_fraud, score, model_version = predict(payload.features)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail="Model not trained yet.") from exc
    return PredictionResponse(
        is_fraud=is_fraud,
        score=score,
        model_version=model_version,
    )
