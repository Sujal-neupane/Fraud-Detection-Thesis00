from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_service import predict

app = FastAPI(title="Fraud Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(payload: PredictionRequest) -> PredictionResponse:
    try:
        is_fraud, score, model_version, threshold, explanations, business_impact = predict(
            payload.features
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail="Model not trained yet.") from exc
    return PredictionResponse(
        is_fraud=is_fraud,
        score=score,
        model_version=model_version,
        threshold=threshold,
        explanations=explanations,
        business_impact=business_impact,
    )
