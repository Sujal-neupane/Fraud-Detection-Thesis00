import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import METRICS_PATH
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_service import predict

app = FastAPI(title="ShieldML Fraud Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
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
        result = predict(payload.features)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail="Model not trained yet.") from exc
    return PredictionResponse(**result)


@app.get("/model-comparison")
def model_comparison() -> dict:
    comp_path = Path(METRICS_PATH).parent / "model_comparison.json"
    if not comp_path.exists():
        return {"models": []}
    try:
        return json.loads(comp_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read comparison: {exc}") from exc


@app.get("/metrics")
def get_metrics() -> dict:
    metrics_path = Path(METRICS_PATH)
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Train the model first.",
        )
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {exc}") from exc


class OptimalThresholdRequest(BaseModel):
    transaction_amount: float
    false_positive_cost: float = 5.0


@app.post("/threshold/optimal")
def optimal_threshold(body: OptimalThresholdRequest) -> dict:
    """
    Compute the Bayesian cost-optimal classification threshold.

    Derived from the utility equation U(x) = P(Fraud|x) * TransactionAmt - C_FP.
    Setting U(x) = 0 and solving: theta* = C_FP / TransactionAmt.
    """
    if body.transaction_amount <= 0:
        raise HTTPException(status_code=422, detail="transaction_amount must be positive.")
    raw = body.false_positive_cost / body.transaction_amount
    theta_star = round(max(0.01, min(0.95, raw)), 6)
    return {
        "transaction_amount": body.transaction_amount,
        "false_positive_cost": body.false_positive_cost,
        "optimal_threshold": theta_star,
        "formula": "theta* = C_FP / TransactionAmt",
    }


@app.get("/presets")
def get_presets() -> dict:
    return {
        "low_risk": {
            "TransactionAmt": 45.50,
            "card1": 10220.0,
            "card2": 512.0,
            "card3": 150.0,
            "card5": 226.0,
            "addr1": 204.0,
            "addr2": 87.0,
            "dist1": 5.0,
            "C1": 1.0,
            "C2": 1.0,
            "C3": 0.0,
            "C4": 0.0,
            "C5": 1.0,
            "C6": 1.0,
            "C7": 0.0,
            "C8": 0.0,
            "C9": 1.0,
            "C10": 0.0,
            "C11": 1.0,
            "C12": 0.0,
            "C13": 1.0,
            "V14": 1.0,
            "V4": 1.0,
            "V258": 1.0,
            "D1": 150.0,
        },
        "high_risk_cnp": {
            "TransactionAmt": 950.00,
            "card1": 4045.0,
            "card2": 111.0,
            "card3": 185.0,
            "card5": 102.0,
            "addr1": 325.0,
            "addr2": 87.0,
            "dist1": 120.0,
            "C1": 15.0,
            "C2": 20.0,
            "C3": 0.0,
            "C4": 4.0,
            "C5": 0.0,
            "C6": 12.0,
            "C7": 2.0,
            "C8": 8.0,
            "C9": 0.0,
            "C10": 6.0,
            "C11": 14.0,
            "C12": 3.0,
            "C13": 45.0,
            "V14": -3.5,
            "V4": -2.0,
            "V258": 0.1,
            "D1": 12.5,
        },
        "suspicious_intl": {
            "TransactionAmt": 1500.50,
            "card1": 9500.0,
            "card2": 321.0,
            "card3": 185.0,
            "card5": 224.0,
            "addr1": 440.0,
            "addr2": 87.0,
            "dist1": 400.0,
            "C1": 25.0,
            "C2": 35.0,
            "C3": 0.0,
            "C4": 10.0,
            "C5": 0.0,
            "C6": 22.0,
            "C7": 5.0,
            "C8": 15.0,
            "C9": 0.0,
            "C10": 12.0,
            "C11": 28.0,
            "C12": 8.0,
            "C13": 85.0,
            "V14": -5.2,
            "V4": 4.5,
            "V258": 0.0,
            "D1": 2.0,
        },
    }
