from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    transaction_id: Optional[str] = None
    features: Dict[str, float] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    is_fraud: bool
    score: float
    model_version: str
