from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    transaction_id: Optional[str] = None
    features: Dict[str, float] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    is_fraud: bool
    score: float
    model_version: str
    threshold: float
    explanations: List["FeatureContribution"] = Field(default_factory=list)


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: str


PredictionResponse.model_rebuild()
