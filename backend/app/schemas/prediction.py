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
    business_impact: Optional["BusinessImpact"] = None
    dynamic_threshold: Optional[float] = None
    dynamic_is_fraud: Optional[bool] = None
    dynamic_business_impact: Optional["BusinessImpact"] = None


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: str


class BusinessImpact(BaseModel):
    threshold: float
    flagged: bool
    transaction_amount: float
    expected_fraud_loss: float
    false_positive_cost: float
    net_benefit: float


PredictionResponse.model_rebuild()

