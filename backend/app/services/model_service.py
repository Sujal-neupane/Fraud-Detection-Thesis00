from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from catboost import Pool

from app.core.config import MODEL_PATH

_MODEL_BUNDLE: dict | None = None


def _load_bundle() -> dict:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise NotImplementedError("Model artifact not found. Train the model first.")

    _MODEL_BUNDLE = joblib.load(model_path)
    return _MODEL_BUNDLE


def _build_frame(
    features: Dict[str, float],
    feature_order: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    row: dict[str, object] = {}
    for col in feature_order:
        if col in features:
            value = features[col]
        else:
            value = "missing" if col in categorical_cols else 0.0

        if col in categorical_cols:
            row[col] = str(value)
        else:
            row[col] = float(value)

    return pd.DataFrame([row])


def _explain(
    model: object,
    frame: pd.DataFrame,
    feature_order: list[str],
    categorical_cols: list[str],
    top_n: int = 8,
) -> List[Dict[str, object]]:
    pool = Pool(frame, cat_features=categorical_cols)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    contributions = shap_values[0][:-1]
    paired = list(zip(feature_order, contributions))
    paired.sort(key=lambda item: abs(item[1]), reverse=True)
    top = []
    for feature, value in paired[:top_n]:
        top.append(
            {
                "feature": feature,
                "contribution": float(value),
                "direction": "increase" if value >= 0 else "decrease",
            }
        )
    return top


def _sparse_risk_signal(features: Dict[str, float]) -> float:
    signal_weights = []

    transaction_amount = features.get("TransactionAmt")
    if transaction_amount is not None:
        amount_signal = min(float(transaction_amount) / 1000.0, 1.0)
        signal_weights.append((amount_signal, 0.35))

    c1_value = features.get("C1")
    if c1_value is not None:
        c1_signal = 1.0 - min(float(c1_value) / 20.0, 1.0)
        signal_weights.append((c1_signal, 0.15))

    d1_value = features.get("D1")
    if d1_value is not None:
        d1_signal = 1.0 - min(float(d1_value) / 200.0, 1.0)
        signal_weights.append((d1_signal, 0.2))

    v14_value = features.get("V14")
    if v14_value is not None:
        v14_signal = min(max(float(v14_value), 0.0) / 4.0, 1.0)
        signal_weights.append((v14_signal, 0.2))

    v258_value = features.get("V258")
    if v258_value is not None:
        v258_signal = 1.0 - min(max(float(v258_value), 0.0) / 2.0, 1.0)
        signal_weights.append((v258_signal, 0.1))

    if not signal_weights:
        return 0.5

    weighted_sum = sum(signal * weight for signal, weight in signal_weights)
    total_weight = sum(weight for _, weight in signal_weights)
    return float(weighted_sum / total_weight)


def _blend_scores(model_score: float, sparse_risk: float) -> float:
    return float((0.5 * model_score) + (0.5 * sparse_risk))


def predict(
    features: Dict[str, float],
) -> Tuple[bool, float, str, float, List[Dict[str, object]], Dict[str, object]]:
    bundle = _load_bundle()
    model = bundle["model"]
    feature_order = bundle["features"]
    categorical_cols = bundle.get("categorical_features", [])
    threshold = float(bundle.get("threshold", 0.5))
    model_version = str(bundle.get("model_version", "unknown"))

    frame = _build_frame(features, feature_order, categorical_cols)
    model_score = float(model.predict_proba(frame)[:, 1][0])
    sparse_risk = _sparse_risk_signal(features)
    score = _blend_scores(model_score, sparse_risk)
    is_fraud = score >= threshold
    explanations = _explain(model, frame, feature_order, categorical_cols)
    transaction_amount = float(features.get("TransactionAmt", 0.0) or 0.0)
    expected_fraud_loss = score * transaction_amount
    false_positive_cost = 5.0 if is_fraud else 0.0
    net_benefit = expected_fraud_loss - false_positive_cost
    business_impact = {
        "threshold": threshold,
        "flagged": is_fraud,
        "transaction_amount": transaction_amount,
        "expected_fraud_loss": float(expected_fraud_loss),
        "false_positive_cost": float(false_positive_cost),
        "net_benefit": float(net_benefit),
    }

    return is_fraud, score, model_version, threshold, explanations, business_impact
