from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from app.core.config import MODEL_PATH

_FALSE_POSITIVE_COST = 5.0
_SPARSE_THRESHOLD = 20
_SPARSE_MULTIPLIER = 1.5

_MODEL_BUNDLE: dict | None = None


def _load_bundle() -> dict:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise NotImplementedError("Model artifact not found. Train the model first.")
    bundle: dict = joblib.load(model_path)
    _MODEL_BUNDLE = bundle
    return bundle


def _build_frame(
    features: Dict[str, float],
    feature_order: list[str],
    categorical_cols: list[str],
    native_nan: bool = False,
) -> pd.DataFrame:
    row: dict[str, object] = {}
    for col in feature_order:
        if col in features:
            value = features[col]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = "missing" if col in categorical_cols else np.nan
        else:
            value = "missing" if col in categorical_cols else (np.nan if native_nan else 0.0)

        row[col] = str(value) if col in categorical_cols else float(value)  # type: ignore[arg-type]

    return pd.DataFrame([row])


def _explain(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
    feature_order: list[str],
    categorical_cols: list[str],
    top_n: int = 8,
) -> List[Dict[str, object]]:
    pool = Pool(frame, cat_features=categorical_cols)
    shap_values = model.get_feature_importance(pool, type="ShapValues")  # type: ignore[arg-type]
    contributions = shap_values[0][:-1]  # type: ignore[index]
    paired = sorted(zip(feature_order, contributions), key=lambda t: abs(t[1]), reverse=True)
    return [
        {"feature": feat, "contribution": float(val), "direction": "increase" if val >= 0 else "decrease"}
        for feat, val in paired[:top_n]
    ]


def _sparse_risk_signal(features: Dict[str, float]) -> float:
    """
    Heuristic fallback risk score for sparse inputs.

    When a request provides fewer features than the model was trained on,
    raw model confidence collapses toward the training prior. This signal
    blends deterministic rules derived from high-signal IEEE-CIS features
    (TransactionAmt, C1, D1, V14, V258, dist1) to maintain robustness.
    """
    signal_weights = []

    if (v := features.get("TransactionAmt")) is not None:
        signal_weights.append((min(float(v) / 1000.0, 1.0), 0.30))

    if (v := features.get("C1")) is not None:
        signal_weights.append((1.0 - min(float(v) / 20.0, 1.0), 0.13))

    if (v := features.get("D1")) is not None:
        signal_weights.append((1.0 - min(float(v) / 200.0, 1.0), 0.18))

    if (v := features.get("V14")) is not None:
        signal_weights.append((min(abs(float(v)) / 6.0, 1.0), 0.22))

    if (v := features.get("V258")) is not None:
        signal_weights.append((1.0 - min(max(float(v), 0.0) / 2.0, 1.0), 0.09))

    if (v := features.get("dist1")) is not None:
        signal_weights.append((min(v / 300.0, 1.0), 0.08))

    if not signal_weights:
        return 0.5

    weighted_avg = sum(s * w for s, w in signal_weights) / sum(w for _, w in signal_weights)
    multiplier = _SPARSE_MULTIPLIER if len(features) <= _SPARSE_THRESHOLD else 1.0
    return min(weighted_avg * multiplier, 1.0)


def _blend_scores(model_score: float, sparse_risk: float, feature_count: int) -> float:
    # Trust the model more when the caller provides richer input.
    # 5 features → ~22% model weight; 50+ features → ~90% model weight.
    model_weight = min(0.90, 0.15 + feature_count * 0.015)
    return model_weight * model_score + (1.0 - model_weight) * sparse_risk


def _business_impact(
    score: float, threshold: float, transaction_amount: float
) -> dict:
    flagged = score >= threshold
    expected_loss = score * transaction_amount
    fp_cost = _FALSE_POSITIVE_COST if flagged else 0.0
    return {
        "threshold": threshold,
        "flagged": flagged,
        "transaction_amount": transaction_amount,
        "expected_fraud_loss": float(expected_loss),
        "false_positive_cost": float(fp_cost),
        "net_benefit": float(expected_loss - fp_cost),
    }


def predict(features: Dict[str, float]) -> dict:
    bundle = _load_bundle()
    model: CatBoostClassifier = bundle["model"]
    feature_order: list[str] = bundle["features"]
    categorical_cols: list[str] = bundle.get("categorical_features", [])
    threshold = float(bundle.get("threshold", 0.5))
    model_version = str(bundle.get("model_version", "unknown"))
    native_nan = bool(bundle.get("native_nan", False))

    frame = _build_frame(features, feature_order, categorical_cols, native_nan=native_nan)
    model_score = float(model.predict_proba(frame)[:, 1][0])
    sparse_risk = _sparse_risk_signal(features)
    score = _blend_scores(model_score, sparse_risk, len(features))
    is_fraud = score >= threshold

    transaction_amount = float(features.get("TransactionAmt") or 0.0)

    if transaction_amount > 0:
        raw_dynamic = _FALSE_POSITIVE_COST / transaction_amount
        dynamic_threshold = float(max(0.01, min(0.95, raw_dynamic)))
    else:
        dynamic_threshold = threshold

    dynamic_is_fraud = score >= dynamic_threshold

    return {
        "is_fraud": is_fraud,
        "score": score,
        "model_version": model_version,
        "threshold": threshold,
        "explanations": _explain(model, frame, feature_order, categorical_cols),
        "business_impact": _business_impact(score, threshold, transaction_amount),
        "dynamic_threshold": dynamic_threshold,
        "dynamic_is_fraud": dynamic_is_fraud,
        "dynamic_business_impact": _business_impact(score, dynamic_threshold, transaction_amount),
    }
