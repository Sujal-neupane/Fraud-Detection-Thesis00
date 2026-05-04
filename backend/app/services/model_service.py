from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

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


def predict(features: Dict[str, float]) -> Tuple[bool, float, str]:
    bundle = _load_bundle()
    model = bundle["model"]
    feature_order = bundle["features"]
    categorical_cols = bundle.get("categorical_features", [])
    threshold = float(bundle.get("threshold", 0.5))
    model_version = str(bundle.get("model_version", "unknown"))

    frame = _build_frame(features, feature_order, categorical_cols)
    score = float(model.predict_proba(frame)[:, 1][0])
    is_fraud = score >= threshold

    return is_fraud, score, model_version
