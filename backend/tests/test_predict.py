"""
Unit tests for the model_service.predict() function.

Run with:  cd backend && python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.model_service import predict


def test_predict_returns_dict() -> None:
    result = predict({"TransactionAmt": 100.0, "D1": 100.0, "C1": 2.0})
    assert isinstance(result, dict)


def test_predict_required_keys() -> None:
    result = predict({"TransactionAmt": 100.0})
    for key in ("is_fraud", "score", "model_version", "threshold", "explanations",
                "business_impact", "dynamic_threshold", "dynamic_is_fraud"):
        assert key in result, f"Missing key: {key}"


def test_score_in_unit_interval() -> None:
    result = predict({"TransactionAmt": 250.0, "D1": 60.0})
    assert 0.0 <= result["score"] <= 1.0


def test_threshold_in_unit_interval() -> None:
    result = predict({"TransactionAmt": 250.0})
    assert 0.0 <= result["threshold"] <= 1.0


def test_high_risk_scores_above_low_risk() -> None:
    low = predict({"TransactionAmt": 30.0, "D1": 500.0, "C1": 1.0, "V14": 2.5, "dist1": 2.0})
    high = predict({"TransactionAmt": 6000.0, "D1": 1.0, "C1": 28.0, "V14": -5.5, "dist1": 490.0})
    assert high["score"] > low["score"], (
        f"Expected high-risk ({high['score']:.4f}) > low-risk ({low['score']:.4f})"
    )


def test_explanations_structure() -> None:
    result = predict({"TransactionAmt": 100.0, "D1": 100.0, "C1": 2.0})
    assert isinstance(result["explanations"], list)
    assert len(result["explanations"]) > 0
    for exp in result["explanations"]:
        assert "feature" in exp
        assert "contribution" in exp
        assert exp["direction"] in ("increase", "decrease")


def test_dynamic_threshold_inverse_of_amount() -> None:
    small_tx = predict({"TransactionAmt": 10.0})
    large_tx = predict({"TransactionAmt": 1000.0})
    assert small_tx["dynamic_threshold"] > large_tx["dynamic_threshold"], (
        "Small transactions should have a higher dynamic threshold (harder to flag)"
    )


def test_is_fraud_consistent_with_score_and_threshold() -> None:
    result = predict({"TransactionAmt": 500.0, "D1": 5.0, "C1": 20.0})
    expected = result["score"] >= result["threshold"]
    assert result["is_fraud"] == expected
