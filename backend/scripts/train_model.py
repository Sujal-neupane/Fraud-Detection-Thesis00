from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fraud detection model on synthetic IEEE-CIS data."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "synthetic"
        / "ieee-cis"
        / "synthetic_2019_2025.csv",
        help="Synthetic dataset CSV path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def time_split(
    df: pd.DataFrame, train_ratio: float, valid_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if TIME_COL not in df.columns:
        raise ValueError(f"{TIME_COL} is missing from the dataset")

    df_sorted = df.sort_values(TIME_COL).reset_index(drop=True)
    total = len(df_sorted)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))

    train = df_sorted.iloc[:train_end].copy()
    valid = df_sorted.iloc[train_end:valid_end].copy()
    test = df_sorted.iloc[valid_end:].copy()

    return train, valid, test


def split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError("Target column isFraud not found in dataset")

    drop_cols = [ID_COL, "TransactionDate"]
    features = df.drop(columns=[TARGET_COL] + [c for c in drop_cols if c in df.columns])
    target = df[TARGET_COL].astype(int)

    return features, target


def build_imputation_map(
    train: pd.DataFrame, categorical_cols: List[str]
) -> Dict[str, float]:
    numeric_cols = [col for col in train.columns if col not in categorical_cols]
    medians = train[numeric_cols].median(numeric_only=True).to_dict()
    for col, value in medians.items():
        if pd.isna(value):
            medians[col] = 0.0
    return medians


def apply_imputation(
    df: pd.DataFrame, numeric_map: Dict[str, float], categorical_cols: List[str]
) -> pd.DataFrame:
    filled = df.copy()
    for col, median in numeric_map.items():
        filled[col] = filled[col].fillna(median)
    for col in categorical_cols:
        filled[col] = filled[col].fillna("missing").astype(str)
    return filled


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


def main() -> None:
    args = parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_path}")

    df = pd.read_csv(args.input_path, low_memory=False)

    if args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed)

    train_df, valid_df, test_df = time_split(df, args.train_ratio, args.valid_ratio)

    x_train, y_train = split_features(train_df)
    x_valid, y_valid = split_features(valid_df)
    x_test, y_test = split_features(test_df)

    categorical_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_map = build_imputation_map(x_train, categorical_cols)

    x_train = apply_imputation(x_train, numeric_map, categorical_cols)
    x_valid = apply_imputation(x_valid, numeric_map, categorical_cols)
    x_test = apply_imputation(x_test, numeric_map, categorical_cols)

    train_pool = Pool(x_train, y_train, cat_features=categorical_cols)
    valid_pool = Pool(x_valid, y_valid, cat_features=categorical_cols)
    test_pool = Pool(x_test, y_test, cat_features=categorical_cols)

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.seed,
        verbose=False,
    )

    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50)

    valid_scores = model.predict_proba(valid_pool)[:, 1]
    test_scores = model.predict_proba(test_pool)[:, 1]

    valid_auc = roc_auc_score(y_valid, valid_scores)
    test_auc = roc_auc_score(y_test, test_scores)

    valid_pr_auc = average_precision_score(y_valid, valid_scores)
    test_pr_auc = average_precision_score(y_test, test_scores)

    threshold, best_f1 = find_best_threshold(y_valid.to_numpy(), valid_scores)

    model_version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = args.out_dir / "model.joblib"
    bundle = {
        "model": model,
        "features": x_train.columns.tolist(),
        "categorical_features": categorical_cols,
        "threshold": threshold,
        "model_version": model_version,
    }

    joblib.dump(bundle, artifact_path)

    metrics = {
        "valid_auc": float(valid_auc),
        "test_auc": float(test_auc),
        "valid_pr_auc": float(valid_pr_auc),
        "test_pr_auc": float(test_pr_auc),
        "threshold": float(threshold),
        "best_f1": float(best_f1),
        "rows": {
            "train": int(len(x_train)),
            "valid": int(len(x_valid)),
            "test": int(len(x_test)),
        },
        "fraud_rate": {
            "train": float(y_train.mean()),
            "valid": float(y_valid.mean()),
            "test": float(y_test.mean()),
        },
        "model_version": model_version,
        "input_path": str(args.input_path),
    }

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved model to {artifact_path}")
    print(f"Metrics saved to {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
