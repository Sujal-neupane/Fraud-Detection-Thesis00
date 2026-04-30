from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

ID_COL = "TransactionID"
TIME_COL = "TransactionDT"
TARGET_COL = "isFraud"


def load_merged(raw_dir: Path, split: str) -> pd.DataFrame:
    transaction_path = raw_dir / f"{split}_transaction.csv"
    identity_path = raw_dir / f"{split}_identity.csv"

    if not transaction_path.exists():
        raise FileNotFoundError(f"Missing file: {transaction_path}")

    transactions = pd.read_csv(transaction_path)

    if identity_path.exists():
        identity = pd.read_csv(identity_path)
        merged = transactions.merge(identity, on=ID_COL, how="left")
    else:
        merged = transactions

    return merged


def time_split(
    df: pd.DataFrame, train_ratio: float, valid_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be < 1")

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


def build_imputation_map(df: pd.DataFrame) -> Tuple[Dict[str, float], list[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != TARGET_COL]

    medians = df[numeric_cols].median(numeric_only=True).to_dict()
    for col, value in medians.items():
        if pd.isna(value):
            medians[col] = 0.0

    categorical_cols = [
        col for col in df.columns if col not in numeric_cols and col != TARGET_COL
    ]

    return medians, categorical_cols


def apply_imputation(
    df: pd.DataFrame, numeric_map: Dict[str, float], categorical_cols: list[str]
) -> pd.DataFrame:
    filled = df.copy()

    for col, median in numeric_map.items():
        filled[col] = filled[col].fillna(median)

    for col in categorical_cols:
        filled[col] = filled[col].fillna("missing").astype(str)

    return filled


def fraud_rate(df: pd.DataFrame) -> float:
    if TARGET_COL not in df.columns:
        return 0.0
    return float(df[TARGET_COL].mean())


def save_outputs(
    out_dir: Path,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    imputation_map: Dict[str, float],
    categorical_cols: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(out_dir / "train.csv", index=False)
    valid.to_csv(out_dir / "valid.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    def to_native(value: object) -> object:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()
        return value

    metadata = {
        "rows": {
            "total": int(len(train) + len(valid) + len(test)),
            "train": int(len(train)),
            "valid": int(len(valid)),
            "test": int(len(test)),
        },
        "fraud_rate": {
            "train": float(fraud_rate(train)),
            "valid": float(fraud_rate(valid)),
            "test": float(fraud_rate(test)),
        },
        "time_range": {
            "train": [
                to_native(train[TIME_COL].min()),
                to_native(train[TIME_COL].max()),
            ],
            "valid": [
                to_native(valid[TIME_COL].min()),
                to_native(valid[TIME_COL].max()),
            ],
            "test": [
                to_native(test[TIME_COL].min()),
                to_native(test[TIME_COL].max()),
            ],
        },
        "columns": {
            "total": int(len(train.columns)),
            "categorical": int(len(categorical_cols)),
            "numeric": int(len(imputation_map)),
        },
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with (out_dir / "imputation.json").open("w", encoding="utf-8") as file:
        json.dump({"numeric": imputation_map, "categorical": categorical_cols}, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare IEEE-CIS dataset with a time-based split."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "ieee-cis",
        help="Path to raw IEEE-CIS CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "ieee-cis",
        help="Output directory for processed CSV files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading raw training data...")
    merged = load_merged(args.raw_dir, "train")

    if TARGET_COL not in merged.columns:
        raise ValueError("Target column isFraud not found in training data.")

    print("Splitting data by time...")
    train, valid, test = time_split(merged, args.train_ratio, args.valid_ratio)

    print("Building imputation map from training split...")
    numeric_map, categorical_cols = build_imputation_map(train)

    print("Applying imputations...")
    train = apply_imputation(train, numeric_map, categorical_cols)
    valid = apply_imputation(valid, numeric_map, categorical_cols)
    test = apply_imputation(test, numeric_map, categorical_cols)

    print("Saving outputs...")
    save_outputs(args.out_dir, train, valid, test, numeric_map, categorical_cols)

    print(f"Done. Files written to: {args.out_dir}")


if __name__ == "__main__":
    main()
