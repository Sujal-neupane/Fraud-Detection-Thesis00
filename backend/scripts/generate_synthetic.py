from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic IEEE-CIS-like data with 2019-2025 timestamps."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "ieee-cis"
        / "train.csv",
        help="Source CSV for distribution sampling.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "synthetic"
        / "ieee-cis",
        help="Directory to write the synthetic dataset.",
    )
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--start-date", type=str, default="2019-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=None,
        help="Optional fraud ratio override (0-1).",
    )
    return parser.parse_args()


def sample_by_fraud_ratio(
    df: pd.DataFrame, rows: int, fraud_rate: float, rng: np.random.Generator
) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        return df.sample(n=rows, replace=True, random_state=int(rng.integers(0, 1_000_000)))

    fraud = df[df[TARGET_COL] == 1]
    non_fraud = df[df[TARGET_COL] == 0]

    fraud_rows = int(round(rows * fraud_rate))
    non_fraud_rows = rows - fraud_rows

    fraud_sample = fraud.sample(
        n=fraud_rows, replace=True, random_state=int(rng.integers(0, 1_000_000))
    )
    non_fraud_sample = non_fraud.sample(
        n=non_fraud_rows, replace=True, random_state=int(rng.integers(0, 1_000_000))
    )

    combined: pd.DataFrame = pd.concat([fraud_sample, non_fraud_sample], ignore_index=True)  # type: ignore[assignment]
    combined = combined.sample(
        frac=1.0, replace=False, random_state=int(rng.integers(0, 1_000_000))
    ).reset_index(drop=True)

    return combined


def build_time_values(
    source: pd.DataFrame,
    rows: int,
    start_date: str,
    end_date: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if end <= start:
        raise ValueError("end_date must be after start_date")

    total_days = (end - start).days + 1
    random_days = rng.integers(0, total_days, size=rows, endpoint=False)

    if TIME_COL in source.columns:
        seconds_in_day = (source[TIME_COL] % 86_400).to_numpy()
        sampled_seconds = rng.choice(seconds_in_day, size=rows, replace=True)
    else:
        sampled_seconds = rng.integers(0, 86_400, size=rows)

    transaction_dt = random_days * 86_400 + sampled_seconds
    transaction_dates = start + pd.to_timedelta(random_days, unit="D") + pd.to_timedelta(
        sampled_seconds, unit="s"
    )

    return transaction_dt, transaction_dates.to_numpy()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not args.input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_path}")

    df = pd.read_csv(args.input_path, low_memory=False)

    if args.fraud_rate is None:
        fraud_rate = float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else 0.0
    else:
        if not 0 <= args.fraud_rate <= 1:
            raise ValueError("fraud_rate must be between 0 and 1")
        fraud_rate = args.fraud_rate

    sampled = sample_by_fraud_ratio(df, args.rows, fraud_rate, rng)

    transaction_dt, transaction_dates = build_time_values(
        sampled, args.rows, args.start_date, args.end_date, rng
    )

    sampled[ID_COL] = np.arange(1, args.rows + 1)
    sampled[TIME_COL] = transaction_dt
    sampled["TransactionDate"] = pd.to_datetime(transaction_dates).astype(str)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "synthetic_2019_2025.csv"
    sampled.to_csv(output_path, index=False)

    metadata = {
        "rows": args.rows,
        "fraud_rate": fraud_rate,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "source": str(args.input_path),
        "output": str(output_path),
    }

    with (args.out_dir / "synthetic_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Synthetic data saved to: {output_path}")


if __name__ == "__main__":
    main()
