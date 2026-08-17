from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
from scipy import stats

try:
    import optuna
except ImportError:
    optuna = None

try:
    from imblearn.over_sampling import SMOTENC
except ImportError:
    SMOTENC = None


def safe_roc_auc_score(y_true, y_score) -> float:
    try:
        y_true_arr = np.asarray(y_true)
        if len(np.unique(y_true_arr)) < 2:
            return 0.5
        return float(roc_auc_score(y_true_arr, y_score))
    except Exception:
        return 0.5


def safe_average_precision_score(y_true, y_score) -> float:
    try:
        y_true_arr = np.asarray(y_true)
        if len(np.unique(y_true_arr)) < 2:
            return 0.0
        return float(average_precision_score(y_true_arr, y_score))
    except Exception:
        return 0.0


def safe_roc_curve(y_true, y_score) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        y_true_arr = np.asarray(y_true)
        if len(np.unique(y_true_arr)) < 2:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
        return roc_curve(y_true_arr, y_score)
    except Exception:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])


def safe_precision_recall_curve(y_true, y_score) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        y_true_arr = np.asarray(y_true)
        if len(np.unique(y_true_arr)) < 2:
            return np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
        return precision_recall_curve(y_true_arr, y_score)
    except Exception:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TransactionAmt" in out.columns:
        out["LogTransactionAmt"] = np.log1p(out["TransactionAmt"].clip(lower=0))
    if TIME_COL in out.columns:
        out["TxHour"] = (out[TIME_COL].astype(np.int64) % 86400) // 3600
    if "TransactionAmt" in out.columns and "D1" in out.columns:
        d1 = pd.Series(pd.to_numeric(out["D1"], errors="coerce")).fillna(1.0).clip(lower=1.0)
        out["AmtPerDay"] = out["TransactionAmt"] / d1
    return out


def compare_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> List[Dict[str, object]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    results: List[Dict[str, object]] = []

    # Logistic Regression baseline
    try:
        scaler = StandardScaler()
        x_tr_scaled = scaler.fit_transform(x_train.select_dtypes(include="number"))
        x_te_scaled = scaler.transform(x_test.select_dtypes(include="number"))
        t0 = time.time()
        lr = LogisticRegression(class_weight="balanced", max_iter=300, random_state=seed)
        lr.fit(x_tr_scaled, y_train)
        lr_time = time.time() - t0
        lr_scores = lr.predict_proba(x_te_scaled)[:, 1]
        lr_auc = safe_roc_auc_score(y_test, lr_scores)
        lr_prauc = safe_average_precision_score(y_test, lr_scores)
        results.append({"model": "Logistic Regression", "auc": round(lr_auc, 4), "pr_auc": round(lr_prauc, 4), "train_seconds": round(lr_time, 1)})
        print(f"  LR   AUC={lr_auc:.4f}  PR-AUC={lr_prauc:.4f}  ({lr_time:.1f}s)")
    except Exception as exc:
        print(f"  LR failed: {exc}")

    # XGBoost
    try:
        from xgboost import XGBClassifier
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        t0 = time.time()
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos, eval_metric="auc",
            random_state=seed, verbosity=0, use_label_encoder=False,  # type: ignore[call-arg]
        )
        xgb.fit(x_train.select_dtypes(include="number"), y_train)
        xgb_time = time.time() - t0
        xgb_scores = xgb.predict_proba(x_test.select_dtypes(include="number"))[:, 1]
        xgb_auc = safe_roc_auc_score(y_test, xgb_scores)
        xgb_prauc = safe_average_precision_score(y_test, xgb_scores)
        results.append({"model": "XGBoost", "auc": round(xgb_auc, 4), "pr_auc": round(xgb_prauc, 4), "train_seconds": round(xgb_time, 1)})
        print(f"  XGB  AUC={xgb_auc:.4f}  PR-AUC={xgb_prauc:.4f}  ({xgb_time:.1f}s)")
    except Exception as exc:
        print(f"  XGB failed: {exc}")

    return results


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
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTENC for class imbalance")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--experiment", type=str, default="fraud-synthetic")
    parser.add_argument("--average-fraud-value", type=float, default=250.0)
    parser.add_argument("--false-positive-cost", type=float, default=5.0)
    parser.add_argument("--target-precision", type=float, default=0.6)
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "ieee-cis"
        / "train.csv",
        help="Real IEEE-CIS dataset — the training distribution baseline for drift.",
    )
    parser.add_argument(
        "--monitor-path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "synthetic"
        / "ieee-cis"
        / "synthetic_2019_2025.csv",
        help="Dataset representing incoming production traffic (compared against reference for drift).",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Also benchmark XGBoost and Logistic Regression and save a comparison table.",
    )
    parser.add_argument(
        "--native-nan",
        action="store_true",
        help="Bypass numeric median imputation and use CatBoost native NaN handling.",
    )
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

    drop_cols = [ID_COL, TIME_COL, "TransactionDate"]
    features: pd.DataFrame = df.drop(  # type: ignore[assignment]
        columns=[TARGET_COL] + [c for c in drop_cols if c in df.columns]
    )
    target: pd.Series = df[TARGET_COL].astype(int)  # type: ignore[assignment]

    return features, target


def build_imputation_map(
    train: pd.DataFrame, categorical_cols: List[str]
) -> Dict[str, float]:
    numeric_cols = [col for col in train.columns if col not in categorical_cols]
    medians: Dict[str, float] = pd.Series(train[numeric_cols].median(numeric_only=True)).to_dict()
    for col, value in medians.items():
        if pd.isna(value):
            medians[col] = 0.0
    return medians


def apply_imputation(
    df: pd.DataFrame, numeric_map: Dict[str, float], categorical_cols: List[str], impute_numeric: bool = True
) -> pd.DataFrame:
    filled = df.copy()
    if impute_numeric:
        for col, median in numeric_map.items():
            filled[col] = filled[col].fillna(median)
    for col in categorical_cols:
        filled[col] = filled[col].fillna("missing").astype(str)
    return filled


def select_strict_threshold(
    y_true: np.ndarray, scores: np.ndarray, target_precision: float
) -> Tuple[float, float, float, float]:
    # Scan from 0.1 to 0.75; find the threshold that maximises F1
    # while meeting the minimum precision floor (target_precision).
    # Falls back to best-F1 across the full range if nothing clears the floor.
    thresholds = np.linspace(0.10, 0.75, 131)
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_score = -1.0

    fallback_threshold = 0.5
    fallback_f1 = 0.0

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        precision = float(precision_score(y_true, preds, zero_division=0))  # type: ignore[call-overload]
        recall = float(recall_score(y_true, preds, zero_division=0))  # type: ignore[call-overload]
        f1 = float(f1_score(y_true, preds, zero_division=0))  # type: ignore[call-overload]

        if f1 > fallback_f1:
            fallback_threshold = float(threshold)
            fallback_f1 = f1

        if precision >= target_precision and recall >= 0.05:
            if f1 > best_score:
                best_score = f1
                best_threshold = float(threshold)
                best_precision = precision
                best_recall = recall
                best_f1 = f1

    if best_score >= 0:
        return best_threshold, best_precision, best_recall, best_f1

    # Precision floor not met — use fallback that maximises F1 without floor
    preds_fb = (scores >= fallback_threshold).astype(int)
    return (
        fallback_threshold,
        float(precision_score(y_true, preds_fb, zero_division=0)),  # type: ignore[call-overload]
        float(recall_score(y_true, preds_fb, zero_division=0)),  # type: ignore[call-overload]
        float(f1_score(y_true, preds_fb, zero_division=0)),  # type: ignore[call-overload]
    )


def population_stability_index(
    expected: pd.Series, actual: pd.Series, buckets: int = 10
) -> float:
    expected_values: pd.Series = pd.Series(pd.to_numeric(expected, errors="coerce")).dropna()
    actual_values: pd.Series = pd.Series(pd.to_numeric(actual, errors="coerce")).dropna()

    if expected_values.empty or actual_values.empty:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    split_points = np.unique(expected_values.quantile(quantiles).to_numpy())
    if len(split_points) < 3:
        return 0.0

    expected_bins: pd.Series = pd.cut(  # type: ignore[assignment]
        expected_values,
        bins=split_points,
        include_lowest=True,
        duplicates="drop",
    )
    actual_bins: pd.Series = pd.cut(  # type: ignore[assignment]
        actual_values,
        bins=split_points,
        include_lowest=True,
        duplicates="drop",
    )

    expected_dist = expected_bins.value_counts(normalize=True, sort=False)
    actual_dist = actual_bins.value_counts(normalize=True, sort=False)

    aligned = pd.concat([expected_dist, actual_dist], axis=1, keys=["expected", "actual"]).fillna(0.0001)
    expected_pct = aligned["expected"].replace(0, 0.0001)
    actual_pct = aligned["actual"].replace(0, 0.0001)
    return float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())


def compute_ks(expected: pd.Series, actual: pd.Series) -> float:
    expected_values: pd.Series = pd.Series(pd.to_numeric(expected, errors="coerce")).dropna()
    actual_values: pd.Series = pd.Series(pd.to_numeric(actual, errors="coerce")).dropna()
    if expected_values.empty or actual_values.empty:
        return 0.0
    result = stats.ks_2samp(expected_values, actual_values)
    return float(result.statistic)  # type: ignore[union-attr]


def compute_business_impact(
    threshold: float,
    scores: np.ndarray,
    average_fraud_value: float,
    false_positive_cost: float,
) -> Dict[str, float]:
    flagged = scores >= threshold
    flagged_per_10k = float(flagged.mean() * 10000)
    estimated_savings = float(scores.mean() * 10000 * average_fraud_value)
    estimated_friction = float(flagged_per_10k * false_positive_cost)
    return {
        "threshold": float(threshold),
        "flagged_per_10k": flagged_per_10k,
        "average_fraud_value": float(average_fraud_value),
        "false_positive_cost": float(false_positive_cost),
        "estimated_savings": estimated_savings,
        "estimated_friction": estimated_friction,
        "net_benefit": float(estimated_savings - estimated_friction),
    }


def main() -> None:
    args = parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input_path}")

    df = pd.read_csv(args.input_path, low_memory=False)

    if args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed)

    print("Engineering features...")
    df = engineer_features(df)

    train_df, valid_df, test_df = time_split(df, args.train_ratio, args.valid_ratio)

    x_train, y_train = split_features(train_df)
    x_valid, y_valid = split_features(valid_df)
    x_test, y_test = split_features(test_df)

    categorical_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_map = build_imputation_map(x_train, categorical_cols)
    impute_numeric = not args.native_nan
    if not impute_numeric:
        print("Bypassing numeric median imputation. Using CatBoost NATIVE NaN handling.")
    else:
        print("Applying MEDIAN imputation on numeric columns.")

    x_train = apply_imputation(x_train, numeric_map, categorical_cols, impute_numeric=impute_numeric)
    x_valid = apply_imputation(x_valid, numeric_map, categorical_cols, impute_numeric=impute_numeric)
    x_test = apply_imputation(x_test, numeric_map, categorical_cols, impute_numeric=impute_numeric)

    if args.use_smote:
        if SMOTENC is None:
            raise ImportError("imbalanced-learn is required for SMOTENC.")
        print("Applying SMOTENC class oversampling...")
        minority_count = int(y_train.value_counts().min())
        if minority_count < 2:
            print(f"Skipping SMOTENC: minority class has only {minority_count} sample(s), need at least 2.")
        else:
            k_neighbors = min(5, minority_count - 1)
            cat_indices = [x_train.columns.get_loc(c) for c in categorical_cols]
            smote = SMOTENC(categorical_features=cat_indices, random_state=args.seed, k_neighbors=k_neighbors)
            x_train, y_train = smote.fit_resample(x_train, y_train)  # type: ignore[assignment]
            print(f"SMOTENC applied with k_neighbors={k_neighbors}. New train size: {len(x_train)}")

    train_pool = Pool(x_train, y_train, cat_features=categorical_cols)
    valid_pool = Pool(x_valid, y_valid, cat_features=categorical_cols)
    test_pool = Pool(x_test, y_test, cat_features=categorical_cols)

    best_params = {
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": args.seed,
        "auto_class_weights": "Balanced",
        "l2_leaf_reg": 6.0,
        "random_strength": 1.5,
        "verbose": False,
    }

    if args.tune:
        if optuna is None:
            raise ImportError("optuna is required for hyperparameter tuning. Run pip install optuna")
        print("Running Optuna hyperparameter tuning...")
        
        def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
            param = {
                "iterations": trial.suggest_int("iterations", 100, 300),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": args.seed,
                "auto_class_weights": "Balanced",
                "verbose": False,
            }
            model_opt = CatBoostClassifier(**param)
            model_opt.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=20)
            preds = model_opt.predict_proba(valid_pool)[:, 1]
            return safe_roc_auc_score(y_valid, preds)


        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)
        print("Best Optuna params:", study.best_params)
        best_params.update(study.best_params)

    model = CatBoostClassifier(**best_params)

    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50)

    valid_scores = model.predict_proba(valid_pool)[:, 1]
    test_scores = model.predict_proba(test_pool)[:, 1]

    valid_auc = safe_roc_auc_score(y_valid, valid_scores)
    test_auc = safe_roc_auc_score(y_test, test_scores)

    valid_pr_auc = safe_average_precision_score(y_valid, valid_scores)
    test_pr_auc = safe_average_precision_score(y_test, test_scores)

    threshold, selected_precision, selected_recall, best_f1 = select_strict_threshold(
        y_valid.to_numpy(), valid_scores, args.target_precision
    )
    business_impact = compute_business_impact(
        threshold,
        valid_scores,
        args.average_fraud_value,
        args.false_positive_cost,
    )

    if not args.reference_path.exists():
        raise FileNotFoundError(f"Missing reference CSV for drift analysis: {args.reference_path}")

    reference_df = pd.read_csv(args.reference_path, low_memory=False)

    # Monitor path = synthetic/production data; reference = real training distribution.
    # Comparing the same dataset against itself would yield PSI ≈ 0 everywhere.
    if args.monitor_path.exists():
        monitor_df = pd.read_csv(args.monitor_path, low_memory=False)
        print(f"Drift: comparing reference ({len(reference_df):,} rows) vs monitor ({len(monitor_df):,} rows).")
    else:
        monitor_df = df
        print("Drift: monitor-path not found, falling back to training data.")

    drift_features = [
        col
        for col in monitor_df.columns
        if col in reference_df.columns and pd.api.types.is_numeric_dtype(monitor_df[col])
    ]
    drift_summary = []
    if drift_features:
        for feature in drift_features[:25]:
            drift_summary.append(
                {
                    "feature": feature,
                    "ks": compute_ks(pd.Series(reference_df[feature]), pd.Series(monitor_df[feature])),
                    "psi": population_stability_index(pd.Series(reference_df[feature]), pd.Series(monitor_df[feature])),
                }
            )

    # Model comparison benchmark
    comparison_results: List[Dict[str, object]] = []
    if args.compare_models:
        print("\nRunning model comparison (XGBoost + Logistic Regression)...")
        comparison_results = compare_models(
            pd.DataFrame(x_train), pd.Series(y_train), x_test, y_test, args.seed
        )
        catboost_entry: Dict[str, object] = {
            "model": "CatBoost (ours)",
            "auc": round(test_auc, 4),
            "pr_auc": round(test_pr_auc, 4),
            "train_seconds": None,
        }
        comparison_results.append(catboost_entry)
        print(f"  CAT  AUC={test_auc:.4f}  PR-AUC={test_pr_auc:.4f}")

    model_version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = args.out_dir / "model.joblib"
    bundle = {
        "model": model,
        "features": x_train.columns.tolist(),
        "categorical_features": categorical_cols,
        "threshold": threshold,
        "model_version": model_version,
        "native_nan": args.native_nan,
    }

    joblib.dump(bundle, artifact_path)

    # Calculate ROC and Precision-Recall Curves for test set
    fpr, tpr, _ = safe_roc_curve(y_test, test_scores)
    precision, recall, _ = safe_precision_recall_curve(y_test, test_scores)
    
    # Downsample curves to 30 points to keep the metrics JSON compact
    indices_roc = np.linspace(0, len(fpr) - 1, 30, dtype=int)
    roc_points = [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in indices_roc]
    
    indices_pr = np.linspace(0, len(precision) - 1, 30, dtype=int)
    pr_points = [{"precision": float(precision[i]), "recall": float(recall[i])} for i in indices_pr]
    
    # Global Feature Importance
    importances: list[float] = list(model.get_feature_importance())  # type: ignore[arg-type]
    feature_names = x_train.columns.tolist()
    global_importance = sorted(
        [{"feature": f, "importance": float(imp)} for f, imp in zip(feature_names, importances)],
        key=lambda item: item["importance"],
        reverse=True,
    )[:10]

    metrics = {
        "valid_auc": float(valid_auc),
        "test_auc": float(test_auc),
        "valid_pr_auc": float(valid_pr_auc),
        "test_pr_auc": float(test_pr_auc),
        "threshold": float(threshold),
        "selected_precision": float(selected_precision),
        "selected_recall": float(selected_recall),
        "best_f1": float(best_f1),
        "business_impact": business_impact,
        "drift_summary": drift_summary,
        "roc_curve": roc_points,
        "pr_curve": pr_points,
        "global_importance": global_importance,
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
        "model_comparison": comparison_results if comparison_results else [],
    }

    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    mlruns_dir = Path(__file__).resolve().parents[1] / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlruns_dir}/mlflow.db")
    mlflow.set_experiment(args.experiment)

    run_name = f"catboost_{model_version}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "input_path": str(args.input_path),
                "train_ratio": args.train_ratio,
                "valid_ratio": args.valid_ratio,
                "seed": args.seed,
                "iterations": args.iterations,
                "depth": args.depth,
                "learning_rate": args.learning_rate,
                "max_rows": args.max_rows,
                "fraud_rate_train": float(y_train.mean()),
            }
        )
        mlflow.log_metrics(
            {
                "valid_auc": float(valid_auc),
                "test_auc": float(test_auc),
                "valid_pr_auc": float(valid_pr_auc),
                "test_pr_auc": float(test_pr_auc),
                "threshold": float(threshold),
                "selected_precision": float(selected_precision),
                "selected_recall": float(selected_recall),
                "best_f1": float(best_f1),
                "business_impact_flagged_per_10k": float(
                    business_impact["flagged_per_10k"]
                ),
                "business_impact_estimated_savings": float(
                    business_impact["estimated_savings"]
                ),
                "business_impact_estimated_friction": float(
                    business_impact["estimated_friction"]
                ),
                "business_impact_net_benefit": float(business_impact["net_benefit"]),
            }
        )
        mlflow.log_artifact(str(artifact_path))
        mlflow.log_artifact(str(metrics_path))

    analysis_path = args.out_dir / "analysis.json"
    with analysis_path.open("w", encoding="utf-8") as file:
        json.dump({"business_impact": business_impact, "drift_summary": drift_summary}, file, indent=2)
    mlflow.log_artifact(str(analysis_path))

    if comparison_results:
        comp_path = args.out_dir / "model_comparison.json"
        with comp_path.open("w", encoding="utf-8") as file:
            json.dump({"models": comparison_results}, file, indent=2)
        print(f"Model comparison saved to {comp_path}")

    print(f"\n{'='*50}")
    print(f"  Test AUC:       {test_auc:.4f}")
    print(f"  Test PR-AUC:    {test_pr_auc:.4f}")
    print(f"  Threshold:      {threshold:.3f}")
    print(f"  Precision:      {selected_precision:.1%}")
    print(f"  Recall:         {selected_recall:.1%}")
    print(f"  F1:             {best_f1:.4f}")
    print(f"  Net benefit:    ${business_impact['net_benefit']:,.0f}/10k transactions")
    print(f"{'='*50}")
    print(f"Saved model to {artifact_path}")
    print(f"Metrics saved to {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
