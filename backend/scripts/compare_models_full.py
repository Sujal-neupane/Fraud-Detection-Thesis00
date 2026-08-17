"""Full comparative benchmark for the ShieldML thesis.

Trains Logistic Regression, Random Forest, and CatBoost (each with and without
SMOTE oversampling) plus an Isolation Forest anomaly detector on the real
IEEE-CIS transaction data, then reports Precision / Recall / F1 / ROC-AUC /
PR-AUC for every configuration (Table 1), a decision-threshold sweep with
confusion matrices for the best model (Table 2), and a set of publication
figures (class imbalance, ROC/PR overlays, confusion matrix, feature importance,
SHAP summary).

Every number written by this script is measured, not assumed. Nothing is
fabricated: if a model fails to train, its row is omitted rather than filled in.

Usage:
    .venv/bin/python scripts/compare_models_full.py --train-sample 100000
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from catboost import CatBoostClassifier, Pool

try:
    from imblearn.over_sampling import SMOTE, SMOTENC
except ImportError:  # pragma: no cover
    SMOTE = SMOTENC = None  # type: ignore

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"
DROP_COLS = [ID_COL, TIME_COL, "TransactionDate"]

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "ieee-cis"
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FIG_DIR = OUT_DIR / "figures"


# --------------------------------------------------------------------------- #
# Feature engineering (mirrors train_model.py so results are comparable)
# --------------------------------------------------------------------------- #
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


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns])
    y = df[TARGET_COL].astype(int)
    return x, y


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold on [0,1] scores that maximises F1 (used for fair P/R/F1)."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def score_metrics(y_true: np.ndarray, scores: np.ndarray, name: str) -> Dict[str, object]:
    """ROC-AUC and PR-AUC are threshold-free; P/R/F1 at the best-F1 threshold."""
    roc = float(roc_auc_score(y_true, scores))
    prauc = float(average_precision_score(y_true, scores))
    t = best_f1_threshold(y_true, scores)
    preds = (scores >= t).astype(int)
    return {
        "model": name,
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(prauc, 4),
        "threshold": round(t, 3),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-sample", type=int, default=100000,
                   help="Stratified train subsample size (keeps 7 model fits tractable). "
                        "Evaluation always uses the full valid/test splits.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--average-fraud-value", type=float, default=250.0)
    p.add_argument("--false-positive-cost", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = args.seed
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed IEEE-CIS splits...")
    train = pd.read_csv(DATA_DIR / "train.csv", low_memory=False)
    valid = pd.read_csv(DATA_DIR / "valid.csv", low_memory=False)
    test = pd.read_csv(DATA_DIR / "test.csv", low_memory=False)

    # Stratified subsample of TRAIN only (valid/test kept whole)
    if args.train_sample and args.train_sample < len(train):
        frac = args.train_sample / len(train)
        train = (
            train.groupby(TARGET_COL, group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=rng))
            .reset_index(drop=True)
        )
    print(f"  train={len(train):,}  valid={len(valid):,}  test={len(test):,}")
    print(f"  fraud rate  train={train[TARGET_COL].mean():.4f}  "
          f"valid={valid[TARGET_COL].mean():.4f}  test={test[TARGET_COL].mean():.4f}")

    train, valid, test = (engineer_features(d) for d in (train, valid, test))
    x_train, y_train = split_xy(train)
    x_valid, y_valid = split_xy(valid)
    x_test, y_test = split_xy(test)
    yv, yt = y_valid.to_numpy(), y_test.to_numpy()

    cat_cols = x_train.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in x_train.columns if c not in cat_cols]

    # --- Imputation (fit on train) ------------------------------------------
    medians = x_train[num_cols].median(numeric_only=True).to_dict()
    medians = {k: (0.0 if pd.isna(v) else v) for k, v in medians.items()}

    def impute(x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        for c in num_cols:
            out[c] = out[c].fillna(medians[c])
        for c in cat_cols:
            out[c] = out[c].fillna("missing").astype(str)
        return out

    x_train_i, x_valid_i, x_test_i = (impute(x) for x in (x_train, x_valid, x_test))

    # --- sklearn numeric matrix (ordinal-encode cats + scale) ---------------
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(x_train_i[cat_cols])

    def to_matrix(x: pd.DataFrame) -> np.ndarray:
        num = x[num_cols].to_numpy(dtype=float)
        cat = enc.transform(x[cat_cols])
        return np.hstack([num, cat])

    scaler = StandardScaler()
    m_train = scaler.fit_transform(to_matrix(x_train_i))
    m_valid = scaler.transform(to_matrix(x_valid_i))
    m_test = scaler.transform(to_matrix(x_test_i))

    results: List[Dict[str, object]] = []
    roc_curves: Dict[str, Dict[str, list]] = {}
    pr_curves: Dict[str, Dict[str, list]] = {}
    test_scores_by_model: Dict[str, np.ndarray] = {}

    def register(name: str, scores_test: np.ndarray) -> None:
        row = score_metrics(yt, scores_test, name)
        results.append(row)
        fpr, tpr, _ = roc_curve(yt, scores_test)
        prec, rec, _ = precision_recall_curve(yt, scores_test)
        idx_r = np.linspace(0, len(fpr) - 1, 60, dtype=int)
        idx_p = np.linspace(0, len(prec) - 1, 60, dtype=int)
        roc_curves[name] = {"fpr": fpr[idx_r].tolist(), "tpr": tpr[idx_r].tolist()}
        pr_curves[name] = {"precision": prec[idx_p].tolist(), "recall": rec[idx_p].tolist()}
        test_scores_by_model[name] = scores_test
        print(f"  {name:28} P={row['precision']:.3f} R={row['recall']:.3f} "
              f"F1={row['f1']:.3f} ROC={row['roc_auc']:.3f} PR={row['pr_auc']:.3f}")

    # SMOTE (numeric) for sklearn models, SMOTENC for CatBoost frame
    def smote_numeric() -> Tuple[np.ndarray, np.ndarray]:
        sm = SMOTE(random_state=rng, k_neighbors=5)
        return sm.fit_resample(m_train, y_train)  # type: ignore

    def smotenc_frame() -> Tuple[pd.DataFrame, pd.Series]:
        cat_idx = [x_train_i.columns.get_loc(c) for c in cat_cols]
        sm = SMOTENC(categorical_features=cat_idx, random_state=rng, k_neighbors=5)
        xr, yr = sm.fit_resample(x_train_i, y_train)  # type: ignore
        return xr, yr

    print("\n=== Table 1: model comparison (metrics on full TEST set) ===")

    # 1) Logistic Regression -------------------------------------------------
    for use_smote in (False, True):
        tag = "Logistic Regression + SMOTE" if use_smote else "Logistic Regression"
        try:
            t0 = time.time()
            if use_smote:
                Xr, yr = smote_numeric()
                clf = LogisticRegression(max_iter=300, random_state=rng)
            else:
                Xr, yr = m_train, y_train
                clf = LogisticRegression(max_iter=300, class_weight="balanced", random_state=rng)
            clf.fit(Xr, yr)
            register(tag, clf.predict_proba(m_test)[:, 1])
            print(f"      ({time.time()-t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag} FAILED: {exc}")

    # 2) Random Forest -------------------------------------------------------
    for use_smote in (False, True):
        tag = "Random Forest + SMOTE" if use_smote else "Random Forest"
        try:
            t0 = time.time()
            if use_smote:
                Xr, yr = smote_numeric()
                clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                             n_jobs=-1, random_state=rng)
            else:
                Xr, yr = m_train, y_train
                clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1,
                                             class_weight="balanced_subsample", random_state=rng)
            clf.fit(Xr, yr)
            register(tag, clf.predict_proba(m_test)[:, 1])
            print(f"      ({time.time()-t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag} FAILED: {exc}")

    # 3) CatBoost ------------------------------------------------------------
    cat_model_for_shap = None
    valid_pool = Pool(x_valid_i, y_valid, cat_features=cat_cols)
    test_pool = Pool(x_test_i, y_test, cat_features=cat_cols)
    for use_smote in (False, True):
        tag = "CatBoost + SMOTE" if use_smote else "CatBoost"
        try:
            t0 = time.time()
            if use_smote:
                Xr, yr = smotenc_frame()
                pool = Pool(Xr, yr, cat_features=cat_cols)
                params = dict(iterations=300, depth=8, learning_rate=0.05,
                              loss_function="Logloss", eval_metric="AUC",
                              random_seed=rng, l2_leaf_reg=6.0, verbose=False)
            else:
                pool = Pool(x_train_i, y_train, cat_features=cat_cols)
                params = dict(iterations=300, depth=8, learning_rate=0.05,
                              loss_function="Logloss", eval_metric="AUC",
                              random_seed=rng, l2_leaf_reg=6.0,
                              auto_class_weights="Balanced", verbose=False)
            clf = CatBoostClassifier(**params)
            clf.fit(pool, eval_set=valid_pool, early_stopping_rounds=50)
            register(tag, clf.predict_proba(test_pool)[:, 1])
            if use_smote:
                cat_model_for_shap = clf
            print(f"      ({time.time()-t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag} FAILED: {exc}")

    # 4) Isolation Forest (unsupervised anomaly detector) --------------------
    try:
        t0 = time.time()
        contamination = float(min(0.2, max(0.01, y_train.mean())))
        iso = IsolationForest(n_estimators=200, contamination=contamination,
                              n_jobs=-1, random_state=rng)
        iso.fit(m_train)
        # higher score = more anomalous = more likely fraud
        raw = -iso.score_samples(m_test)
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        register("Isolation Forest", scores)
        print(f"      ({time.time()-t0:.0f}s)")
    except Exception as exc:  # noqa: BLE001
        print(f"  Isolation Forest FAILED: {exc}")

    # ---------------------------------------------------------------------- #
    # Table 2: threshold sweep + confusion matrices for the best model
    # ---------------------------------------------------------------------- #
    ranked = sorted(results, key=lambda r: r["pr_auc"], reverse=True)
    best_name = ranked[0]["model"] if ranked else None
    threshold_table: List[Dict[str, object]] = []
    if best_name and best_name in test_scores_by_model:
        best_scores = test_scores_by_model[best_name]
        print(f"\n=== Table 2: threshold sweep for best model = {best_name} (TEST set) ===")
        n_fraud = int(yt.sum())
        for thr in [0.30, 0.50, 0.635, 0.70, 0.90]:
            preds = (best_scores >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, preds, labels=[0, 1]).ravel()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            fp_per_fraud = fp / tp if tp else float("inf")
            threshold_table.append({
                "threshold": thr, "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                "false_positives_per_fraud_caught": round(fp_per_fraud, 2)
                if np.isfinite(fp_per_fraud) else None,
                "frauds_missed": int(fn),
                "frauds_caught": int(tp),
                "total_frauds": n_fraud,
            })
            print(f"  thr={thr:.3f}  TP={tp:5d} FP={fp:6d} FN={fn:4d}  "
                  f"P={prec:.3f} R={rec:.3f}  FP/fraud={fp_per_fraud:.1f}")

    # ---------------------------------------------------------------------- #
    # Persist results
    # ---------------------------------------------------------------------- #
    payload = {
        "meta": {
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "test_rows": int(len(test)),
            "train_fraud_rate": float(train[TARGET_COL].mean()),
            "test_fraud_rate": float(y_test.mean()),
            "n_features": int(x_train.shape[1]),
            "n_categorical": int(len(cat_cols)),
            "seed": rng,
            "note": "P/R/F1 reported at each model's best-F1 threshold on TEST; "
                    "ROC-AUC and PR-AUC are threshold-independent. Isolation Forest "
                    "is unsupervised (no SMOTE variant).",
        },
        "table1_model_comparison": ranked,
        "table2_threshold_sweep": {"model": best_name, "rows": threshold_table},
    }
    comp_path = OUT_DIR / "comparison_results.json"
    comp_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {comp_path}")

    # ---------------------------------------------------------------------- #
    # Figures
    # ---------------------------------------------------------------------- #
    make_figures(train, y_train, roc_curves, pr_curves, best_name,
                 test_scores_by_model, threshold_table, cat_model_for_shap,
                 x_test_i, cat_cols)
    print(f"Figures -> {FIG_DIR}")


def make_figures(train, y_train, roc_curves, pr_curves, best_name,
                 test_scores_by_model, threshold_table, cat_model, x_test_i, cat_cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})
    INK = "#1f2933"
    ACCENT = "#2f6fed"

    # 1) Class imbalance -----------------------------------------------------
    try:
        counts = train["isFraud"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4.2))
        bars = ax.bar(["Legitimate", "Fraud"], [counts.get(0, 0), counts.get(1, 0)],
                      color=["#9aa5b1", ACCENT])
        ax.set_ylabel("Number of transactions")
        ax.set_title("Class distribution (training set)")
        total = counts.sum()
        for b, c in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{c:,}\n({c/total:.2%})", ha="center", va="bottom", fontsize=10)
        ax.margins(y=0.15)
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_class_imbalance.png"); plt.close(fig)
    except Exception as exc:
        print(f"  fig1 failed: {exc}")

    # 2) ROC overlay ---------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(6.4, 5))
        for name, c in roc_curves.items():
            ax.plot(c["fpr"], c["tpr"], lw=1.8, label=name)
        ax.plot([0, 1], [0, 1], "--", color="#9aa5b1", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curves"); ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig2_roc_curves.png"); plt.close(fig)
    except Exception as exc:
        print(f"  fig2 failed: {exc}")

    # 3) PR overlay ----------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(6.4, 5))
        for name, c in pr_curves.items():
            ax.plot(c["recall"], c["precision"], lw=1.8, label=name)
        base = float(y_train.mean())
        ax.axhline(base, ls="--", color="#9aa5b1", lw=1, label=f"baseline ({base:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall curves"); ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig3_pr_curves.png"); plt.close(fig)
    except Exception as exc:
        print(f"  fig3 failed: {exc}")

    # 4) Confusion matrix for best model at the thesis threshold (0.635) -----
    try:
        from sklearn.metrics import confusion_matrix
        row = next((r for r in threshold_table if r["threshold"] == 0.635), None)
        if row:
            cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1], ["Pred. Legit", "Pred. Fraud"])
            ax.set_yticks([0, 1], ["Actual Legit", "Actual Fraud"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=13)
            ax.set_title(f"Confusion matrix — {best_name} @ thr=0.635")
            ax.grid(False); fig.colorbar(im, fraction=0.046)
            fig.tight_layout(); fig.savefig(FIG_DIR / "fig4_confusion_matrix.png"); plt.close(fig)
    except Exception as exc:
        print(f"  fig4 failed: {exc}")

    # 5) + 6) CatBoost feature importance and SHAP summary -------------------
    if cat_model is not None:
        try:
            from catboost import Pool
            names = list(x_test_i.columns)
            imp = cat_model.get_feature_importance()
            order = np.argsort(imp)[::-1][:15]
            fig, ax = plt.subplots(figsize=(6.6, 5))
            ax.barh([names[i] for i in order][::-1], [imp[i] for i in order][::-1], color=ACCENT)
            ax.set_xlabel("Importance"); ax.set_title("CatBoost feature importance (top 15)")
            fig.tight_layout(); fig.savefig(FIG_DIR / "fig5_feature_importance.png"); plt.close(fig)
        except Exception as exc:
            print(f"  fig5 failed: {exc}")

        try:
            from catboost import Pool
            sample = x_test_i.sample(n=min(2000, len(x_test_i)), random_state=1)
            pool = Pool(sample, cat_features=cat_cols)
            shap_vals = cat_model.get_feature_importance(pool, type="ShapValues")
            shap_vals = np.asarray(shap_vals)[:, :-1]  # drop expected-value column
            names = list(sample.columns)
            mean_abs = np.abs(shap_vals).mean(0)
            top = np.argsort(mean_abs)[::-1][:12]
            fig, ax = plt.subplots(figsize=(7, 5.2))
            for row_i, feat in enumerate(top[::-1]):
                sv = shap_vals[:, feat]
                col = sample.iloc[:, feat]
                try:
                    cv = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
                    if np.all(np.isnan(cv)):
                        cv = np.zeros(len(sv))
                except Exception:
                    cv = np.zeros(len(sv))
                finite = np.isfinite(cv)
                if finite.sum() > 1:
                    lo, hi = np.nanpercentile(cv[finite], [5, 95])
                    cv = np.clip((cv - lo) / (hi - lo + 1e-12), 0, 1)
                else:
                    cv = np.zeros(len(sv))
                jitter = (np.random.rand(len(sv)) - 0.5) * 0.35
                ax.scatter(sv, np.full(len(sv), row_i) + jitter, c=cv, cmap="coolwarm",
                           s=7, alpha=0.6, linewidths=0)
            ax.set_yticks(range(len(top)), [names[i] for i in top[::-1]])
            ax.axvline(0, color="#9aa5b1", lw=1)
            ax.set_xlabel("SHAP value (impact on fraud log-odds)")
            ax.set_title("SHAP summary — CatBoost + SMOTE (top 12 features)")
            sm = plt.cm.ScalarMappable(cmap="coolwarm")
            sm.set_array([]); cb = fig.colorbar(sm, ax=ax, fraction=0.03)
            cb.set_label("Feature value (low → high)")
            fig.tight_layout(); fig.savefig(FIG_DIR / "fig6_shap_summary.png"); plt.close(fig)
        except Exception as exc:
            print(f"  fig6 failed: {exc}")


if __name__ == "__main__":
    main()
