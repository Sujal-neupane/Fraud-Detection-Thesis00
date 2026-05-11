# Fraud Detection Thesis Backend

This backend application powers the "Fraud Intelligence Command Center." It is built primarily using Python, FastAPI, and CatBoost, and uses state-of-the-art machine learning practices designed to provide a highly interpretable and accurate system.

## Machine Learning Architecture

The AI module is built heavily for explainability and cost-sensitive business performance rather than purely raw academic metrics, bridging the gap between Data Science and System Architecture.

### 1. Advanced Hyperparameter Tuning (Optuna)
Instead of manually guessing parameter configurations, the pipeline leverages **Optuna** (a hyperparameter optimization framework).
*   During training, Optuna performs Bayesian Optimization to iteratively select target permutations across `learning_rate`, `tree depth`, `l2_leaf_reg`, and `random_strength`.
*   This automatically discovers the mathematically optimal parameters that maximize the models AUC before returning the final configuration.

### 2. Handling Class Imbalance (SMOTENC)
Real-world fraud data is inherently heavily imbalanced (e.g., 97% legitimate, 3% fraud). If not handled, models inevitably collapse into predicting "Not Fraud" for everything.
*   We utilize `auto_class_weights="Balanced"` within the framework.
*   More importantly, we leverage **SMOTENC** (Synthetic Minority Over-sampling Technique for Nominal and Continuous data). 
*   During the training pipeline, SMOTENC synthetically generates artificial positive examples of fraud transactions *inside* the minority space, utilizing nearest-neighbor boundaries to synthetically balance the dataset before the model begins building classification trees.

### 3. Sparse Input Calibration
Real-world application inputs rarely match complete datasets. Often, a user will submit a subset of data (a "sparse" payload). 
*   The model blends the raw CatBoost classification risk alongside a deterministic sparse-signal weight heuristic so that the score realistically captures risk flags such as extreme `TransactionAmt` or isolated missing variables, maintaining strong separation thresholds.

### 4. Interpretable AI (SHAP & Business Impact)
The backend does not just output a flat boolean prediction. The FastAPI `predict` endpoint incorporates two massive layers of observability:
1.  **SHAP (SHapley Additive exPlanations):** Automatically parses the underlying gradient boosting decision structures to extract exactly *why* a decision was made. The `model_service.py` evaluates the local feature importance to identify the exact numerical influence of properties (e.g., "Feature C14 increased the risk by +41%").
2.  **Cost-Sensitive Business Logic:** By incorporating typical fraud-loss vs. friction-cost matrices, the backend dynamically calculates the "Net Benefit" in dollars derived from the specific input evaluation.

## Running the API

1. Navigate to this `backend` directory.
2. Activate your Virtual Environment: `source .venv/bin/activate`.
3. Start the FastAPI server: `python -m uvicorn app.main:app --reload --port 8000`.

## Retraining the Model

If deploying fresh data or tuning the implementation:

```bash
python scripts/train_model.py --tune --n-trials 15 --use-smote
```
*   `--tune`: Enables Optuna Bayesian Optimization.
*   `--n-trials`: Number of hyperparameter guesses to iterate over.
*   `--use-smote`: Enables SMOTENC for synthetic class balance logic.