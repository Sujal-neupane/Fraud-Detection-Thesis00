import os
from pathlib import Path

APP_NAME = os.getenv("APP_NAME", "fraud-detection")
BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = os.getenv("MODEL_PATH", str(BACKEND_DIR / "artifacts" / "model.joblib"))
METRICS_PATH = os.getenv("METRICS_PATH", str(BACKEND_DIR / "artifacts" / "metrics.json"))
ANALYSIS_PATH = os.getenv("ANALYSIS_PATH", str(BACKEND_DIR / "artifacts" / "analysis.json"))
