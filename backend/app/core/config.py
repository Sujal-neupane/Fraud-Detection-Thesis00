import os

APP_NAME = os.getenv("APP_NAME", "fraud-detection")
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
