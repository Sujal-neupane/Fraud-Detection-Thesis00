import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.services.model_service import predict

features_high = {
  "TransactionAmt": 1500.50,
  "C1": 15.0,
  "V14": -5.2,
  "V4": 4.5,
  "card1": 4045.0,
  "card2": 111.0
}

features_low = {
  "TransactionAmt": 24.99,
  "C1": 1.0,
  "V14": 0.5,
  "V4": -1.2,
  "card1": 10220.0,
  "card2": 512.0
}

high_result = predict(features_high)
low_result = predict(features_low)

print("High risk score:", high_result["score"])
print("High risk is_fraud:", high_result["is_fraud"])
print("Low risk score:", low_result["score"])
print("Low risk is_fraud:", low_result["is_fraud"])
