"""
Prediction Module
- Used by Kafka consumer and FastAPI
- Loads model once at startup (singleton)
- Exposes predict() function
"""

import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger
from typing import Dict, Any, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH    = ARTIFACTS_DIR / "xgb_fraud_model.joblib"
SCALER_PATH   = ARTIFACTS_DIR / "scaler.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.joblib"
THRESHOLD_PATH= ARTIFACTS_DIR / "optimal_threshold.joblib"


class FraudPredictor:
    """
    Singleton class for fraud prediction.
    Loads model once — reused across all predictions.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """Load model artifacts from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run ml/train.py first."
            )

        self.model         = joblib.load(MODEL_PATH)
        self.scaler        = joblib.load(SCALER_PATH)
        self.feature_names = joblib.load(FEATURES_PATH)
        self.threshold     = joblib.load(THRESHOLD_PATH) \
                             if THRESHOLD_PATH.exists() else 0.5

        logger.success(
            f"FraudPredictor loaded | "
            f"Features: {len(self.feature_names)} | "
            f"Threshold: {self.threshold:.4f}"
        )

    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud probability for a single transaction.

        Args:
            transaction: dict with keys matching feature_names

        Returns:
            dict with fraud_probability, is_fraud, risk_level
        """
        # Build feature vector in correct order
        feature_vector = []
        for feat in self.feature_names:
            val = transaction.get(feat, 0.0)
            feature_vector.append(float(val))

        X = pd.DataFrame([feature_vector], columns=self.feature_names)
        X_scaled = self.scaler.transform(X)

        fraud_prob = float(self.model.predict_proba(X_scaled)[0][1])
        is_fraud   = fraud_prob >= self.threshold

        risk_level = self._get_risk_level(fraud_prob)

        return {
            "fraud_probability" : round(fraud_prob, 6),
            "is_fraud"          : is_fraud,
            "risk_level"        : risk_level,
            "threshold_used"    : round(self.threshold, 4),
        }

    def _get_risk_level(self, prob: float) -> str:
        """Map probability to human-readable risk level."""
        if prob >= 0.85:
            return "CRITICAL"
        elif prob >= 0.65:
            return "HIGH"
        elif prob >= 0.40:
            return "MEDIUM"
        elif prob >= 0.20:
            return "LOW"
        else:
            return "MINIMAL"


# ── Module-level predictor instance ─────────────────────────────────────────
predictor = FraudPredictor()


def predict_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Public function used by other modules."""
    return predictor.predict(transaction)


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    # Load one real sample from dataset
    df      = pd.read_csv(BASE_DIR / "data" / "creditcard.csv")
    sample  = df.drop(columns=["Time", "Amount", "Class"]).iloc[0].to_dict()
    sample["Amount_log"] = float(np.log1p(df.iloc[0]["Amount"]))

    result  = predict_transaction(sample)

    logger.info("─" * 40)
    logger.info(f"Fraud Probability : {result['fraud_probability']}")
    logger.info(f"Is Fraud          : {result['is_fraud']}")
    logger.info(f"Risk Level        : {result['risk_level']}")
    logger.info("─" * 40)