"""
Prediction Route
POST /api/predict
- Accepts a transaction
- Returns fraud prediction instantly
"""

import uuid
import numpy as np

from fastapi import APIRouter, HTTPException
from loguru import logger
from datetime import datetime, timezone
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.predict import predict_transaction
from backend.schemas import TransactionInput, PredictionResponse

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionInput):
    """
    Real-time fraud prediction for a single transaction.

    - Accepts transaction features
    - Returns fraud probability, risk level, and verdict
    """
    try:
        # Build feature dict for ML model
        tx_dict = transaction.model_dump()

        # Add log-transformed amount
        tx_dict["Amount_log"] = float(np.log1p(transaction.amount))

        # Run prediction
        result = predict_transaction(tx_dict)

        tx_id = str(uuid.uuid4())

        # Build human-readable message
        if result["is_fraud"]:
            message = (
                f"⚠️ FRAUD DETECTED with {result['fraud_probability']*100:.1f}% "
                f"confidence. Risk level: {result['risk_level']}."
            )
        else:
            message = (
                f"✅ Transaction appears LEGITIMATE with "
                f"{(1 - result['fraud_probability'])*100:.1f}% confidence."
            )

        logger.info(
            f"Prediction | ID: {tx_id[:8]}... | "
            f"Fraud: {result['is_fraud']} | "
            f"Risk: {result['risk_level']} | "
            f"Prob: {result['fraud_probability']:.4f}"
        )

        return PredictionResponse(
            transaction_id    = tx_id,
            fraud_probability = result["fraud_probability"],
            is_fraud          = result["is_fraud"],
            risk_level        = result["risk_level"],
            threshold_used    = result["threshold_used"],
            amount            = transaction.amount,
            merchant          = transaction.merchant,
            message           = message,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))