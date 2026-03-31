"""
Explanation Route
POST /api/explain  - full RAG + LLM fraud explanation
POST /api/alert    - short alert message
"""

import sys
import numpy as np
from pathlib  import Path
from fastapi  import APIRouter, HTTPException
from loguru   import logger

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from llm.llm_engine  import llm_engine
from ml.predict      import predict_transaction
from backend.schemas import TransactionInput

router = APIRouter(prefix="/api", tags=["LLM Explanation"])


@router.post("/explain")
async def explain_transaction(transaction: TransactionInput):
    """
    Full fraud explanation pipeline:
    1. ML prediction
    2. RAG retrieval of similar cases
    3. LLM generates human-readable explanation

    Returns explanation + context + similar cases
    """
    try:
        # Build feature dict
        tx_dict             = transaction.model_dump()
        tx_dict["Amount_log"] = float(np.log1p(transaction.amount))

        # ML prediction
        prediction = predict_transaction(tx_dict)

        # LLM explanation with RAG context
        import asyncio
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: llm_engine.explain(
                transaction = tx_dict,
                prediction  = prediction,
                top_k       = 3,
            )
        )
 

        return {
            "transaction_summary" : {
                "amount"   : transaction.amount,
                "merchant" : transaction.merchant,
                "category" : transaction.category,
            },
            "prediction"          : result["prediction"],
            "explanation"         : result["explanation"],
            "similar_cases_count" : result["total_retrieved"],
            "similar_cases"       : [
                {
                    "similarity"        : c["similarity"],
                    "amount"            : c["metadata"]["amount"],
                    "merchant"          : c["metadata"]["merchant"],
                    "risk_level"        : c["metadata"]["risk_level"],
                    "fraud_probability" : c["metadata"]["fraud_probability"],
                }
                for c in result["retrieved_cases"]
            ],
            "model_used"          : result["model_used"],
            "latency_seconds"     : result["latency_seconds"],
        }

    except Exception as e:
        logger.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alert")
async def generate_alert(transaction: TransactionInput):
    """
    Generate a short fraud alert message.
    Used for notifications and alerts.
    """
    try:
        tx_dict               = transaction.model_dump()
        tx_dict["Amount_log"] = float(np.log1p(transaction.amount))

        prediction = predict_transaction(tx_dict)

        alert = llm_engine.generate_alert(
            transaction = tx_dict,
            prediction  = prediction,
        )

        return {
            "alert"      : alert,
            "is_fraud"   : prediction["is_fraud"],
            "risk_level" : prediction["risk_level"],
            "amount"     : transaction.amount,
            "merchant"   : transaction.merchant,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))