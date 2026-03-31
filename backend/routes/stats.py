"""
Stats Route
GET /api/stats - aggregated dashboard metrics
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.database import get_collection
from backend.schemas import StatsResponse

router = APIRouter(prefix="/api", tags=["Stats"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Aggregated statistics for the dashboard.
    - Total counts
    - Fraud rate
    - Average amounts
    - Risk level breakdown
    """
    collection = get_collection()

    try:
        # ── Total counts ──────────────────────────────────
        total_transactions = await collection.count_documents({})
        total_fraud        = await collection.count_documents({"is_fraud": True})
        total_legit        = total_transactions - total_fraud
        fraud_rate_pct     = (
            round(total_fraud / total_transactions * 100, 2)
            if total_transactions > 0 else 0.0
        )

        # ── Average amounts ───────────────────────────────
        fraud_pipeline = [
            {"$match": {"is_fraud": True}},
            {"$group": {"_id": None, "avg": {"$avg": "$amount"}}}
        ]
        legit_pipeline = [
            {"$match": {"is_fraud": False}},
            {"$group": {"_id": None, "avg": {"$avg": "$amount"}}}
        ]

        fraud_avg_cur = await collection.aggregate(fraud_pipeline).to_list(1)
        legit_avg_cur = await collection.aggregate(legit_pipeline).to_list(1)

        avg_fraud_amount = round(fraud_avg_cur[0]["avg"], 2) if fraud_avg_cur else 0.0
        avg_legit_amount = round(legit_avg_cur[0]["avg"], 2) if legit_avg_cur else 0.0

        # ── Risk level breakdown ──────────────────────────
        risk_pipeline = [
            {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}
        ]
        risk_cur = await collection.aggregate(risk_pipeline).to_list(10)
        risk_breakdown = {
            item["_id"]: item["count"]
            for item in risk_cur
            if item["_id"]
        }

        # ── Recent fraud (last 100 transactions) ──────────
# ── Recent fraud (last 100 transactions) ──────────
        recent_pipeline = [
            {"$sort"  : {"timestamp": -1}},
            {"$limit" : 100},
        ]
        recent_docs        = await collection.aggregate(recent_pipeline).to_list(100)
        recent_fraud_count = sum(1 for d in recent_docs if d.get("is_fraud"))

        return StatsResponse(
            total_transactions = total_transactions,
            total_fraud        = total_fraud,
            total_legit        = total_legit,
            fraud_rate_pct     = fraud_rate_pct,
            avg_fraud_amount   = avg_fraud_amount,
            avg_legit_amount   = avg_legit_amount,
            risk_breakdown     = risk_breakdown,
            recent_fraud_count = recent_fraud_count,
        )

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))