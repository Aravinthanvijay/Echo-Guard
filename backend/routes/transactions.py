"""
Transactions Route
GET /api/transactions       - list with filters
GET /api/transactions/{id}  - single record
GET /api/transactions/fraud - only fraud records
"""

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from typing import Optional, List
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.database import get_collection
from backend.schemas import TransactionRecord

router = APIRouter(prefix="/api/transactions", tags=["Transactions"])


def clean_doc(doc: dict) -> dict:
    """Remove MongoDB _id and nested features for API response."""
    doc.pop("_id", None)
    doc.pop("features", None)
    return doc


@router.get("", response_model=List[TransactionRecord])
async def list_transactions(
    is_fraud   : Optional[bool]  = Query(None,  description="Filter by fraud flag"),
    risk_level : Optional[str]   = Query(None,  description="MINIMAL/LOW/MEDIUM/HIGH/CRITICAL"),
    min_amount : Optional[float] = Query(None,  description="Minimum transaction amount"),
    max_amount : Optional[float] = Query(None,  description="Maximum transaction amount"),
    limit      : int             = Query(20,    le=100),
    skip       : int             = Query(0,     ge=0),
):
    """
    List transactions with optional filters.
    Sorted by most recent first.
    """
    collection = get_collection()
    query      = {}

    if is_fraud is not None:
        query["is_fraud"] = is_fraud

    if risk_level:
        query["risk_level"] = risk_level.upper()

    if min_amount is not None or max_amount is not None:
        query["amount"] = {}
        if min_amount is not None:
            query["amount"]["$gte"] = min_amount
        if max_amount is not None:
            query["amount"]["$lte"] = max_amount

    try:
        cursor = collection.find(query) \
                           .sort("timestamp", -1) \
                           .skip(skip) \
                           .limit(limit)

        docs = await cursor.to_list(length=limit)
        return [clean_doc(doc) for doc in docs]

    except Exception as e:
        logger.error(f"List transactions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fraud", response_model=List[TransactionRecord])
async def list_fraud_transactions(
    limit : int = Query(20, le=100),
    skip  : int = Query(0,  ge=0),
):
    """Get only fraud transactions sorted by probability."""
    collection = get_collection()
    try:
        cursor = collection.find({"is_fraud": True}) \
                           .sort("fraud_probability", -1) \
                           .skip(skip) \
                           .limit(limit)

        docs = await cursor.to_list(length=limit)
        return [clean_doc(doc) for doc in docs]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{transaction_id}", response_model=TransactionRecord)
async def get_transaction(transaction_id: str):
    """Get a single transaction by its ID."""
    collection = get_collection()
    try:
        doc = await collection.find_one({"transaction_id": transaction_id})
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Transaction {transaction_id} not found"
            )
        return clean_doc(doc)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))