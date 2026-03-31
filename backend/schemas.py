"""
Pydantic Schemas
- Request and Response models for all endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# ── Prediction Request ────────────────────────────────────────────────────────
class TransactionInput(BaseModel):
    """
    Input schema for real-time prediction endpoint.
    V1-V28 are PCA features from creditcard dataset.
    """
    amount    : float = Field(..., gt=0, description="Transaction amount in USD")
    merchant  : Optional[str] = "Unknown"
    category  : Optional[str] = "unknown"
    card_type : Optional[str] = "visa"

    # V features (PCA components)
    V1  : float = 0.0
    V2  : float = 0.0
    V3  : float = 0.0
    V4  : float = 0.0
    V5  : float = 0.0
    V6  : float = 0.0
    V7  : float = 0.0
    V8  : float = 0.0
    V9  : float = 0.0
    V10 : float = 0.0
    V11 : float = 0.0
    V12 : float = 0.0
    V13 : float = 0.0
    V14 : float = 0.0
    V15 : float = 0.0
    V16 : float = 0.0
    V17 : float = 0.0
    V18 : float = 0.0
    V19 : float = 0.0
    V20 : float = 0.0
    V21 : float = 0.0
    V22 : float = 0.0
    V23 : float = 0.0
    V24 : float = 0.0
    V25 : float = 0.0
    V26 : float = 0.0
    V27 : float = 0.0
    V28 : float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 250.00,
                "merchant": "Amazon",
                "category": "online_retail",
                "card_type": "visa",
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V14": -0.311169,
            }
        }


# ── Prediction Response ───────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    transaction_id    : str
    fraud_probability : float
    is_fraud          : bool
    risk_level        : str
    threshold_used    : float
    amount            : float
    merchant          : str
    message           : str


# ── Transaction Record (from MongoDB) ─────────────────────────────────────────
class TransactionRecord(BaseModel):
    transaction_id    : str
    timestamp         : Optional[str]
    processed_at      : Optional[str]
    amount            : float
    currency          : Optional[str] = "USD"
    card_type         : Optional[str]
    merchant          : Optional[str]
    category          : Optional[str]
    merchant_country  : Optional[str]
    fraud_probability : float
    is_fraud          : bool
    risk_level        : str
    actual_class      : Optional[int]


# ── Stats Response ────────────────────────────────────────────────────────────
class StatsResponse(BaseModel):
    total_transactions : int
    total_fraud        : int
    total_legit        : int
    fraud_rate_pct     : float
    avg_fraud_amount   : float
    avg_legit_amount   : float
    risk_breakdown     : Dict[str, int]
    recent_fraud_count : int    # last 100 transactions


# ── Query Filters ─────────────────────────────────────────────────────────────
class TransactionFilter(BaseModel):
    is_fraud   : Optional[bool]   = None
    risk_level : Optional[str]    = None
    min_amount : Optional[float]  = None
    max_amount : Optional[float]  = None
    limit      : int              = Field(default=20, le=100)
    skip       : int              = Field(default=0, ge=0)