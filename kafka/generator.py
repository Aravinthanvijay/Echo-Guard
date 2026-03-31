"""
Transaction Generator
- Samples REAL rows from creditcard.csv
- Adds realistic metadata (transaction_id, timestamp, merchant, etc.)
- Used by the Kafka producer
"""

import uuid
import random
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Iterator

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "ml" / "data" / "creditcard.csv"

# ── Merchant metadata for realism ────────────────────────────────────────────
MERCHANTS = [
    "Amazon", "Walmart", "Target", "BestBuy", "Costco",
    "Netflix", "Uber", "Airbnb", "Steam", "Apple Store",
    "Shell Gas", "Starbucks", "McDonald's", "Subway", "CVS Pharmacy",
]

CATEGORIES = [
    "online_retail", "grocery", "entertainment", "travel",
    "food_dining", "gas_transport", "health_wellness", "electronics",
]

COUNTRIES = ["US", "UK", "CA", "AU", "IN", "DE", "FR", "SG", "JP", "BR"]

CARD_TYPES = ["visa", "mastercard", "amex", "discover"]


class TransactionGenerator:
    """
    Generates realistic transaction dicts by sampling
    from the real creditcard.csv dataset.
    """

    def __init__(self, fraud_rate: float = 0.02):
        """
        Args:
            fraud_rate: probability of injecting a fraud sample (default 2%)
        """
        self.fraud_rate = fraud_rate
        self._load_dataset()

    def _load_dataset(self):
        """Load and separate legitimate vs fraud rows."""
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)

        # Separate by class
        self.legit_df = df[df["Class"] == 0].reset_index(drop=True)
        self.fraud_df = df[df["Class"] == 1].reset_index(drop=True)

        print(f"[Generator] Loaded {len(self.legit_df):,} legit | "
              f"{len(self.fraud_df):,} fraud samples")

    def _sample_row(self, is_fraud: bool) -> pd.Series:
        """Sample a random row from legit or fraud pool."""
        df  = self.fraud_df if is_fraud else self.legit_df
        idx = random.randint(0, len(df) - 1)
        return df.iloc[idx]

    def _build_transaction(self, row: pd.Series, is_fraud: bool) -> Dict[str, Any]:
        """
        Build a full transaction dict from a dataset row.
        Adds metadata fields for realism.
        """
        # Extract V1–V28 features
        v_features = {
            f"V{i}": round(float(row[f"V{i}"]), 6)
            for i in range(1, 29)
        }

        amount     = round(float(row["Amount"]), 2)
        amount_log = round(float(np.log1p(amount)), 6)

        transaction = {
            # ── Identity ──────────────────────────────────────
            "transaction_id"  : str(uuid.uuid4()),
            "timestamp"       : datetime.now(timezone.utc).isoformat(),

            # ── Financial ─────────────────────────────────────
            "amount"          : amount,
            "Amount_log"      : amount_log,
            "currency"        : "USD",
            "card_type"       : random.choice(CARD_TYPES),

            # ── Merchant ──────────────────────────────────────
            "merchant"        : random.choice(MERCHANTS),
            "category"        : random.choice(CATEGORIES),
            "merchant_country": random.choice(COUNTRIES),

            # ── Ground Truth (for evaluation) ─────────────────
            "actual_class"    : int(is_fraud),

            # ── Raw ML Features ───────────────────────────────
            **v_features,
        }

        return transaction

    def generate_one(self) -> Dict[str, Any]:
        """Generate a single transaction."""
        is_fraud = random.random() < self.fraud_rate
        row      = self._sample_row(is_fraud)
        return self._build_transaction(row, is_fraud)

    def generate_stream(self, total: int = None) -> Iterator[Dict[str, Any]]:
        """
        Yield transactions indefinitely (or up to `total`).

        Args:
            total: stop after this many. None = infinite
        """
        count = 0
        while True:
            yield self.generate_one()
            count += 1
            if total and count >= total:
                break


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gen = TransactionGenerator(fraud_rate=0.1)
    tx  = gen.generate_one()

    print("\n── Sample Transaction ──────────────────────")
    for k, v in tx.items():
        if not k.startswith("V"):   # skip V features for readability
            print(f"  {k:<22}: {v}")
    print(f"  V1 ... V28          : (28 features present)")
    print("────────────────────────────────────────────")