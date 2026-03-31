"""
Kafka Consumer
- Reads transactions from Kafka topic
- Runs ML fraud prediction
- Stores results in MongoDB
- Logs alerts for high-risk transactions
"""

import os
import sys
import json
import time

from pathlib import Path
from datetime import datetime, timezone
from loguru import logger
from dotenv import load_dotenv
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# ── Load env ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

KAFKA_SERVERS  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC    = os.getenv("KAFKA_TOPIC", "fraud_transactions")
KAFKA_GROUP    = os.getenv("KAFKA_GROUP_ID", "fraud_detector_group")
MONGO_URI      = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB       = os.getenv("MONGO_DB", "fraud_detection")
MONGO_COL      = os.getenv("MONGO_COLLECTION", "transactions")

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
from ml.predict import predict_transaction

# ── Logger ───────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
)


def create_consumer(retries: int = 5) -> KafkaConsumer:
    """Create Kafka consumer with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers  = KAFKA_SERVERS,
    group_id           = KAFKA_GROUP,
    value_deserializer = lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset  = "latest",
    enable_auto_commit = True,
    max_poll_records   = 10,
    session_timeout_ms = 30000,
    api_version        = (2, 8, 0),   # ← ADD THIS LINE
)
            logger.success(f"Kafka consumer connected → {KAFKA_SERVERS}")
            return consumer

        except NoBrokersAvailable:
            logger.warning(
                f"Attempt {attempt}/{retries}: Kafka not ready. "
                f"Retrying in 5s..."
            )
            time.sleep(5)

    logger.error("Could not connect to Kafka. Is Docker running?")
    sys.exit(1)


def create_mongo_client():
    """Create and verify MongoDB connection."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        logger.success(f"MongoDB connected → {MONGO_URI}")
        return client
    except ConnectionFailure:
        logger.error("MongoDB connection failed. Is Docker running?")
        sys.exit(1)


def build_record(transaction: dict, prediction: dict) -> dict:
    """
    Merge transaction + prediction into a MongoDB document.
    """
    return {
        # ── Identity ──────────────────────────────────────
        "transaction_id"   : transaction.get("transaction_id"),
        "timestamp"        : transaction.get("timestamp"),
        "processed_at"     : datetime.now(timezone.utc).isoformat(),

        # ── Financial ─────────────────────────────────────
        "amount"           : transaction.get("amount"),
        "currency"         : transaction.get("currency", "USD"),
        "card_type"        : transaction.get("card_type"),

        # ── Merchant ──────────────────────────────────────
        "merchant"         : transaction.get("merchant"),
        "category"         : transaction.get("category"),
        "merchant_country" : transaction.get("merchant_country"),

        # ── ML Prediction ─────────────────────────────────
        "fraud_probability": prediction.get("fraud_probability"),
        "is_fraud"         : prediction.get("is_fraud"),
        "risk_level"       : prediction.get("risk_level"),
        "threshold_used"   : prediction.get("threshold_used"),

        # ── Ground Truth ──────────────────────────────────
        "actual_class"     : transaction.get("actual_class"),

        # ── Raw Features (for RAG retrieval later) ────────
        "features"         : {
            f"V{i}": transaction.get(f"V{i}", 0.0)
            for i in range(1, 29)
        },
        "Amount_log"       : transaction.get("Amount_log", 0.0),
    }


def log_alert(record: dict):
    """Print a visual alert for high-risk transactions."""
    risk  = record["risk_level"]
    prob  = record["fraud_probability"]
    tx_id = record["transaction_id"][:8]

    if risk == "CRITICAL":
        logger.critical(
            f"🚨🚨 CRITICAL FRAUD ALERT | "
            f"ID: {tx_id}... | "
            f"Prob: {prob:.4f} | "
            f"Amount: ${record['amount']:.2f} | "
            f"Merchant: {record['merchant']}"
        )
    elif risk == "HIGH":
        logger.error(
            f"🚨 HIGH RISK | "
            f"ID: {tx_id}... | "
            f"Prob: {prob:.4f} | "
            f"Amount: ${record['amount']:.2f}"
        )
    elif risk == "MEDIUM":
        logger.warning(
            f"⚠️  MEDIUM RISK | "
            f"ID: {tx_id}... | "
            f"Prob: {prob:.4f}"
        )


def run_consumer():
    """Main consumer loop."""
    logger.info("=" * 55)
    logger.info("   KAFKA CONSUMER STARTING")
    logger.info(f"   Topic  : {KAFKA_TOPIC}")
    logger.info(f"   Group  : {KAFKA_GROUP}")
    logger.info(f"   Mongo  : {MONGO_DB}.{MONGO_COL}")
    logger.info("=" * 55)

    consumer    = create_consumer()
    mongo_cli   = create_mongo_client()
    collection  = mongo_cli[MONGO_DB][MONGO_COL]

    # Create indexes for fast querying
    collection.create_index("transaction_id", unique=True)
    collection.create_index("is_fraud")
    collection.create_index("risk_level")
    collection.create_index("timestamp")
    logger.info("MongoDB indexes created.")

    processed = 0
    fraud_hits = 0

    try:
        logger.info("Waiting for messages from Kafka...")
        for message in consumer:
            transaction = message.value

            # ── Predict ───────────────────────────────────
            prediction = predict_transaction(transaction)

            # ── Build record ──────────────────────────────
            record = build_record(transaction, prediction)

            # ── Save to MongoDB ───────────────────────────
            try:
                collection.update_one(
                    {"transaction_id": record["transaction_id"]},
                    {"$set": record},
                    upsert=True
                )
            except Exception as e:
                logger.error(f"MongoDB write error: {e}")
                continue

            processed  += 1
            fraud_hits += int(record["is_fraud"])

            # ── Log ───────────────────────────────────────
            risk    = record["risk_level"]
            is_fraud= record["is_fraud"]
            prob    = record["fraud_probability"]

            status = "🚨 FRAUD" if is_fraud else "✅ LEGIT"
            logger.info(
                f"[{processed:>5}] {status} | "
                f"Risk: {risk:<8} | "
                f"Prob: {prob:.4f} | "
                f"${record['amount']:>8.2f} | "
                f"{record['merchant']}"
            )

            # ── Alert for high risk ────────────────────────
            if risk in ("CRITICAL", "HIGH", "MEDIUM"):
                log_alert(record)

            # ── Stats every 50 messages ───────────────────
            if processed % 50 == 0:
                logger.info(
                    f"── STATS | Processed: {processed} | "
                    f"Fraud Detected: {fraud_hits} | "
                    f"Rate: {fraud_hits/processed*100:.1f}% ──"
                )

    except KeyboardInterrupt:
        logger.warning("Consumer stopped by user.")

    finally:
        consumer.close()
        mongo_cli.close()
        logger.success(
            f"Consumer closed. "
            f"Processed: {processed} | Fraud: {fraud_hits}"
        )


if __name__ == "__main__":
    run_consumer()