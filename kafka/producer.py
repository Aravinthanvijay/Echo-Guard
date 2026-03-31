"""
Kafka Producer
- Reads transactions from TransactionGenerator
- Serializes to JSON
- Sends to Kafka topic
- Configurable speed (delay between messages)
"""

import os
import sys
import json
import time
import argparse

from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

# ── Load env ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC   = os.getenv("KAFKA_TOPIC", "fraud_transactions")

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
from generator import TransactionGenerator

# ── Logger ───────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
)


def create_producer(retries: int = 5) -> KafkaProducer:
    """Create Kafka producer with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
    bootstrap_servers  = KAFKA_SERVERS,
    value_serializer   = lambda v: json.dumps(v).encode("utf-8"),
    key_serializer     = lambda k: k.encode("utf-8"),
    acks               = "all",
    retries            = 3,
    max_block_ms       = 10000,
    request_timeout_ms = 30000,
    api_version        = (2, 8, 0),   # ← ADD THIS LINE
)
            logger.success(f"Kafka producer connected → {KAFKA_SERVERS}")
            return producer

        except NoBrokersAvailable:
            logger.warning(
                f"Attempt {attempt}/{retries}: Kafka not ready. "
                f"Retrying in 5s..."
            )
            time.sleep(5)

    logger.error("Could not connect to Kafka. Is Docker running?")
    sys.exit(1)


def on_success(metadata):
    logger.debug(
        f"Sent → topic={metadata.topic} | "
        f"partition={metadata.partition} | "
        f"offset={metadata.offset}"
    )


def on_error(e):
    logger.error(f"Send failed: {e}")


def run_producer(
    total: int  = None,
    delay: float = 0.5,
    fraud_rate: float = 0.05
):
    """
    Main producer loop.

    Args:
        total     : total transactions to send (None = infinite)
        delay     : seconds between messages
        fraud_rate: probability of fraud transaction
    """
    logger.info("=" * 55)
    logger.info("   KAFKA PRODUCER STARTING")
    logger.info(f"   Topic      : {KAFKA_TOPIC}")
    logger.info(f"   Delay      : {delay}s between messages")
    logger.info(f"   Fraud Rate : {fraud_rate * 100:.1f}%")
    logger.info(f"   Total      : {total or 'infinite'}")
    logger.info("=" * 55)

    producer  = create_producer()
    generator = TransactionGenerator(fraud_rate=fraud_rate)

    sent_count  = 0
    fraud_count = 0

    try:
        for transaction in generator.generate_stream(total=total):
            tx_id    = transaction["transaction_id"]
            is_fraud = transaction["actual_class"]

            # Send to Kafka
            future = producer.send(
                topic     = KAFKA_TOPIC,
                key       = tx_id,
                value     = transaction,
            )
            future.add_callback(on_success)
            future.add_errback(on_error)

            sent_count  += 1
            fraud_count += is_fraud

            # Log every transaction
            fraud_tag = "🚨 FRAUD" if is_fraud else "✅ LEGIT"
            logger.info(
                f"[{sent_count:>5}] {fraud_tag} | "
                f"ID: {tx_id[:8]}... | "
                f"Amount: ${transaction['amount']:>8.2f} | "
                f"Merchant: {transaction['merchant']}"
            )

            # Flush every 10 messages
            if sent_count % 10 == 0:
                producer.flush()
                logger.info(
                    f"── Flushed | Sent: {sent_count} | "
                    f"Fraud: {fraud_count} ──"
                )

            time.sleep(delay)

    except KeyboardInterrupt:
        logger.warning("Producer stopped by user.")

    finally:
        producer.flush()
        producer.close()
        logger.success(
            f"Producer closed. "
            f"Total sent: {sent_count} | Fraud: {fraud_count}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Fraud Transaction Producer")
    parser.add_argument("--total",      type=int,   default=None,  help="Total transactions (default: infinite)")
    parser.add_argument("--delay",      type=float, default=0.5,   help="Delay between messages in seconds")
    parser.add_argument("--fraud-rate", type=float, default=0.05,  help="Fraud probability (0.0 - 1.0)")
    args = parser.parse_args()

    run_producer(
        total      = args.total,
        delay      = args.delay,
        fraud_rate = args.fraud_rate,
    )