"""
MongoDB Connection Manager
- Single shared client across the app
- Async using Motor
"""

import os
import sys
from pathlib import Path
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB", "fraud_detection")
MONGO_COL = os.getenv("MONGO_COLLECTION", "transactions")


class Database:
    client: AsyncIOMotorClient = None


db = Database()


async def connect_db():
    """Connect to MongoDB on app startup."""
    try:
        db.client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        await db.client.admin.command("ping")
        logger.success(f"MongoDB connected → {MONGO_URI}")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        sys.exit(1)


async def close_db():
    """Close MongoDB connection on app shutdown."""
    if db.client:
        db.client.close()
        logger.info("MongoDB connection closed.")


def get_collection():
    """Return the main transactions collection."""
    return db.client[MONGO_DB][MONGO_COL]