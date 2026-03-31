"""
LangChain Transaction Embedder
- Uses LangChain HuggingFaceEmbeddings wrapper
- Wraps Sentence Transformers properly via LangChain
- Used by FAISS vector store
"""

import os
import sys
import numpy as np

from pathlib        import Path
from loguru         import logger
from typing         import Dict, Any, List
from dotenv         import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR        = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class TransactionEmbedder:
    """
    LangChain-wrapped Sentence Transformer embedder.
    Converts transactions to text then to vectors.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """Load LangChain HuggingFace embeddings."""
        logger.info(f"Loading LangChain embeddings: {EMBEDDING_MODEL}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name      = EMBEDDING_MODEL,
            model_kwargs    = {"device": "cpu"},
            encode_kwargs   = {"normalize_embeddings": True},
        )

        # Get dimension by running a test embed
        test_vec  = self.embeddings.embed_query("test")
        self.dim  = len(test_vec)

        logger.success(
            f"LangChain embeddings loaded | "
            f"Model: {EMBEDDING_MODEL} | "
            f"Dim: {self.dim}"
        )

    def transaction_to_text(self, transaction: Dict[str, Any]) -> str:
        """
        Convert transaction dict to rich natural language text.
        Better text = better semantic retrieval.
        """
        amount     = transaction.get("amount", 0)
        merchant   = transaction.get("merchant", "Unknown")
        category   = transaction.get("category", "unknown")
        card_type  = transaction.get("card_type", "unknown")
        country    = transaction.get("merchant_country", "unknown")
        risk_level = transaction.get("risk_level", "unknown")
        fraud_prob = transaction.get("fraud_probability", 0)
        is_fraud   = transaction.get("is_fraud", False)

        # Key PCA fraud indicators
        features   = transaction.get("features", {})
        v14 = transaction.get("V14", features.get("V14", 0))
        v17 = transaction.get("V17", features.get("V17", 0))
        v12 = transaction.get("V12", features.get("V12", 0))
        v10 = transaction.get("V10", features.get("V10", 0))

        label = "FRAUDULENT" if is_fraud else "LEGITIMATE"

        return (
            f"Transaction classified as {label}. "
            f"Amount: ${float(amount):.2f}. "
            f"Merchant: {merchant} in category {category}. "
            f"Card type: {card_type}. Country: {country}. "
            f"Fraud probability: {float(fraud_prob):.4f}. "
            f"Risk level: {risk_level}. "
            f"Key fraud indicators - "
            f"V14: {float(v14):.4f}, "
            f"V17: {float(v17):.4f}, "
            f"V12: {float(v12):.4f}, "
            f"V10: {float(v10):.4f}."
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string via LangChain."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents via LangChain."""
        return self.embeddings.embed_documents(texts)

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Return raw LangChain embeddings object (used by FAISS)."""
        return self.embeddings


# Module-level singleton
embedder = TransactionEmbedder()