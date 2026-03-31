"""
LangChain FAISS Vector Store
- Uses LangChain's FAISS wrapper
- Saves/Loads as FOLDER (index.faiss + index.pkl)
- Supports similarity search with scores
"""

import os
import sys

from pathlib    import Path
from loguru     import logger
from typing     import List, Dict, Any, Tuple, Optional
from dotenv     import load_dotenv
from pymongo    import MongoClient

from langchain_community.vectorstores import FAISS
from langchain.schema                 import Document

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
sys.path.insert(0, str(BASE_DIR))

from rag.embedder import TransactionEmbedder, embedder

# ── This is the FOLDER where LangChain saves index.faiss + index.pkl ─────────
FAISS_INDEX_PATH = BASE_DIR / "ml" / "artifacts" / "faiss_index"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "fraud_detection")
MONGO_COL = os.getenv("MONGO_COLLECTION", "transactions")


class FraudVectorStore:
    """
    LangChain FAISS vector store for fraud transactions.
    """

    def __init__(self):
        self.embedder      = embedder
        self.lc_embeddings = embedder.get_langchain_embeddings()
        self.faiss_store   : Optional[FAISS] = None
        self.dim           = embedder.dim

    def _transaction_to_document(self, tx: Dict[str, Any]) -> Document:
        """Convert transaction dict to LangChain Document."""
        page_content = self.embedder.transaction_to_text(tx)

        metadata = {
            "transaction_id"   : str(tx.get("transaction_id", "")),
            "amount"           : float(tx.get("amount", 0)),
            "merchant"         : str(tx.get("merchant", "Unknown")),
            "category"         : str(tx.get("category", "unknown")),
            "card_type"        : str(tx.get("card_type", "unknown")),
            "merchant_country" : str(tx.get("merchant_country", "unknown")),
            "fraud_probability": float(tx.get("fraud_probability", 0)),
            "is_fraud"         : bool(tx.get("is_fraud", False)),
            "risk_level"       : str(tx.get("risk_level", "unknown")),
            "timestamp"        : str(tx.get("timestamp", "")),
            "Amount_log"       : float(tx.get("Amount_log", 0)),
        }

        return Document(page_content=page_content, metadata=metadata)

    def build_from_mongodb(self, fraud_only: bool = True) -> int:
        """Pull transactions from MongoDB and build FAISS index."""
        logger.info("Fetching transactions from MongoDB...")
        client     = MongoClient(MONGO_URI)
        collection = client[MONGO_DB][MONGO_COL]

        query = {"is_fraud": True} if fraud_only else {}
        docs  = list(collection.find(query, {"_id": 0}))
        client.close()

        if not docs:
            logger.warning("No transactions in MongoDB. Run Kafka producer first.")
            return 0

        logger.info(f"Building LangChain FAISS from {len(docs)} documents...")

        lc_documents = [self._transaction_to_document(tx) for tx in docs]

        self.faiss_store = FAISS.from_documents(
            documents = lc_documents,
            embedding = self.lc_embeddings,
        )

        logger.success(f"LangChain FAISS built | Documents: {len(lc_documents)}")

        self._save()
        return len(lc_documents)

    def _save(self):
        """Save FAISS store as folder (LangChain standard format)."""
        FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        self.faiss_store.save_local(str(FAISS_INDEX_PATH))
        logger.success(f"LangChain FAISS saved → {FAISS_INDEX_PATH}")
        logger.info(f"  Contains: index.faiss + index.pkl")

    def load(self) -> bool:
        """
        Load LangChain FAISS store from disk.
        LangChain saves as FOLDER with index.faiss + index.pkl inside.
        """
        index_file = FAISS_INDEX_PATH / "index.faiss"
        pkl_file   = FAISS_INDEX_PATH / "index.pkl"

        logger.info(f"Looking for FAISS at: {FAISS_INDEX_PATH}")
        logger.info(f"  index.faiss exists: {index_file.exists()}")
        logger.info(f"  index.pkl   exists: {pkl_file.exists()}")

        if not index_file.exists() or not pkl_file.exists():
            logger.warning(
                f"FAISS folder not found at {FAISS_INDEX_PATH}. "
                f"Run build_index() first."
            )
            return False

        try:
            self.faiss_store = FAISS.load_local(
                folder_path                     = str(FAISS_INDEX_PATH),
                embeddings                      = self.lc_embeddings,
                allow_dangerous_deserialization = True,
            )
            logger.success(
                f"LangChain FAISS loaded successfully | "
                f"Vectors: {self.faiss_store.index.ntotal}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load FAISS: {e}")
            return False

    def similarity_search(
        self,
        transaction : Dict[str, Any],
        top_k       : int = 5,
    ) -> List[Tuple[Document, float]]:
        """LangChain similarity search with relevance scores."""
        if not self.faiss_store:
            logger.warning("FAISS store not loaded.")
            return []

        query_text = self.embedder.transaction_to_text(transaction)

        results = self.faiss_store.similarity_search_with_relevance_scores(
            query = query_text,
            k     = top_k,
        )
        return results

    def add_transaction(self, transaction: Dict[str, Any]):
        """Add a new fraud transaction to the live index."""
        if not self.faiss_store:
            logger.warning("Cannot add — FAISS store not initialized.")
            return

        doc = self._transaction_to_document(transaction)
        self.faiss_store.add_documents([doc])
        logger.debug("New fraud transaction added to FAISS index.")

    @property
    def total_vectors(self) -> int:
        """Total number of vectors in the index."""
        if self.faiss_store and self.faiss_store.index:
            return self.faiss_store.index.ntotal
        return 0


# Module-level singleton
vector_store = FraudVectorStore()