"""
LangChain RAG Engine — Main Pipeline
- Uses LangChain FAISS + Retriever properly
- Builds and queries the index
- Feeds context to LLM in next step
"""

import sys
from pathlib import Path
from loguru  import logger
from typing  import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.vector_store import FraudVectorStore, vector_store
from rag.retriever    import FraudRetriever,   retriever


class RAGEngine:
    """
    Main LangChain RAG pipeline for fraud analysis.
    """

    def __init__(self):
        self.store         = vector_store
        self.retriever     = retriever
        self._initialized  = False

    def initialize(self) -> bool:
        """Load LangChain FAISS index from disk on startup."""
        loaded             = self.store.load()
        self._initialized  = loaded

        if loaded:
            logger.success(
                f"LangChain RAG Engine ready | "
                f"Index: {self.store.total_vectors} vectors"
            )
        else:
            logger.warning(
                "RAG Engine: No index found. "
                "Call /api/rag/build after running Kafka producer."
            )
        return loaded

    def build_index(self, fraud_only: bool = True) -> int:
        """Build LangChain FAISS index from MongoDB."""
        count              = self.store.build_from_mongodb(fraud_only)
        self._initialized  = count > 0
        return count

    def query(
        self,
        transaction : Dict[str, Any],
        prediction  : Dict[str, Any],
        top_k       : int = 3,
    ) -> Dict[str, Any]:
        """
        Full LangChain RAG query.

        1. Convert transaction to text
        2. Embed with LangChain HuggingFace embeddings
        3. Search LangChain FAISS store
        4. Format context for LLM

        Returns:
            retrieved_cases, context string, metadata
        """
        if not self._initialized:
            logger.warning("RAG not initialized. Trying to load from disk...")
            self.initialize()

        # Retrieve via LangChain
        retrieved = self.retriever.retrieve(
            transaction = transaction,
            top_k       = top_k,
            min_score   = 0.2,
        )

        # Format context string for LLM
        context = self.retriever.format_context(
            retrieved   = retrieved,
            transaction = transaction,
            prediction  = prediction,
        )

        # Get native LangChain retriever (for LangChain chain in LLM step)
        lc_retriever = self.retriever.get_langchain_retriever(top_k=top_k)

        return {
            "retrieved_cases"  : retrieved,
            "context"          : context,
            "total_retrieved"  : len(retrieved),
            "index_size"       : self.store.total_vectors,
            "lc_retriever"     : lc_retriever,   # used by LLM chain
        }

    def add_new_fraud(self, transaction: Dict[str, Any]):
        """Live index update — add new fraud transaction."""
        if transaction.get("is_fraud") and self._initialized:
            self.store.add_transaction(transaction)


# Module-level singleton
rag_engine = RAGEngine()