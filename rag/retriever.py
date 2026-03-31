"""
LangChain Retriever
- Wraps FAISS store as a proper LangChain Retriever
- Uses LangChain's as_retriever() interface
- Returns formatted context for LLM
"""

import sys
from pathlib    import Path
from loguru     import logger
from typing     import Dict, Any, List

from langchain.schema           import Document
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.vector_store import FraudVectorStore, vector_store


class FraudRetriever:
    """
    LangChain-based retriever for fraud transactions.
    Uses LangChain's native retriever interface.
    """

    def __init__(self, store: FraudVectorStore = None):
        self.store = store or vector_store

    def get_langchain_retriever(self, top_k: int = 3):
        """
        Return a native LangChain retriever object.
        This can be plugged directly into LangChain chains.
        """
        if not self.store.faiss_store:
            logger.warning("FAISS store not loaded for retriever.")
            return None

        return self.store.faiss_store.as_retriever(
            search_type   = "similarity",
            search_kwargs = {"k": top_k},
        )

    def retrieve(
        self,
        transaction : Dict[str, Any],
        top_k       : int = 3,
        min_score   : float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar fraud transactions via LangChain.

        Returns:
            List of dicts with document content + metadata + score
        """
        results = self.store.similarity_search(
            transaction = transaction,
            top_k       = top_k,
        )

        filtered = []
        for doc, score in results:
            if score >= min_score:
                filtered.append({
                    "content"   : doc.page_content,
                    "metadata"  : doc.metadata,
                    "similarity": round(score, 4),
                })

        logger.info(
            f"LangChain Retriever | "
            f"Query: ${transaction.get('amount', 0):.2f} | "
            f"Retrieved: {len(filtered)} / {len(results)} cases"
        )

        return filtered

    def format_context(
        self,
        retrieved   : List[Dict[str, Any]],
        transaction : Dict[str, Any],
        prediction  : Dict[str, Any],
    ) -> str:
        """
        Format retrieved LangChain documents into
        a rich context string for LLM reasoning.
        """
        lines = []

        # ── Current Transaction ───────────────────────────
        lines.append("=== CURRENT TRANSACTION UNDER ANALYSIS ===")
        lines.append(f"Amount           : ${transaction.get('amount', 0):.2f}")
        lines.append(f"Merchant         : {transaction.get('merchant', 'Unknown')}")
        lines.append(f"Category         : {transaction.get('category', 'unknown')}")
        lines.append(f"Card Type        : {transaction.get('card_type', 'unknown')}")
        lines.append(f"Country          : {transaction.get('merchant_country', 'unknown')}")
        lines.append("")

        # ── ML Prediction ─────────────────────────────────
        lines.append("=== ML MODEL PREDICTION ===")
        lines.append(f"Fraud Probability : {prediction.get('fraud_probability', 0):.4f}")
        lines.append(f"Risk Level        : {prediction.get('risk_level', 'unknown')}")
        lines.append(f"Is Fraud          : {prediction.get('is_fraud', False)}")
        lines.append(f"Threshold Used    : {prediction.get('threshold_used', 0):.4f}")
        lines.append("")

        # ── Similar Historical Cases ───────────────────────
        if retrieved:
            lines.append(
                f"=== {len(retrieved)} SIMILAR HISTORICAL FRAUD CASES ==="
            )
            for i, item in enumerate(retrieved, 1):
                meta  = item["metadata"]
                score = item["similarity"]

                lines.append(f"\n[Case {i}] Similarity Score: {score:.4f}")
                lines.append(f"  Amount           : ${meta.get('amount', 0):.2f}")
                lines.append(f"  Merchant         : {meta.get('merchant', 'Unknown')}")
                lines.append(f"  Category         : {meta.get('category', 'unknown')}")
                lines.append(f"  Country          : {meta.get('merchant_country', 'unknown')}")
                lines.append(f"  Fraud Probability: {meta.get('fraud_probability', 0):.4f}")
                lines.append(f"  Risk Level       : {meta.get('risk_level', 'unknown')}")
                lines.append(f"  Timestamp        : {meta.get('timestamp', 'N/A')}")
                lines.append(f"  Description      : {item['content']}")
        else:
            lines.append("=== NO SIMILAR HISTORICAL CASES FOUND ===")
            lines.append("This transaction may represent a novel fraud pattern.")

        return "\n".join(lines)


# Module-level singleton
retriever = FraudRetriever()