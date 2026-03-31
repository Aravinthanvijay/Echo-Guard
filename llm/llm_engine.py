"""
LLM Explanation Engine
- Combines RAG + LangChain chain + GPT OSS 120B
- Full pipeline:
  Transaction → RAG Retrieval → Prompt → GPT OSS 120B → Explanation
"""

import sys
import time

from pathlib import Path
from loguru  import logger
from typing  import Dict, Any
from dotenv  import load_dotenv

from langchain.schema.output_parser import StrOutputParser

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
sys.path.insert(0, str(BASE_DIR))

from llm.groq_llm   import LLMProvider, llm_provider
from llm.prompt     import get_fraud_prompt, get_alert_prompt
from rag.rag_engine import RAGEngine, rag_engine


class LLMExplainEngine:
    """
    Full pipeline:

    ┌─────────────┐
    │ Transaction │
    └──────┬──────┘
           │
           ▼
    ┌─────────────────┐
    │  RAG Retrieval  │ ← finds similar fraud cases from FAISS
    │  (LangChain)    │
    └──────┬──────────┘
           │
           ▼
    ┌──────────────────┐
    │  Format Context  │ ← transaction + prediction + similar cases
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │  LangChain Chain                 │
    │  prompt | llm | output_parser    │
    │  GPT OSS 120B via Groq           │
    └──────┬───────────────────────────┘
           │
           ▼
    ┌──────────────┐
    │  Explanation │ ← human readable fraud analysis
    └──────────────┘
    """

    def __init__(self):
        self.llm_provider = llm_provider
        self.rag_engine   = rag_engine
        self._build_chains()

    def _build_chains(self):
        """
        Build LangChain LCEL chains.
        LCEL pipe syntax: prompt | llm | parser
        """
        llm = self.llm_provider.get_llm()

        # ── Main analysis chain ───────────────────────────
        self.fraud_chain = (
            get_fraud_prompt()
            | llm
            | StrOutputParser()
        )

        # ── Short alert chain ─────────────────────────────
        self.alert_chain = (
            get_alert_prompt()
            | llm
            | StrOutputParser()
        )

        logger.success(
            "LangChain LCEL chains built | "
            "Using GPT OSS 120B via Groq"
        )

    def explain(
        self,
        transaction : Dict[str, Any],
        prediction  : Dict[str, Any],
        top_k       : int = 3,
    ) -> Dict[str, Any]:
        """
        Full explanation pipeline.

        Args:
            transaction : transaction feature dict
            prediction  : ML model prediction dict
            top_k       : similar fraud cases to retrieve

        Returns:
            dict with explanation, context, cases, timing
        """
        start = time.time()

        # ── Step 1: Init RAG if needed ────────────────────
        if not self.rag_engine._initialized:
            logger.info("Initializing RAG engine...")
            self.rag_engine.initialize()

        # ── Step 2: RAG Retrieval ─────────────────────────
        logger.info("Step 1/3 → RAG retrieval...")
        rag_result = self.rag_engine.query(
            transaction = transaction,
            prediction  = prediction,
            top_k       = top_k,
        )

        context   = rag_result["context"]
        retrieved = rag_result["retrieved_cases"]
        logger.info(f"RAG found {len(retrieved)} similar cases")

        # ── Step 3: Run GPT OSS 120B via LangChain ────────
        logger.info(
            f"Step 2/3 → Calling {self.llm_provider.model} via Groq..."
        )
        try:
            explanation = self.fraud_chain.invoke({
                "context": context
            })
            logger.success("Step 3/3 → Explanation generated!")

        except Exception as e:
            logger.error(f"GPT OSS 120B call failed: {e}")
            logger.warning("Using fallback explanation.")
            explanation = self._fallback_explanation(prediction)

        elapsed = round(time.time() - start, 2)

        logger.success(
            f"Pipeline complete | "
            f"Time: {elapsed}s | "
            f"Model: {self.llm_provider.model} | "
            f"RAG Cases: {len(retrieved)}"
        )

        return {
            "explanation"     : explanation,
            "context_used"    : context,
            "retrieved_cases" : retrieved,
            "total_retrieved" : len(retrieved),
            "prediction"      : prediction,
            "latency_seconds" : elapsed,
            "model_used"      : self.llm_provider.model,
            "provider"        : self.llm_provider.provider,
        }

    def generate_alert(
        self,
        transaction : Dict[str, Any],
        prediction  : Dict[str, Any],
    ) -> str:
        """Generate short 2-sentence fraud alert."""
        try:
            return self.alert_chain.invoke({
                "amount"           : f"{transaction.get('amount', 0):.2f}",
                "merchant"         : transaction.get("merchant", "Unknown"),
                "risk_level"       : prediction.get("risk_level", "HIGH"),
                "fraud_probability": f"{prediction.get('fraud_probability', 0):.2%}",
            })
        except Exception as e:
            logger.error(f"Alert failed: {e}")
            return self._fallback_alert(transaction, prediction)

    def _fallback_explanation(self, prediction: Dict[str, Any]) -> str:
        """Fallback when GPT OSS 120B call fails."""
        prob     = prediction.get("fraud_probability", 0)
        risk     = prediction.get("risk_level", "UNKNOWN")
        is_fraud = prediction.get("is_fraud", False)

        return (
            f"VERDICT: Transaction classified as "
            f"{'FRAUDULENT' if is_fraud else 'LEGITIMATE'}.\n\n"
            f"RISK FACTORS:\n"
            f"• Fraud probability: {prob:.2%}\n"
            f"• Risk level: {risk}\n"
            f"• Flagged by ML model\n\n"
            f"SIMILAR PATTERNS:\n"
            f"• Matches known fraud signatures in database\n\n"
            f"RECOMMENDATION:\n"
            f"{'Block transaction and contact cardholder.' if is_fraud else 'Transaction safe to approve.'}"
        )

    def _fallback_alert(
        self,
        transaction : Dict[str, Any],
        prediction  : Dict[str, Any],
    ) -> str:
        """Fallback when alert call fails."""
        return (
            f"FRAUD ALERT: Transaction of "
            f"${transaction.get('amount', 0):.2f} at "
            f"{transaction.get('merchant', 'Unknown')} flagged with "
            f"{prediction.get('fraud_probability', 0):.2%} fraud probability. "
            f"Risk level: {prediction.get('risk_level', 'HIGH')}. "
            f"Immediate review required."
        )


# ── Module-level singleton ────────────────────────────────────────────────────
llm_engine = LLMExplainEngine()