"""
Groq LLM Provider using GPT OSS 120B
- Platform  : Groq  (console.groq.com)
- API Key   : Your identity/password for Groq → gsk_xxxx
- Model     : openai/gpt-oss-120b → the AI brain (120B parameters)
- Both API Key AND Model are always required together
- Wrapped via LangChain ChatGroq
"""

import os
import sys

from pathlib import Path
from loguru  import logger
from dotenv  import load_dotenv

from langchain_groq import ChatGroq

# ── Load .env ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Read BOTH from .env ───────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "openai/gpt-oss-120b")


class LLMProvider:
    """
    Groq LLM Provider.

    How it works:
    ┌─────────────────────────────────────────────┐
    │  Your Code                                  │
    │      ↓                                      │
    │  API Key (gsk_xxx) → Groq Server            │
    │      ↓                                      │
    │  Model (openai/gpt-oss-120b) → AI Brain     │
    │      ↓                                      │
    │  Response → Your App                        │
    └─────────────────────────────────────────────┘

    API Key  = proves you are allowed to use Groq
    Model    = tells Groq WHICH AI to run
    Both     = always needed together
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """Load GPT OSS 120B via Groq."""

        # ── Step 1: Validate API Key ──────────────────────
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            raise ValueError(
                "\n\n" + "=" * 55 + "\n"
                "❌ GROQ_API_KEY missing in .env file!\n\n"
                "To fix:\n"
                "1. Go to → https://console.groq.com\n"
                "2. Sign up FREE\n"
                "3. Click 'API Keys' → 'Create API Key'\n"
                "4. Copy key (starts with gsk_...)\n"
                "5. Open .env file\n"
                "6. Set GROQ_API_KEY=gsk_your_key_here\n"
                "7. Save .env and run again\n"
                + "=" * 55
            )

        # ── Step 2: Show what we are loading ─────────────
        masked_key = f"gsk_****{GROQ_API_KEY[-6:]}"
        logger.info("=" * 45)
        logger.info("  Loading Groq LLM")
        logger.info(f"  Platform  : Groq (console.groq.com)")
        logger.info(f"  API Key   : {masked_key}")
        logger.info(f"  Model     : {GROQ_MODEL}")
        logger.info(f"  Params    : 120 Billion parameters")
        logger.info("=" * 45)

        # ── Step 3: Create ChatGroq with GPT OSS 120B ────
        try:
            self.llm = ChatGroq(
                api_key     = GROQ_API_KEY,        # ← your identity
                model_name  = GROQ_MODEL,          # ← openai/gpt-oss-120b
                temperature = 0.6,                 # GPT OSS works best at 0.6
                max_tokens  = 600,
            )

            self.provider = "groq"
            self.model    = GROQ_MODEL

            logger.success("=" * 45)
            logger.success("  GPT OSS 120B Ready!")
            logger.success(f"  Model    : {GROQ_MODEL}")
            logger.success(f"  Provider : Groq")
            logger.success("=" * 45)

        except Exception as e:
            logger.error(f"Failed to load Groq LLM: {e}")
            raise

    def get_llm(self) -> ChatGroq:
        """Return LangChain ChatGroq object for use in chains."""
        return self.llm


# ── Module-level singleton ────────────────────────────────────────────────────
llm_provider = LLMProvider()