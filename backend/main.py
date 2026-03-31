"""
FastAPI Application Entry Point
- Registers all routes
- Handles startup/shutdown
- CORS enabled for frontend
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR))

from backend.database import connect_db, close_db
from backend.routes.predict      import router as predict_router
from backend.routes.transactions import router as transactions_router
from backend.routes.stats        import router as stats_router
from backend.routes.explain import router as explain_router
# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "🔐 Fraud Detection API",
    description = "Real-time fraud detection powered by XGBoost, RAG, and LLM",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS (allow frontend access) ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Startup / Shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("Starting Fraud Detection API...")
    await connect_db()
    logger.success("API ready.")


@app.on_event("shutdown")
async def shutdown():
    await close_db()
    logger.info("API shut down.")


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(predict_router)
app.include_router(transactions_router)
app.include_router(stats_router)
app.include_router(explain_router)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "status"  : "running",
        "service" : "Fraud Detection API",
        "version" : "1.0.0",
        "docs"    : "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host    = os.getenv("APP_HOST", "0.0.0.0"),
        port    = int(os.getenv("APP_PORT", 8000)),
        reload  = True,
        log_level = "info",
    )