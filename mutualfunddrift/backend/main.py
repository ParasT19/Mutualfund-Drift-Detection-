"""
MutualFundDrift — FastAPI application entry point.
Configures CORS, lifespan, routers, health check, and global exception handlers.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.database import check_db_connection, create_db_and_tables
from backend.routers import alerts, drift, funds

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup and shutdown logic
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    FastAPI lifespan context manager.
    Initialises the database tables on startup and logs the service start.
    """
    logger.info("Starting MutualFundDrift API v1.0.0 ...")
    try:
        create_db_and_tables()
        logger.info("Database tables initialised successfully.")
    except Exception as exc:
        logger.error("Database initialisation failed: %s", exc)
        # Allow the app to start even if DB is temporarily unavailable
    yield
    logger.info("MutualFundDrift API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app instance
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MutualFundDrift API",
    version="1.0.0",
    description=(
        "Indian Mutual Fund Style Drift Detector — "
        "Detects mandate drift using AMFI portfolio data, Morningstar-style box coordinates, "
        "XGBoost prediction, SHAP explainability, and LLM investor alerts."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS middleware
# ─────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(funds.router, prefix="/api/funds")
app.include_router(drift.router, prefix="/api/drift")
app.include_router(alerts.router, prefix="/api/alerts")


# ─────────────────────────────────────────────────────────────────────────────
# Core endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["root"])
def root() -> dict:
    """Return a welcome message with links to the interactive API documentation."""
    return {
        "service": "MutualFundDrift API",
        "version": "1.0.0",
        "description": "Indian Mutual Fund Style Drift Detector & Early Warning System",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.get("/health", tags=["health"])
def health_check() -> dict:
    """
    Return the operational health status of the API, database, and ML model.

    Checks:
      - Database connectivity (via a lightweight SELECT 1 query)
      - ML model file existence at the configured model_save_path
    """
    db_connected = check_db_connection()
    model_loaded = os.path.exists(settings.model_save_path)

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "db_connected": db_connected,
        "version": "1.0.0",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Global exception handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle ValueError exceptions with an HTTP 422 Unprocessable Entity response."""
    logger.warning("ValueError at %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "ValueError"},
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    """Handle FileNotFoundError exceptions with an HTTP 404 Not Found response."""
    logger.warning("FileNotFoundError at %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "type": "FileNotFoundError"},
    )
