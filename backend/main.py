"""FastAPI application entry point for the fraud detection backend."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from .config import settings
    from .routes import router as prediction_router
    from .schemas import HealthCheckResponse
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import settings
    from routes import router as prediction_router
    from schemas import HealthCheckResponse


APP_VERSION = "1.0.0"


def configure_logging() -> None:
    """Configure logging for console and file output."""
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


configure_logging()
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Fraud Detection API",
    description="Production-ready FastAPI backend for fraud prediction and risk scoring.",
    version=APP_VERSION,
    debug=settings.debug,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    """Log startup status and model paths."""
    logger.info("Starting Fraud Detection API...")
    logger.info("Debug mode: %s", settings.debug)
    logger.info("Model path: %s", settings.model_path)
    logger.info("Scaler path: %s", settings.scaler_path)
    logger.info("Decision threshold: %.3f", settings.decision_threshold)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Log shutdown status."""
    logger.info("Shutting down Fraud Detection API...")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a clear 422 error for malformed or incomplete requests."""
    logger.warning("Request validation failed: %s", exc)
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request payload.",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Catch unexpected server errors and avoid leaking internals."""
    logger.exception("Unhandled API error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
        },
    )


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """Return a small welcome payload."""
    return {
        "message": "Fraud Detection API is running",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["health"])
async def health_check() -> HealthCheckResponse:
    """Report API and model readiness."""
    from models import fraud_model

    return HealthCheckResponse(
        status="healthy",
        message="Fraud Detection API is operational",
        model_loaded=fraud_model.model_loaded,
        api_version=APP_VERSION,
    )


# Keep the prediction endpoint at /predict, as requested.
app.include_router(prediction_router)


__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )