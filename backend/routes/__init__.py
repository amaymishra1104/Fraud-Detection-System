"""Prediction API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

try:
    from ..models import fraud_model
    from ..schemas import BatchPredictionRequest, PredictionResponse, TransactionRequest
except ImportError:  # pragma: no cover - fallback for direct script execution
    from models import fraud_model
    from schemas import BatchPredictionRequest, PredictionResponse, TransactionRequest


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_transaction(request: TransactionRequest) -> PredictionResponse:
    """Predict whether a single transaction is fraudulent.

    The request body must match the raw features used during training so the
    backend can apply the exact same preprocessing pipeline.
    """
    try:
        if not fraud_model.model_loaded:
            raise HTTPException(status_code=503, detail="Fraud model is not loaded")

        # Pydantic v2 uses model_dump() to convert the validated request into a dict.
        features = request.model_dump()
        result = fraud_model.predict(features)

        return PredictionResponse(
            transaction_id=request.transaction_id,
            **result,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Validation error during prediction: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc


@router.post("/predict-batch")
async def predict_batch(request: BatchPredictionRequest) -> dict:
    """Predict fraud risk for a batch of transactions."""
    try:
        predictions = []
        for transaction in request.transactions:
            result = await predict_transaction(transaction)
            predictions.append(result.model_dump())

        return {
            "status": "success",
            "count": len(predictions),
            "predictions": predictions,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal batch prediction error") from exc
