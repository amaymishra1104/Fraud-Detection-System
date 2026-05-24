"""Request and response schemas for the fraud detection API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """Raw transaction payload expected by the trained preprocessing pipeline.

    The field names here mirror the training data so the backend can apply the
    exact same preprocessing steps without feature drift or column mismatch.
    """

    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="Unique user identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    transaction_type: str = Field(..., description="Transaction type")
    merchant_category: str = Field(..., description="Merchant category")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp")
    transaction_frequency: float = Field(..., ge=0, description="User transaction frequency")
    avg_user_amount: float = Field(..., ge=0, description="User average transaction amount")
    deviation_from_avg: float = Field(..., description="Deviation from user average amount")
    transaction_gap_seconds: float = Field(..., ge=0, description="Seconds since previous transaction")
    account_age_days: float = Field(..., ge=0, description="Account age in days")
    failed_attempts: int = Field(..., ge=0, description="Failed transaction attempts")
    device_type: str = Field(..., description="Device type")
    location: str = Field(..., description="Transaction location")
    is_foreign_transaction: int = Field(..., ge=0, le=1, description="Foreign transaction flag")
    unusual_amount_flag: int = Field(..., ge=0, le=1, description="Unusual amount flag")
    velocity_flag: int = Field(..., ge=0, le=1, description="Velocity flag")
    new_device_flag: int = Field(..., ge=0, le=1, description="New device flag")
    location_change_flag: int = Field(..., ge=0, le=1, description="Location change flag")
    night_transaction_flag: int = Field(..., ge=0, le=1, description="Night transaction flag")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "transaction_id": "TXN_00001",
                "user_id": "U001",
                "amount": 249.99,
                "transaction_type": "payment",
                "merchant_category": "electronics",
                "timestamp": "2026-05-17T10:30:00",
                "transaction_frequency": 8,
                "avg_user_amount": 180.0,
                "deviation_from_avg": 69.99,
                "transaction_gap_seconds": 3600,
                "account_age_days": 540,
                "failed_attempts": 0,
                "device_type": "web",
                "location": "USA",
                "is_foreign_transaction": 0,
                "unusual_amount_flag": 0,
                "velocity_flag": 0,
                "new_device_flag": 0,
                "location_change_flag": 0,
                "night_transaction_flag": 0,
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response schema for a single fraud prediction."""

    transaction_id: str = Field(..., description="Transaction identifier")
    prediction: str = Field(..., description="Fraud prediction label")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score in the range 0 to 1")
    risk_category: str = Field(..., description="Human-readable risk band")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN_00001",
                "prediction": "Legitimate",
                "fraud_probability": 0.12,
                "risk_score": 0.12,
                "risk_category": "Low",
                "confidence": 0.88,
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""

    transactions: list[TransactionRequest] = Field(..., description="List of transactions")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "transactions": [
                    {
                        "transaction_id": "TXN_00001",
                        "user_id": "U001",
                        "amount": 249.99,
                        "transaction_type": "payment",
                        "merchant_category": "electronics",
                        "timestamp": "2026-05-17T10:30:00",
                        "transaction_frequency": 8,
                        "avg_user_amount": 180.0,
                        "deviation_from_avg": 69.99,
                        "transaction_gap_seconds": 3600,
                        "account_age_days": 540,
                        "failed_attempts": 0,
                        "device_type": "web",
                        "location": "USA",
                        "is_foreign_transaction": 0,
                        "unusual_amount_flag": 0,
                        "velocity_flag": 0,
                        "new_device_flag": 0,
                        "location_change_flag": 0,
                        "night_transaction_flag": 0,
                    }
                ]
            }
        }
    )


class HealthCheckResponse(BaseModel):
    """Health check response schema."""

    status: str
    message: str
    model_loaded: bool = False
    api_version: str = "1.0.0"
