"""Production model loading and inference logic for fraud detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

try:
    from ..config import settings
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import settings

from preprocessing_utils import ProductionPreprocessor


logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """Fraud detection inference wrapper.

    This class loads the trained model and the saved preprocessing artifacts,
    then applies the same preprocessing pipeline used during training before
    producing a fraud score.
    """

    EXPECTED_FIELDS: List[str] = [
        "transaction_id",
        "user_id",
        "amount",
        "transaction_type",
        "merchant_category",
        "timestamp",
        "transaction_frequency",
        "avg_user_amount",
        "deviation_from_avg",
        "transaction_gap_seconds",
        "account_age_days",
        "failed_attempts",
        "device_type",
        "location",
        "is_foreign_transaction",
        "unusual_amount_flag",
        "velocity_flag",
        "new_device_flag",
        "location_change_flag",
        "night_transaction_flag",
    ]

    def __init__(self) -> None:
        self.model = None
        self.preprocessor: ProductionPreprocessor | None = None
        self.threshold = float(settings.decision_threshold)
        self.model_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model and preprocessing artifacts from disk."""
        try:
            model_path = Path(settings.model_path)
            models_dir = model_path.parent

            self.model = joblib.load(model_path)
            self.preprocessor = ProductionPreprocessor(models_dir=str(models_dir))

            # If the training pipeline saved a tuned threshold in metadata, use it.
            if self.preprocessor.metadata and "best_threshold" in self.preprocessor.metadata:
                self.threshold = float(self.preprocessor.metadata["best_threshold"])

            self.model_loaded = True
            logger.info("Fraud model and preprocessing artifacts loaded successfully")
        except FileNotFoundError as exc:
            logger.error("Model artifact not found: %s", exc)
            self.model_loaded = False
        except Exception as exc:
            logger.exception("Error loading fraud model: %s", exc)
            self.model_loaded = False

    def validate_features(self, features: Dict[str, Any]) -> None:
        """Ensure the request payload matches the expected raw feature schema.

        This prevents silent feature drift, missing fields, or accidental use of
        a request body that does not match the training pipeline.
        """
        incoming_fields = set(features.keys())
        expected_fields = set(self.EXPECTED_FIELDS)

        missing = sorted(expected_fields - incoming_fields)
        extra = sorted(incoming_fields - expected_fields)

        if missing or extra:
            raise ValueError(
                "Feature mismatch detected. "
                f"Missing fields: {missing}. "
                f"Unexpected fields: {extra}."
            )

    def _risk_category(self, probability: float) -> str:
        """Convert a fraud probability into a human-readable risk band."""
        if probability >= 0.85:
            return "Critical"
        if probability >= 0.65:
            return "High"
        if probability >= 0.35:
            return "Medium"
        return "Low"

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single-transaction fraud prediction."""
        if not self.model_loaded or self.model is None or self.preprocessor is None:
            raise ValueError("Fraud model is not loaded")

        self.validate_features(features)

        try:
            df = pd.DataFrame([features])
            processed_df = self.preprocessor.transform(df)

            probability = float(self.model.predict_proba(processed_df)[0][1])
            prediction = "Fraud" if probability >= self.threshold else "Legitimate"
            confidence = float(max(probability, 1.0 - probability))

            return {
                "prediction": prediction,
                "fraud_probability": probability,
                "risk_score": probability,
                "risk_category": self._risk_category(probability),
                "confidence": confidence,
            }
        except Exception as exc:
            logger.exception("Error making fraud prediction: %s", exc)
            raise


# Global model instance used by the API.
fraud_model = FraudDetectionModel()
