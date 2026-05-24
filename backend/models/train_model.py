"""
Production-grade fraud detection model training pipeline.

This script:
- Loads processed train/test datasets
- Trains Logistic Regression, Random Forest, and XGBoost
- Handles class imbalance using class weights and threshold tuning
- Compares models using accuracy, precision, recall, F1-score, and ROC-AUC
- Displays and saves confusion matrices
- Saves feature importance / coefficient importance plots
- Produces a leaderboard table for all models
- Selects the best model prioritizing fraud recall, then precision
- Saves the best model to models/fraud_model.pkl
- Saves evaluation metrics as JSON

The processed datasets are expected to come from the preprocessing pipeline
already present in the project.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TARGET_COLUMN = "is_fraud"
BEST_MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
METRICS_JSON_PATH = MODELS_DIR / "fraud_model_metrics.json"
LEADERBOARD_CSV_PATH = MODELS_DIR / "fraud_model_leaderboard.csv"

RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
	"""Configure file and console logging for training."""
	MODELS_DIR.mkdir(parents=True, exist_ok=True)
	PLOTS_DIR.mkdir(parents=True, exist_ok=True)

	logger = logging.getLogger("fraud_training")
	logger.setLevel(logging.INFO)
	logger.propagate = False

	if not logger.handlers:
		formatter = logging.Formatter(
			"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
		)

		file_handler = logging.FileHandler(MODELS_DIR / "train_model.log", encoding="utf-8")
		file_handler.setFormatter(formatter)

		console_handler = logging.StreamHandler()
		console_handler.setFormatter(formatter)

		logger.addHandler(file_handler)
		logger.addHandler(console_handler)

	return logger


logger = setup_logging()


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------


@dataclass
class ModelResult:
	"""Container for a single model's validation and test results."""

	name: str
	model: BaseEstimator
	threshold: float
	val_accuracy: float
	val_precision: float
	val_recall: float
	val_f1_score: float
	val_roc_auc: float
	test_accuracy: float
	test_precision: float
	test_recall: float
	test_f1_score: float
	test_roc_auc: float
	test_confusion_matrix: np.ndarray
	test_classification_report: str
	validation_metrics: Dict[str, float]
	test_metrics: Dict[str, float]


# ----------------------------------------------------------------------------
# Data loading and validation
# ----------------------------------------------------------------------------


def load_processed_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Load processed train and test datasets."""
	logger.info("Loading processed datasets...")
	if not train_path.exists():
		raise FileNotFoundError(f"Training dataset not found: {train_path}")
	if not test_path.exists():
		raise FileNotFoundError(f"Test dataset not found: {test_path}")

	train_df = pd.read_csv(train_path)
	test_df = pd.read_csv(test_path)

	logger.info("Train shape: %s", train_df.shape)
	logger.info("Test shape: %s", test_df.shape)
	return train_df, test_df


def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
	"""Split a dataframe into feature matrix and target vector."""
	if target_column not in df.columns:
		raise ValueError(f"Target column '{target_column}' not found in dataframe.")

	X = df.drop(columns=[target_column])
	y = df[target_column].astype(int)
	return X, y


def validate_binary_target(y: pd.Series, dataset_name: str) -> None:
	"""Ensure the target is binary with values 0 and 1."""
	unique_values = sorted(y.dropna().unique().tolist())
	if unique_values != [0, 1]:
		raise ValueError(
			f"{dataset_name} target must contain only [0, 1]. Found: {unique_values}"
		)


def validate_feature_alignment(train_features: pd.DataFrame, test_features: pd.DataFrame) -> None:
	"""Ensure train and test feature sets are aligned."""
	if train_features.columns.tolist() != test_features.columns.tolist():
		raise ValueError("Train and test feature columns do not match exactly.")


# ----------------------------------------------------------------------------
# Imbalance helpers
# ----------------------------------------------------------------------------


def compute_balanced_class_weights(y: pd.Series) -> Dict[int, float]:
	"""Compute class weights for the current training labels."""
	classes = np.array(sorted(y.unique()))
	weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
	return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def compute_scale_pos_weight(y: pd.Series) -> float:
	"""Compute XGBoost scale_pos_weight for imbalanced binary classification."""
	negative = int((y == 0).sum())
	positive = int((y == 1).sum())
	if positive == 0:
		return 1.0
	return negative / positive


def build_models(y_train: pd.Series, random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
	"""Build the candidate models to compare."""
	class_weights = compute_balanced_class_weights(y_train)
	scale_pos_weight = compute_scale_pos_weight(y_train)

	return {
		"logistic_regression": LogisticRegression(
			max_iter=3000,
			class_weight=class_weights,
			solver="liblinear",
			random_state=random_state,
		),
		"random_forest": RandomForestClassifier(
			n_estimators=400,
			class_weight=class_weights,
			random_state=random_state,
			n_jobs=-1,
		),
		"xgboost": XGBClassifier(
			n_estimators=450,
			learning_rate=0.05,
			max_depth=5,
			subsample=0.9,
			colsample_bytree=0.9,
			reg_lambda=1.0,
			objective="binary:logistic",
			eval_metric="logloss",
			scale_pos_weight=scale_pos_weight,
			random_state=random_state,
			tree_method="hist",
		),
	}


# ----------------------------------------------------------------------------
# Threshold tuning and evaluation
# ----------------------------------------------------------------------------


def tune_threshold(
	y_true: pd.Series,
	y_proba: np.ndarray,
	min_threshold: float = 0.05,
	max_threshold: float = 0.95,
	step: float = 0.01,
) -> Tuple[float, Dict[str, float]]:
	"""Tune the classification threshold for fraud detection.

	Priority is recall first, then precision, then F1.
	"""
	best_threshold = 0.5
	best_score = (-1.0, -1.0, -1.0)
	best_metrics = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

	for threshold in np.arange(min_threshold, max_threshold + 1e-9, step):
		y_pred = (y_proba >= threshold).astype(int)
		precision = precision_score(y_true, y_pred, zero_division=0)
		recall = recall_score(y_true, y_pred, zero_division=0)
		f1 = f1_score(y_true, y_pred, zero_division=0)

		candidate_score = (recall, precision, f1)
		if candidate_score > best_score:
			best_score = candidate_score
			best_threshold = float(threshold)
			best_metrics = {
				"precision": float(precision),
				"recall": float(recall),
				"f1_score": float(f1),
			}

	return best_threshold, best_metrics


def evaluate_with_threshold(
	model: BaseEstimator,
	X: pd.DataFrame,
	y: pd.Series,
	threshold: float,
) -> Dict[str, Any]:
	"""Evaluate a fitted model using a specific probability threshold."""
	y_proba = model.predict_proba(X)[:, 1]
	y_pred = (y_proba >= threshold).astype(int)

	return {
		"accuracy": float(accuracy_score(y, y_pred)),
		"precision": float(precision_score(y, y_pred, zero_division=0)),
		"recall": float(recall_score(y, y_pred, zero_division=0)),
		"f1_score": float(f1_score(y, y_pred, zero_division=0)),
		"roc_auc": float(roc_auc_score(y, y_proba)),
		"confusion_matrix": confusion_matrix(y, y_pred),
		"classification_report": classification_report(y, y_pred, zero_division=0),
		"probabilities": y_proba,
		"predictions": y_pred,
	}


def print_metric_summary(title: str, metrics: Dict[str, Any], threshold: float) -> None:
	"""Print a readable metric summary to the log."""
	logger.info("")
	logger.info("=" * 90)
	logger.info("%s", title.upper())
	logger.info("Threshold: %.2f", threshold)
	logger.info("=" * 90)
	logger.info("Accuracy : %.4f", metrics["accuracy"])
	logger.info("Precision: %.4f", metrics["precision"])
	logger.info("Recall   : %.4f", metrics["recall"])
	logger.info("F1-score : %.4f", metrics["f1_score"])
	logger.info("ROC-AUC  : %.4f", metrics["roc_auc"])
	logger.info("Confusion Matrix:\n%s", metrics["confusion_matrix"])
	logger.info("Classification Report:\n%s", metrics["classification_report"])


# ----------------------------------------------------------------------------
# Visualizations
# ----------------------------------------------------------------------------


def save_confusion_matrix(
	y_true: pd.Series,
	y_pred: np.ndarray,
	title: str,
	output_path: Path,
) -> None:
	"""Save a confusion matrix heatmap."""
	cm = confusion_matrix(y_true, y_pred)

	plt.figure(figsize=(6, 5))
	sns.heatmap(
		cm,
		annot=True,
		fmt="d",
		cmap="Blues",
		xticklabels=["Legitimate", "Fraud"],
		yticklabels=["Legitimate", "Fraud"],
	)
	plt.title(title)
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def save_feature_importance(
	model: BaseEstimator,
	feature_names: List[str],
	title: str,
	output_path: Path,
	top_n: int = 20,
) -> None:
	"""Save a feature importance or coefficient plot.

	Tree models expose feature_importances_. Logistic Regression exposes coef_.
	"""
	if hasattr(model, "feature_importances_"):
		importance_values = np.asarray(model.feature_importances_)
	elif hasattr(model, "coef_"):
		importance_values = np.abs(np.asarray(model.coef_)).ravel()
	else:
		logger.info("Skipping importance plot for %s because the model does not expose importances.", title)
		return

	top_indices = np.argsort(importance_values)[::-1][:top_n]
	top_features = np.array(feature_names)[top_indices]
	top_values = importance_values[top_indices]

	plt.figure(figsize=(10, 6))
	sns.barplot(x=top_values, y=top_features, palette="viridis")
	plt.title(title)
	plt.xlabel("Importance")
	plt.ylabel("Feature")
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


# ----------------------------------------------------------------------------
# Leaderboard and selection
# ----------------------------------------------------------------------------


def rank_models(results: Dict[str, ModelResult]) -> pd.DataFrame:
	"""Create a leaderboard sorted by validation recall, then precision."""
	rows = []
	for name, result in results.items():
		rows.append(
			{
				"model": name,
				"threshold": result.threshold,
				"val_accuracy": result.val_accuracy,
				"val_precision": result.val_precision,
				"val_recall": result.val_recall,
				"val_f1_score": result.val_f1_score,
				"val_roc_auc": result.val_roc_auc,
				"test_accuracy": result.test_accuracy,
				"test_precision": result.test_precision,
				"test_recall": result.test_recall,
				"test_f1_score": result.test_f1_score,
				"test_roc_auc": result.test_roc_auc,
			}
		)

	leaderboard = pd.DataFrame(rows)
	leaderboard = leaderboard.sort_values(
		by=["val_recall", "val_precision", "val_f1_score", "val_roc_auc"],
		ascending=False,
		kind="mergesort",
	).reset_index(drop=True)
	return leaderboard


def select_best_model(results: Dict[str, ModelResult]) -> str:
	"""Pick the best model using validation recall first, then precision."""
	leaderboard = rank_models(results)
	return str(leaderboard.iloc[0]["model"])


# ----------------------------------------------------------------------------
# Training orchestration
# ----------------------------------------------------------------------------


def train_one_model(
	model_name: str,
	model: BaseEstimator,
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_val: pd.DataFrame,
	y_val: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
) -> ModelResult:
	"""Train, tune, and evaluate a single model."""
	logger.info("Training %s...", model_name)

	# Fit only on the training split.
	model.fit(X_train, y_train)

	# Tune the fraud decision threshold using validation data.
	val_proba = model.predict_proba(X_val)[:, 1]
	best_threshold, validation_metrics = tune_threshold(y_val, val_proba)

	logger.info(
		"%s tuned threshold: %.2f | val recall=%.4f | val precision=%.4f | val f1=%.4f",
		model_name,
		best_threshold,
		validation_metrics["recall"],
		validation_metrics["precision"],
		validation_metrics["f1_score"],
	)

	# Final unbiased evaluation on the held-out test set.
	test_metrics = evaluate_with_threshold(model, X_test, y_test, best_threshold)
	print_metric_summary(f"Model: {model_name}", test_metrics, best_threshold)

	# Save confusion matrix for the test set.
	save_confusion_matrix(
		y_test,
		test_metrics["predictions"],
		title=f"Confusion Matrix - {model_name.replace('_', ' ').title()}",
		output_path=PLOTS_DIR / f"confusion_matrix_{model_name}.png",
	)

	# Save feature importance or coefficient importance.
	save_feature_importance(
		model,
		feature_names=X_train.columns.tolist(),
		title=f"Feature Importance - {model_name.replace('_', ' ').title()}",
		output_path=PLOTS_DIR / f"feature_importance_{model_name}.png",
	)

	return ModelResult(
		name=model_name,
		model=model,
		threshold=best_threshold,
		val_accuracy=float(accuracy_score(y_val, (val_proba >= best_threshold).astype(int))),
		val_precision=float(validation_metrics["precision"]),
		val_recall=float(validation_metrics["recall"]),
		val_f1_score=float(validation_metrics["f1_score"]),
		val_roc_auc=float(roc_auc_score(y_val, val_proba)),
		test_accuracy=test_metrics["accuracy"],
		test_precision=test_metrics["precision"],
		test_recall=test_metrics["recall"],
		test_f1_score=test_metrics["f1_score"],
		test_roc_auc=test_metrics["roc_auc"],
		test_confusion_matrix=test_metrics["confusion_matrix"],
		test_classification_report=test_metrics["classification_report"],
		validation_metrics=validation_metrics,
		test_metrics={
			"accuracy": test_metrics["accuracy"],
			"precision": test_metrics["precision"],
			"recall": test_metrics["recall"],
			"f1_score": test_metrics["f1_score"],
			"roc_auc": test_metrics["roc_auc"],
		},
	)


def retrain_best_model(
	model_name: str,
	y_train_full: pd.Series,
	X_train_full: pd.DataFrame,
) -> BaseEstimator:
	"""Refit the winning model on all processed training data."""
	model = build_models(y_train_full)[model_name]
	model.fit(X_train_full, y_train_full)
	return model


def save_outputs(
	best_model: BaseEstimator,
	best_model_name: str,
	best_threshold: float,
	leaderboard: pd.DataFrame,
	results: Dict[str, ModelResult],
) -> None:
	"""Persist the selected model, leaderboard, and metrics JSON."""
	MODELS_DIR.mkdir(parents=True, exist_ok=True)
	PLOTS_DIR.mkdir(parents=True, exist_ok=True)

	joblib.dump(best_model, BEST_MODEL_PATH)
	logger.info("Saved best model to %s", BEST_MODEL_PATH)

	leaderboard.to_csv(LEADERBOARD_CSV_PATH, index=False)
	logger.info("Saved leaderboard to %s", LEADERBOARD_CSV_PATH)

	metrics_payload = {
		"best_model": best_model_name,
		"best_threshold": best_threshold,
		"leaderboard": leaderboard.to_dict(orient="records"),
		"models": {
			name: {
				"validation": {
					"accuracy": result.val_accuracy,
					"precision": result.val_precision,
					"recall": result.val_recall,
					"f1_score": result.val_f1_score,
					"roc_auc": result.val_roc_auc,
				},
				"test": {
					"accuracy": result.test_accuracy,
					"precision": result.test_precision,
					"recall": result.test_recall,
					"f1_score": result.test_f1_score,
					"roc_auc": result.test_roc_auc,
				},
				"threshold": result.threshold,
				"validation_threshold_metrics": result.validation_metrics,
				"test_confusion_matrix": result.test_confusion_matrix.tolist(),
			}
			for name, result in results.items()
		},
	}

	with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
		json.dump(metrics_payload, f, indent=2)

	logger.info("Saved metrics JSON to %s", METRICS_JSON_PATH)


def print_final_recommendation(best_model_name: str, best_result: ModelResult, leaderboard: pd.DataFrame) -> None:
	"""Print a final recommendation with the reason the model was selected."""
	logger.info("")
	logger.info("=" * 90)
	logger.info("FINAL RECOMMENDATION")
	logger.info("=" * 90)
	logger.info(
		"Selected model: %s because it achieved the strongest fraud recall (%.4f) "
		"while maintaining precision at %.4f using threshold %.2f.",
		best_model_name,
		best_result.val_recall,
		best_result.val_precision,
		best_result.threshold,
	)
	logger.info(
		"Held-out test performance -> accuracy: %.4f | precision: %.4f | recall: %.4f | f1: %.4f | roc_auc: %.4f",
		best_result.test_accuracy,
		best_result.test_precision,
		best_result.test_recall,
		best_result.test_f1_score,
		best_result.test_roc_auc,
	)
	logger.info("Leaderboard:\n%s", leaderboard.to_string(index=False))
	logger.info("=" * 90)


def run_training_pipeline() -> Dict[str, Any]:
	"""Execute the full training workflow end to end."""
	logger.info("Starting fraud model training pipeline...")
	logger.info("Project root: %s", PROJECT_ROOT)

	train_df, test_df = load_processed_data(TRAIN_PATH, TEST_PATH)

	X_train_full, y_train_full = split_features_target(train_df, TARGET_COLUMN)
	X_test, y_test = split_features_target(test_df, TARGET_COLUMN)

	validate_binary_target(y_train_full, "Training")
	validate_binary_target(y_test, "Test")
	validate_feature_alignment(X_train_full, X_test)

	# Split the processed training data again so threshold tuning uses validation
	# data rather than the final held-out test set.
	X_train, X_val, y_train, y_val = train_test_split(
		X_train_full,
		y_train_full,
		test_size=0.2,
		stratify=y_train_full,
		random_state=RANDOM_STATE,
	)

	logger.info("Inner train shape: %s", X_train.shape)
	logger.info("Validation shape: %s", X_val.shape)
	logger.info("Test shape: %s", X_test.shape)

	models = build_models(y_train)
	results: Dict[str, ModelResult] = {}

	for model_name, model in models.items():
		result = train_one_model(
			model_name=model_name,
			model=model,
			X_train=X_train,
			y_train=y_train,
			X_val=X_val,
			y_val=y_val,
			X_test=X_test,
			y_test=y_test,
		)
		results[model_name] = result

	leaderboard = rank_models(results)
	best_model_name = select_best_model(results)
	best_result = results[best_model_name]

	# Retrain the chosen model on the full processed training data before saving.
	logger.info("Retraining winning model on the full training dataset...")
	final_model = retrain_best_model(best_model_name, y_train_full, X_train_full)

	save_outputs(
		best_model=final_model,
		best_model_name=best_model_name,
		best_threshold=best_result.threshold,
		leaderboard=leaderboard,
		results=results,
	)

	print_final_recommendation(best_model_name, best_result, leaderboard)

	return {
		"best_model_name": best_model_name,
		"best_threshold": best_result.threshold,
		"best_model": final_model,
		"leaderboard": leaderboard,
		"results": results,
		"X_test": X_test,
		"y_test": y_test,
	}


def main() -> None:
	"""Script entry point."""
	try:
		output = run_training_pipeline()
		logger.info("Training completed successfully.")
		logger.info("Best model: %s", output["best_model_name"])
		logger.info("Saved model path: %s", BEST_MODEL_PATH)
	except Exception as exc:
		logger.exception("Training pipeline failed: %s", exc)
		raise


if __name__ == "__main__":
	main()
