"""
Production Preprocessing Module for Fraud Detection

This module provides utilities for applying the trained preprocessing pipeline
to new data in production. It loads saved artifacts (scaler and encoders) and
applies the same transformations that were used during training.

This ensures that:
1. New predictions use identical preprocessing
2. Model predictions are consistent and reliable
3. Easy integration with FastAPI backend for real-time predictions
4. Batch processing capability for multiple transactions

Usage:
    >>> from preprocessing_utils import ProductionPreprocessor
    >>> 
    >>> preprocessor = ProductionPreprocessor()
    >>> new_data = pd.read_csv('new_transactions.csv')
    >>> preprocessed_data = preprocessor.transform(new_data)
    >>> predictions = model.predict(preprocessed_data)
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class ProductionPreprocessor:
    """
    Production preprocessing transformer for fraud detection.
    
    Applies the same preprocessing pipeline used during training to new data.
    """
    
    def __init__(self, models_dir: str = './models'):
        """
        Initialize production preprocessor with saved artifacts.
        
        Args:
            models_dir (str): Directory containing saved scaler and encoders
        """
        self.models_dir = models_dir
        self.scaler = None
        self.encoder_dict = None
        self.metadata = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load preprocessing artifacts from disk."""
        try:
            # Load scaler
            scaler_path = f'{self.models_dir}/scaler.pkl'
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✓ Scaler loaded from {scaler_path}")
            
            # Load encoders
            encoder_path = f'{self.models_dir}/encoders.pkl'
            self.encoder_dict = joblib.load(encoder_path)
            logger.info(f"✓ Encoders loaded from {encoder_path}")
            
            # Load metadata
            metadata_path = f'{self.models_dir}/preprocessing_metadata.pkl'
            if Path(metadata_path).exists():
                self.metadata = joblib.load(metadata_path)
                logger.info(f"✓ Metadata loaded from {metadata_path}")
        
        except FileNotFoundError as e:
            logger.error(f"Preprocessing artifact not found: {e}")
            raise
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing pipeline to new data.
        
        Args:
            df (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Preprocessed data ready for model inference
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Drop ID columns
        id_columns = ['transaction_id', 'user_id']
        df = df.drop(columns=[col for col in id_columns if col in df.columns])
        
        # Extract time features
        if 'timestamp' in df.columns:
            df = self._extract_time_features(df)
        
        # Encode categorical features
        df = self._encode_categorical_features(df, is_training=False)
        
        # Scale numerical features
        df = self._scale_numerical_features(df)
        
        return df
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time features from timestamp column."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night_time'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['quarter'] = df['timestamp'].dt.quarter
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        df = df.drop(columns=['timestamp'])
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Apply saved categorical encoders."""
        for col, encoder_info in self.encoder_dict.items():
            if col not in df.columns:
                continue
            
            if encoder_info['type'] == 'onehot':
                # One-hot encoding
                one_hot = pd.get_dummies(df[col], prefix=col, drop_first=False)
                expected_cols = [f"{col}_{cat}" for cat in encoder_info['categories']]
                
                # Ensure all training columns are present
                for expected_col in expected_cols:
                    if expected_col not in one_hot.columns:
                        one_hot[expected_col] = 0
                
                # Remove unexpected columns
                one_hot = one_hot[expected_cols]
                
                df = pd.concat([df, one_hot], axis=1)
                df = df.drop(columns=[col])
            
            elif encoder_info['type'] == 'label':
                # Label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.classes_ = np.array(encoder_info['classes'])
                
                # Handle unknown categories
                df[col] = df[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply saved scaler to numerical features."""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target if present
        if 'is_fraud' in numerical_cols:
            numerical_cols.remove('is_fraud')
        
        # Get intersection of available and expected columns
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        if available_cols:
            df[available_cols] = self.scaler.transform(df[available_cols])
        
        return df


def preprocess_single_transaction(
    transaction_data: Dict[str, Any],
    models_dir: str = './models'
) -> np.ndarray:
    """
    Preprocess a single transaction for real-time prediction.
    
    Args:
        transaction_data (Dict): Single transaction as dictionary
        models_dir (str): Directory containing preprocessing artifacts
        
    Returns:
        np.ndarray: Preprocessed features ready for model
    """
    # Convert to DataFrame
    df = pd.DataFrame([transaction_data])
    
    # Apply preprocessing
    preprocessor = ProductionPreprocessor(models_dir=models_dir)
    df_processed = preprocessor.transform(df)
    
    return df_processed.values[0]


def preprocess_batch(
    transactions_file: str,
    models_dir: str = './models'
) -> pd.DataFrame:
    """
    Preprocess a batch of transactions.
    
    Args:
        transactions_file (str): Path to CSV file with transactions
        models_dir (str): Directory containing preprocessing artifacts
        
    Returns:
        pd.DataFrame: Preprocessed transactions
    """
    # Load data
    df = pd.read_csv(transactions_file)
    
    # Apply preprocessing
    preprocessor = ProductionPreprocessor(models_dir=models_dir)
    df_processed = preprocessor.transform(df)
    
    return df_processed


if __name__ == "__main__":
    """
    Example usage of production preprocessor.
    """
    # Example single transaction
    example_transaction = {
        'transaction_id': 'TXN_99999',
        'user_id': 'U0001',
        'amount': 500.0,
        'transaction_type': 'transfer',
        'merchant_category': 'electronics',
        'timestamp': '2026-05-17 14:30:00',
        'transaction_frequency': 5,
        'avg_user_amount': 450.0,
        'deviation_from_avg': 50.0,
        'transaction_gap_seconds': 3600,
        'account_age_days': 500,
        'failed_attempts': 0,
        'device_type': 'web',
        'location': 'USA',
        'is_foreign_transaction': 0,
        'unusual_amount_flag': 0,
        'velocity_flag': 0,
        'new_device_flag': 0,
        'location_change_flag': 0,
        'night_transaction_flag': 0,
    }
    
    # Preprocess
    features = preprocess_single_transaction(example_transaction)
    print("Preprocessed features:", features)
