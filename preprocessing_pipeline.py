"""
Comprehensive Data Preprocessing Pipeline for Fraud Detection

This module provides a complete, modular preprocessing pipeline for the fraud detection
dataset. It handles data loading, cleaning, feature engineering, encoding, scaling, and
class imbalance management.

Features:
    - Categorical encoding (one-hot and label encoding)
    - Time-based feature extraction from timestamps
    - ID column removal
    - Numerical feature scaling
    - Class imbalance handling using SMOTE
    - Train-test split with stratification
    - Joblib model persistence for production use

Author: Data Science Team
Date: 2024-05-17
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

# Preprocessing and ML libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging() -> logging.Logger:
    """
    Configure logging for the preprocessing pipeline.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


logger = setup_logging()


# ============================================================================
# STAGE 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load fraud detection dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.ParserError: If file is not valid CSV
    """
    logger.info(f"Loading data from {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully! Shape: {df.shape}")
        return df
    
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise


def explore_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform initial data exploration and logging.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict[str, Any]: Dictionary containing exploration metrics
    """
    logger.info("=" * 80)
    logger.info("DATA EXPLORATION")
    logger.info("=" * 80)
    
    exploration_info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }
    
    logger.info(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Duplicate Rows: {exploration_info['duplicates']}")
    logger.info(f"Missing Values:\n{df.isnull().sum()}")
    logger.info(f"Target Variable Distribution:\n{df['is_fraud'].value_counts()}")
    logger.info(f"Target Variable Percentage:\n{df['is_fraud'].value_counts(normalize=True) * 100}")
    
    return exploration_info


# ============================================================================
# STAGE 2: ID COLUMN REMOVAL
# ============================================================================

def drop_id_columns(df: pd.DataFrame, id_columns: list = None) -> pd.DataFrame:
    """
    Remove ID columns that don't contribute to prediction.
    
    ID columns are identifiers that have no predictive power and should be
    removed before model training. In this dataset, transaction_id and user_id
    are pure identifiers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        id_columns (list, optional): List of ID column names. 
                                     Defaults to ['transaction_id', 'user_id']
        
    Returns:
        pd.DataFrame: Dataframe with ID columns removed
    """
    if id_columns is None:
        id_columns = ['transaction_id', 'user_id']
    
    # Filter to only columns that exist in the dataframe
    columns_to_drop = [col for col in id_columns if col in df.columns]
    
    if columns_to_drop:
        logger.info(f"Dropping ID columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
        logger.info(f"Dataframe shape after dropping IDs: {df.shape}")
    
    return df


# ============================================================================
# STAGE 3: TIMESTAMP FEATURE EXTRACTION
# ============================================================================

def extract_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Extract useful time-based features from timestamp column.
    
    Time-based features are valuable for fraud detection because fraudsters
    often operate at specific times (night transactions, weekend activity, etc.).
    This function extracts:
    
    - Hour: Transaction hour (0-23) - useful for detecting night transactions
    - Day of Week: Day of the week (0-6) - fraudsters may operate on weekends
    - Month: Month of the year (1-12) - seasonal fraud patterns
    - Day of Month: Day in month (1-31) - month-end fraud patterns
    - Is Weekend: Binary flag for weekend transactions
    - Is Night Time: Binary flag for night transactions (22:00-06:00)
    - Quarter: Business quarter (1-4)
    - Week of Year: Week number (1-52)
    
    Args:
        df (pd.DataFrame): Input dataframe
        timestamp_col (str): Name of timestamp column. Default: 'timestamp'
        
    Returns:
        pd.DataFrame: Dataframe with extracted time features
    """
    logger.info(f"Extracting time-based features from '{timestamp_col}' column...")
    
    # Convert to datetime if not already
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extract individual time components
    df['hour'] = df[timestamp_col].dt.hour
    logger.info(f"  ✓ Extracted hour (0-23)")
    
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    logger.info(f"  ✓ Extracted day_of_week (0=Monday, 6=Sunday)")
    
    df['month'] = df[timestamp_col].dt.month
    logger.info(f"  ✓ Extracted month (1-12)")
    
    df['day_of_month'] = df[timestamp_col].dt.day
    logger.info(f"  ✓ Extracted day_of_month (1-31)")
    
    # Create binary flags for specific time patterns
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    logger.info(f"  ✓ Created is_weekend flag (Saturday-Sunday)")
    
    df['is_night_time'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    logger.info(f"  ✓ Created is_night_time flag (22:00-06:00)")
    
    df['quarter'] = df[timestamp_col].dt.quarter
    logger.info(f"  ✓ Extracted quarter (1-4)")
    
    df['week_of_year'] = df[timestamp_col].dt.isocalendar().week
    logger.info(f"  ✓ Extracted week_of_year (1-52)")
    
    # Drop original timestamp column as it's no longer needed
    df = df.drop(columns=[timestamp_col])
    logger.info(f"Dropped original '{timestamp_col}' column")
    logger.info(f"Time feature extraction complete! New features: 8")
    
    return df


# ============================================================================
# STAGE 4: CATEGORICAL FEATURE ENCODING
# ============================================================================

def identify_categorical_columns(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify categorical and numerical columns in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Tuple[list, list]: (categorical_columns, numerical_columns)
    """
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    numerical = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
    
    # Remove target variable from numerical if present
    if 'is_fraud' in numerical:
        numerical.remove('is_fraud')
    
    return categorical, numerical


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: list = None,
    encoder_dict: Dict = None,
    is_training: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical features using one-hot encoding.
    
    One-hot encoding converts categorical variables into binary features,
    which is required by most machine learning algorithms. For example:
    - transaction_type: 'transfer' → [1, 0, 0], 'payment' → [0, 1, 0]
    - device_type: 'mobile' → [1, 0], 'web' → [0, 1]
    
    This function:
    1. Identifies high-cardinality categorical features (many unique values)
    2. Uses one-hot encoding for low-cardinality features (≤ 10 unique values)
    3. Uses label encoding for high-cardinality features (> 10 unique values)
    4. Stores encoder information for consistent transformation on test data
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_columns (list, optional): List of categorical column names.
                                              Auto-detected if None
        encoder_dict (Dict, optional): Saved encoder dictionary from training set.
                                       Used for test set transformation.
        is_training (bool): Whether this is training data. If True, encoders
                           are created. If False, saved encoders are used.
        
    Returns:
        Tuple[pd.DataFrame, Dict]: (encoded_dataframe, encoder_dictionary)
    """
    if categorical_columns is None:
        categorical_columns, _ = identify_categorical_columns(df)
    
    logger.info(f"Encoding {len(categorical_columns)} categorical columns...")
    logger.info(f"Categorical columns: {categorical_columns}")
    
    if encoder_dict is None:
        encoder_dict = {}
    
    # Process each categorical column
    for col in categorical_columns:
        n_unique = df[col].nunique()
        logger.info(f"  Processing '{col}': {n_unique} unique values")
        
        # Strategy: Use one-hot encoding for low-cardinality, label encoding for high
        if n_unique <= 10:
            # One-hot encoding for low-cardinality features
            logger.info(f"    → Using one-hot encoding")
            
            if is_training:
                # During training: create and apply encoding
                one_hot = pd.get_dummies(df[col], prefix=col, drop_first=False)
                encoder_dict[col] = {
                    'type': 'onehot',
                    'categories': df[col].unique().tolist()
                }
            else:
                # During testing: apply saved encoding
                one_hot = pd.get_dummies(df[col], prefix=col, drop_first=False)
                # Ensure all training categories are present
                for cat in encoder_dict[col]['categories']:
                    col_name = f"{col}_{cat}"
                    if col_name not in one_hot.columns:
                        one_hot[col_name] = 0
                # Remove extra columns that weren't in training
                expected_cols = [f"{col}_{cat}" for cat in encoder_dict[col]['categories']]
                one_hot = one_hot[expected_cols]
            
            df = pd.concat([df, one_hot], axis=1)
            df = df.drop(columns=[col])
        
        else:
            # Label encoding for high-cardinality features
            logger.info(f"    → Using label encoding")
            
            if is_training:
                # During training: create and apply encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoder_dict[col] = {
                    'type': 'label',
                    'classes': le.classes_.tolist()
                }
            else:
                # During testing: apply saved encoding
                le = LabelEncoder()
                le.classes_ = np.array(encoder_dict[col]['classes'])
                # Handle unknown categories by mapping to -1
                df[col] = df[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    logger.info(f"Categorical encoding complete!")
    logger.info(f"New dataframe shape: {df.shape}")
    
    return df, encoder_dict


# ============================================================================
# STAGE 5: FEATURE SCALING
# ============================================================================

def scale_numerical_features(
    df: pd.DataFrame,
    numerical_columns: list = None,
    scaler: StandardScaler = None,
    is_training: bool = True
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    
    Feature scaling is crucial because:
    1. Algorithms like SVM and KNN use distance metrics (affected by scale)
    2. Gradient descent optimization works better with normalized features
    3. Prevents features with larger ranges from dominating
    
    StandardScaler transforms features to have mean=0 and std=1:
    X_scaled = (X - mean) / std
    
    This function:
    1. Identifies numerical columns
    2. Applies StandardScaler (or uses saved scaler for test data)
    3. Persists scaler for consistent transformation on new data
    
    Args:
        df (pd.DataFrame): Input dataframe
        numerical_columns (list, optional): List of numerical column names.
                                            Auto-detected if None
        scaler (StandardScaler, optional): Pre-fitted scaler from training set.
                                          Used for test set scaling.
        is_training (bool): Whether this is training data. If True, scaler is fit.
                           If False, pre-fit scaler is used.
        
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: (scaled_dataframe, fitted_scaler)
    """
    if numerical_columns is None:
        _, numerical_columns = identify_categorical_columns(df)
    
    logger.info(f"Scaling {len(numerical_columns)} numerical columns...")
    logger.info(f"Numerical columns: {numerical_columns}")
    
    if scaler is None:
        scaler = StandardScaler()
    
    # Apply scaling
    if is_training:
        # During training: fit and transform
        logger.info("  Fitting scaler on training data...")
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        logger.info(f"  ✓ Scaler fitted and applied")
    else:
        # During testing: use fitted scaler
        logger.info("  Applying fitted scaler...")
        df[numerical_columns] = scaler.transform(df[numerical_columns])
        logger.info(f"  ✓ Scaler applied")
    
    logger.info(f"Feature scaling complete!")
    logger.info(f"Scaling parameters: mean={np.mean(scaler.mean_):.4f}, "
                f"std={np.mean(scaler.scale_):.4f}")
    
    return df, scaler


# ============================================================================
# STAGE 6: CLASS IMBALANCE HANDLING
# ============================================================================

def analyze_class_imbalance(df: pd.DataFrame, target_col: str = 'is_fraud') -> Dict:
    """
    Analyze class imbalance in the target variable.
    
    Class imbalance is a common problem in fraud detection:
    - Fraud cases (positive class) are typically rare (~0.1-2%)
    - Legitimate transactions (negative class) are common (~98-99.9%)
    
    This imbalance causes issues:
    1. Model biases towards majority class
    2. Minority class performance is poor
    3. Accuracy metric is misleading
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column. Default: 'is_fraud'
        
    Returns:
        Dict: Class distribution statistics
    """
    logger.info("=" * 80)
    logger.info("CLASS IMBALANCE ANALYSIS")
    logger.info("=" * 80)
    
    # Get class distribution
    class_dist = df[target_col].value_counts()
    class_dist_pct = df[target_col].value_counts(normalize=True) * 100
    
    imbalance_ratio = class_dist.iloc[0] / class_dist.iloc[1]
    
    logger.info(f"\nClass Distribution (Counts):")
    logger.info(f"  Legitimate (0): {class_dist[0]:,} transactions ({class_dist_pct[0]:.2f}%)")
    logger.info(f"  Fraud (1):      {class_dist[1]:,} transactions ({class_dist_pct[1]:.2f}%)")
    logger.info(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1 (majority:minority)")
    
    return {
        'class_0_count': class_dist[0],
        'class_1_count': class_dist[1],
        'class_0_pct': class_dist_pct[0],
        'class_1_pct': class_dist_pct[1],
        'imbalance_ratio': imbalance_ratio
    }


def handle_class_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'smote',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
    
    SMOTE is a sophisticated oversampling method that:
    1. Identifies minority class instances
    2. Creates synthetic samples by interpolating between neighbors
    3. Balances the dataset without simple duplication
    4. Improves model robustness to minority class
    
    Alternative methods:
    - Random Oversampling: Simple duplication (can cause overfitting)
    - Random Undersampling: Remove majority samples (loses information)
    - Class Weights: Penalize minority class errors during training
    - ADASYN: Adaptive synthetic sampling
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        method (str): Oversampling method. Default: 'smote'
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: (resampled_features, resampled_labels)
    """
    logger.info("=" * 80)
    logger.info("HANDLING CLASS IMBALANCE")
    logger.info("=" * 80)
    
    logger.info(f"\nBefore resampling:")
    logger.info(f"  Total samples: {len(y_train):,}")
    logger.info(f"  Class 0: {(y_train == 0).sum():,}")
    logger.info(f"  Class 1: {(y_train == 1).sum():,}")
    
    if method.lower() == 'smote':
        logger.info(f"\nApplying SMOTE (Synthetic Minority Over-sampling Technique)...")
        
        try:
            # Initialize SMOTE
            smote = SMOTE(random_state=random_state)
            
            # Apply SMOTE
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            # Convert back to DataFrame to preserve column names
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_resampled = pd.Series(y_resampled, name=y_train.name)
            
            logger.info(f"\nAfter SMOTE resampling:")
            logger.info(f"  Total samples: {len(y_resampled):,}")
            logger.info(f"  Class 0: {(y_resampled == 0).sum():,}")
            logger.info(f"  Class 1: {(y_resampled == 1).sum():,}")
            logger.info(f"  New balance ratio: 1:1 (perfect balance)")
            
            return X_resampled, y_resampled
        
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Returning original data.")
            return X_train, y_train
    
    else:
        logger.warning(f"Unknown resampling method: {method}. Skipping resampling.")
        return X_train, y_train


# ============================================================================
# STAGE 7: TRAIN-TEST SPLIT
# ============================================================================

def split_train_test_data(
    df: pd.DataFrame,
    target_col: str = 'is_fraud',
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets with stratification.
    
    Stratified split ensures:
    1. Both train and test sets have similar class distributions
    2. Minority class is represented in both sets
    3. Test set is a fair representation of real-world data
    4. Model evaluation metrics are more reliable
    
    Args:
        df (pd.DataFrame): Complete preprocessed dataframe
        target_col (str): Name of target column. Default: 'is_fraud'
        test_size (float): Proportion of data for testing. Default: 0.2
        random_state (int): Random seed for reproducibility. Default: 42
        apply_smote (bool): Whether to apply SMOTE to training data. Default: True
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("=" * 80)
    logger.info("TRAIN-TEST SPLIT WITH STRATIFICATION")
    logger.info("=" * 80)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"\nOriginal dataset size: {len(df):,} samples")
    logger.info(f"Test size: {test_size * 100:.1f}% ({int(len(df) * test_size):,} samples)")
    logger.info(f"Train size: {(1 - test_size) * 100:.1f}% ({int(len(df) * (1 - test_size)):,} samples)")
    
    # Perform stratified train-test split
    logger.info("\nPerforming stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    logger.info(f"✓ Split successful!")
    logger.info(f"  Train: {len(X_train):,} samples")
    logger.info(f"  Test: {len(X_test):,} samples")
    
    # Apply SMOTE to training data only
    if apply_smote:
        logger.info("\nApplying SMOTE to training data only...")
        X_train, y_train = handle_class_imbalance(X_train, y_train)
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# STAGE 8: COMPUTE CLASS WEIGHTS
# ============================================================================

def compute_class_weights(y_train: pd.Series) -> Dict[int, float]:
    """
    Compute class weights for handling imbalance during model training.
    
    Class weights are used to penalize misclassification of minority class more heavily.
    This is useful for:
    1. Tree-based models that support class_weight parameter
    2. Logistic regression and SVM
    3. Neural networks with weighted loss functions
    
    Formula: weight = n_samples / (n_classes * n_samples_per_class)
    
    Args:
        y_train (pd.Series): Training labels
        
    Returns:
        Dict[int, float]: Class weights dictionary {0: weight_0, 1: weight_1}
    """
    logger.info("Computing class weights for imbalanced data...")
    
    # Calculate weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    
    logger.info(f"Class Weights:")
    logger.info(f"  Class 0 (Legitimate): {class_weights[0]:.4f}")
    logger.info(f"  Class 1 (Fraud):      {class_weights[1]:.4f}")
    logger.info(f"  Weight Ratio: {class_weights[1] / class_weights[0]:.2f}x")
    
    return class_weights


# ============================================================================
# STAGE 9: SAVE PREPROCESSING ARTIFACTS
# ============================================================================

def save_preprocessing_artifacts(
    scaler: StandardScaler,
    encoder_dict: Dict,
    output_dir: str = './models'
) -> None:
    """
    Save preprocessing artifacts (scaler and encoders) for production use.
    
    These artifacts must be saved so that:
    1. New data can be preprocessed identically to training data
    2. Test set transformation is consistent
    3. Production inference uses same preprocessing pipeline
    4. Model predictions are reliable and consistent
    
    Args:
        scaler (StandardScaler): Fitted feature scaler
        encoder_dict (Dict): Dictionary of categorical encoders
        output_dir (str): Directory to save artifacts. Default: './models'
    """
    logger.info("=" * 80)
    logger.info("SAVING PREPROCESSING ARTIFACTS")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save scaler
    scaler_path = f'{output_dir}/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Scaler saved to: {scaler_path}")
    
    # Save encoder dictionary
    encoder_path = f'{output_dir}/encoders.pkl'
    joblib.dump(encoder_dict, encoder_path)
    logger.info(f"✓ Encoders saved to: {encoder_path}")
    
    # Save preprocessing metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'scaler_type': 'StandardScaler',
        'n_features_scaler': len(scaler.mean_),
        'n_encoders': len(encoder_dict),
        'encoding_methods': {
            col: info['type'] for col, info in encoder_dict.items()
        }
    }
    
    metadata_path = f'{output_dir}/preprocessing_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    logger.info(f"✓ Metadata saved to: {metadata_path}")
    logger.info(f"\nAll preprocessing artifacts saved successfully!")


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def run_preprocessing_pipeline(
    input_file: str,
    output_dir: str = './data/processed',
    test_size: float = 0.2,
    apply_smote: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute complete preprocessing pipeline from start to finish.
    
    This is the main function that orchestrates all preprocessing steps:
    1. Load data
    2. Explore and validate
    3. Remove ID columns
    4. Extract time features
    5. Encode categorical features
    6. Scale numerical features
    7. Handle class imbalance
    8. Split into train/test sets
    9. Compute class weights
    10. Save artifacts
    11. Save processed datasets
    
    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save processed data. Default: './data/processed'
        test_size (float): Proportion of data for testing. Default: 0.2
        apply_smote (bool): Whether to apply SMOTE. Default: True
        random_state (int): Random seed for reproducibility. Default: 42
        
    Returns:
        Dict[str, Any]: Dictionary containing all processing outputs and metadata
    """
    logger.info("\n" + "=" * 80)
    logger.info("STARTING FRAUD DETECTION PREPROCESSING PIPELINE")
    logger.info("=" * 80 + "\n")
    
    try:
        # ====== STAGE 1: LOAD AND EXPLORE ======
        df = load_data(input_file)
        exploration_info = explore_data(df)
        class_imbalance_before = analyze_class_imbalance(df)
        
        # ====== STAGE 2: DROP ID COLUMNS ======
        df = drop_id_columns(df)
        
        # ====== STAGE 3: EXTRACT TIME FEATURES ======
        df = extract_time_features(df)
        
        # ====== STAGE 4: ENCODE CATEGORICAL FEATURES ======
        df, encoder_dict = encode_categorical_features(df, is_training=True)
        
        # ====== STAGE 5: SCALE NUMERICAL FEATURES ======
        _, numerical_cols = identify_categorical_columns(df)
        # Remove target from numerical columns
        if 'is_fraud' in numerical_cols:
            numerical_cols.remove('is_fraud')
        
        df_for_scaling = df.copy()
        df_for_scaling, scaler = scale_numerical_features(
            df_for_scaling,
            numerical_columns=numerical_cols,
            is_training=True
        )
        df = df_for_scaling
        
        # ====== STAGE 6: TRAIN-TEST SPLIT WITH SMOTE ======
        X_train, X_test, y_train, y_test = split_train_test_data(
            df,
            test_size=test_size,
            apply_smote=apply_smote,
            random_state=random_state
        )
        
        # ====== STAGE 7: COMPUTE CLASS WEIGHTS ======
        class_weights = compute_class_weights(y_train)
        
        # ====== STAGE 8: SAVE ARTIFACTS ======
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save scaler and encoders
        models_dir = './models'
        save_preprocessing_artifacts(scaler, encoder_dict, output_dir=models_dir)
        
        # Save processed datasets
        logger.info("\n" + "=" * 80)
        logger.info("SAVING PROCESSED DATASETS")
        logger.info("=" * 80)
        
        # Combine train data with labels for CSV export
        train_data = X_train.copy()
        train_data['is_fraud'] = y_train.values
        train_file = f'{output_dir}/train.csv'
        train_data.to_csv(train_file, index=False)
        logger.info(f"✓ Training data saved: {train_file} ({len(train_data):,} rows)")
        
        # Save test data
        test_data = X_test.copy()
        test_data['is_fraud'] = y_test.values
        test_file = f'{output_dir}/test.csv'
        test_data.to_csv(test_file, index=False)
        logger.info(f"✓ Test data saved: {test_file} ({len(test_data):,} rows)")
        
        # Save preprocessing statistics
        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        summary_stats = {
            'total_samples': len(df),
            'n_features_original': len(exploration_info['columns']),
            'n_features_final': X_train.shape[1],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_fraud_rate': (y_train == 1).sum() / len(y_train) * 100,
            'test_fraud_rate': (y_test == 1).sum() / len(y_test) * 100,
            'class_imbalance_before': class_imbalance_before,
            'class_weights': class_weights,
        }
        
        logger.info("\nFinal Summary:")
        logger.info(f"  Original features: {summary_stats['n_features_original']}")
        logger.info(f"  Final features: {summary_stats['n_features_final']}")
        logger.info(f"  Training samples: {summary_stats['train_samples']:,}")
        logger.info(f"  Test samples: {summary_stats['test_samples']:,}")
        logger.info(f"  Training fraud rate: {summary_stats['train_fraud_rate']:.2f}%")
        logger.info(f"  Test fraud rate: {summary_stats['test_fraud_rate']:.2f}%")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'encoder_dict': encoder_dict,
            'class_weights': class_weights,
            'summary_stats': summary_stats
        }
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block - runs the complete preprocessing pipeline.
    """
    
    # Configure paths
    input_file = './data/fraud_detection_elite_v3.csv'
    output_dir = './data/processed'
    models_dir = './models'
    
    # Run pipeline
    results = run_preprocessing_pipeline(
        input_file=input_file,
        output_dir=output_dir,
        test_size=0.2,
        apply_smote=True,
        random_state=42
    )
    
    # Access results
    print("\n✅ Pipeline complete! Access the following:")
    print(f"   X_train shape: {results['X_train'].shape}")
    print(f"   X_test shape: {results['X_test'].shape}")
    print(f"   y_train shape: {results['y_train'].shape}")
    print(f"   y_test shape: {results['y_test'].shape}")
