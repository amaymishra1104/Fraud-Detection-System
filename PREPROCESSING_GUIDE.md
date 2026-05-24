"""
Preprocessing Pipeline Usage Guide

This notebook demonstrates how to use the fraud detection preprocessing pipeline
in different scenarios: exploratory analysis, model training, and production inference.

Contents:
    1. Installation & Setup
    2. Basic Usage - Running the Pipeline
    3. Advanced Usage - Custom Configurations
    4. Production Usage - Real-time Predictions
    5. Troubleshooting
"""

# ============================================================================
# SECTION 1: INSTALLATION & SETUP
# ============================================================================

"""
Before running the preprocessing pipeline, ensure all dependencies are installed.

Install required packages:
    pip install pandas numpy scikit-learn imbalanced-learn joblib

Verify installation:
    python -c "import pandas, numpy, sklearn, imblearn; print('All imports successful!')"
"""


# ============================================================================
# SECTION 2: BASIC USAGE - RUNNING THE PIPELINE
# ============================================================================

"""
EXAMPLE 1: Running the complete preprocessing pipeline

This is the simplest way to preprocess your data from start to finish.
"""

# Option A: Using command line
"""
cd fraud_detection
python preprocessing_pipeline.py
"""

# Option B: Using Python script
"""
from preprocessing_pipeline import run_preprocessing_pipeline

# Run complete pipeline with default parameters
results = run_preprocessing_pipeline(
    input_file='./data/fraud_detection_elite_v3.csv',
    output_dir='./data/processed',
    test_size=0.2,
    apply_smote=True,
    random_state=42
)

# Access the preprocessed data
X_train = results['X_train']          # Training features
X_test = results['X_test']            # Test features
y_train = results['y_train']          # Training labels
y_test = results['y_test']            # Test labels
scaler = results['scaler']            # Fitted scaler for new data
encoder_dict = results['encoder_dict'] # Encoders for new data
class_weights = results['class_weights'] # Class weights for model training

# Train your model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight=class_weights)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score:.4f}")
"""


# ============================================================================
# SECTION 3: ADVANCED USAGE - CUSTOM CONFIGURATIONS
# ============================================================================

"""
EXAMPLE 2: Using individual preprocessing functions

For more control, use individual functions instead of the full pipeline.
"""

"""
import pandas as pd
from preprocessing_pipeline import (
    load_data,
    explore_data,
    drop_id_columns,
    extract_time_features,
    encode_categorical_features,
    scale_numerical_features,
    split_train_test_data,
    compute_class_weights
)

# Step 1: Load and explore
df = load_data('./data/fraud_detection_elite_v3.csv')
explore_data(df)

# Step 2: Remove IDs
df = drop_id_columns(df)

# Step 3: Extract time features
df = extract_time_features(df, timestamp_col='timestamp')

# Step 4: Encode categoricals
df, encoder_dict = encode_categorical_features(df, is_training=True)

# Step 5: Scale numerical features
df, scaler = scale_numerical_features(df, is_training=True)

# Step 6: Split data
X_train, X_test, y_train, y_test = split_train_test_data(
    df,
    target_col='is_fraud',
    test_size=0.2,
    apply_smote=True,
    random_state=42
)

# Step 7: Compute weights
class_weights = compute_class_weights(y_train)
"""


# ============================================================================
# SECTION 4: PRODUCTION USAGE - REAL-TIME PREDICTIONS
# ============================================================================

"""
EXAMPLE 3: Using ProductionPreprocessor for real-time predictions

In production, use the saved artifacts to preprocess new data consistently.
"""

"""
from preprocessing_utils import ProductionPreprocessor
import pandas as pd
import joblib

# Initialize preprocessor (loads saved artifacts)
preprocessor = ProductionPreprocessor(models_dir='./models')

# CASE A: Single transaction prediction (real-time)
single_transaction = {
    'transaction_id': 'TXN_12345',
    'user_id': 'U001',
    'amount': 250.50,
    'transaction_type': 'transfer',
    'merchant_category': 'electronics',
    'timestamp': '2026-05-17 14:30:00',
    'transaction_frequency': 5,
    'avg_user_amount': 200.0,
    'deviation_from_avg': 50.50,
    'transaction_gap_seconds': 3600,
    'account_age_days': 365,
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

# Convert to DataFrame and preprocess
df_new = pd.DataFrame([single_transaction])
df_preprocessed = preprocessor.transform(df_new)

# Load model and make prediction
model = joblib.load('./models/fraud_model.pkl')
prediction = model.predict(df_preprocessed)
probability = model.predict_proba(df_preprocessed)[0][1]

print(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Legitimate'}")
print(f"Fraud Probability: {probability:.4f}")
"""

"""
# CASE B: Batch processing (multiple transactions)
batch_transactions = pd.read_csv('./data/new_transactions.csv')
df_preprocessed = preprocessor.transform(batch_transactions)

model = joblib.load('./models/fraud_model.pkl')
predictions = model.predict(df_preprocessed)
probabilities = model.predict_proba(df_preprocessed)[:, 1]

batch_transactions['prediction'] = predictions
batch_transactions['fraud_probability'] = probabilities
batch_transactions.to_csv('./results/predictions.csv', index=False)
"""


# ============================================================================
# SECTION 5: INTEGRATION WITH FASTAPI BACKEND
# ============================================================================

"""
EXAMPLE 4: Using preprocessor in FastAPI backend

This shows how to integrate preprocessing in the prediction API.
"""

"""
# In backend/models/__init__.py
from preprocessing_utils import ProductionPreprocessor
import joblib

class FraudDetectionModel:
    def __init__(self):
        self.preprocessor = ProductionPreprocessor(models_dir='./models')
        self.model = joblib.load('./models/fraud_model.pkl')
    
    def predict(self, transaction_dict):
        # Preprocess transaction
        import pandas as pd
        df = pd.DataFrame([transaction_dict])
        df_processed = self.preprocessor.transform(df)
        
        # Make prediction
        prediction = self.model.predict(df_processed)[0]
        probability = self.model.predict_proba(df_processed)[0][1]
        
        return {
            'prediction': 'Fraud' if prediction == 1 else 'Legitimate',
            'fraud_probability': float(probability),
            'risk_score': float(probability)
        }

# In backend/routes/__init__.py
from fastapi import APIRouter
from models import fraud_model

@router.post('/predict')
async def predict_fraud(transaction: TransactionRequest):
    result = fraud_model.predict(transaction.dict())
    return result
"""


# ============================================================================
# SECTION 6: TROUBLESHOOTING COMMON ISSUES
# ============================================================================

"""
TROUBLESHOOTING GUIDE

Issue 1: "FileNotFoundError: No such file or directory: './data/fraud_detection_elite_v3.csv'"
Solution:
    - Ensure you're in the correct directory: pwd
    - Check file exists: ls ./data/fraud_detection_elite_v3.csv
    - Update path in code if needed

Issue 2: "ModuleNotFoundError: No module named 'imblearn'"
Solution:
    pip install imbalanced-learn

Issue 3: SMOTE failing with "n_samples <= n_neighbors"
Solution:
    - Your minority class has too few samples
    - Set apply_smote=False in preprocessing function
    - Or use class weights instead

Issue 4: Scaler not found in production
Solution:
    - Ensure preprocessing pipeline was run and artifacts saved
    - Check ./models/ directory contains scaler.pkl and encoders.pkl
    - Verify ProductionPreprocessor(models_dir='./models') path

Issue 5: Predictions inconsistent between training and production
Solution:
    - Use ProductionPreprocessor with saved artifacts
    - Don't create new scaler or encoders
    - Apply transformations in same order as training

Issue 6: Memory error with large datasets
Solution:
    - Process in chunks: for chunk in pd.read_csv(..., chunksize=10000)
    - Use dtype optimization: df.astype({col: 'float32'})
    - Reduce test_size to use less memory

Issue 7: Class weights producing NaN values
Solution:
    - Ensure y_train has both classes present
    - Check for missing values in target column
    - Verify no infinite values in features
"""


# ============================================================================
# SECTION 7: PREPROCESSING PIPELINE FLOWCHART
# ============================================================================

"""
PREPROCESSING PIPELINE FLOWCHART

┌─────────────────────────────────────────────────────────┐
│ 1. LOAD DATA                                             │
│    └─ Read CSV file                                      │
│    └─ Check shape and columns                            │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│ 2. EXPLORE & VALIDATE                                    │
│    └─ Missing values analysis                            │
│    └─ Class imbalance analysis                           │
│    └─ Data types verification                            │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│ 3. DROP ID COLUMNS                                       │
│    └─ Remove: transaction_id, user_id                   │
│    └─ These have no predictive power                     │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│ 4. EXTRACT TIME FEATURES                                 │
│    └─ hour, day_of_week, month, day_of_month            │
│    └─ is_weekend, is_night_time                          │
│    └─ quarter, week_of_year                              │
│    └─ Drop original timestamp                            │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│ 5. ENCODE CATEGORICAL FEATURES                           │
│    └─ One-hot encode (cardinality ≤ 10)                 │
│    └─ Label encode (cardinality > 10)                    │
│    └─ Save encoder dictionary                            │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│ 6. SCALE NUMERICAL FEATURES                              │
│    └─ StandardScaler: (X - mean) / std                   │
│    └─ Save fitted scaler                                 │
│    └─ Mean = 0, Std = 1                                  │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│ 7. STRATIFIED TRAIN-TEST SPLIT                           │
│    └─ 80% train, 20% test                                │
│    └─ Stratified by target variable                      │
│    └─ Preserves class distribution                       │
└────────────┬────────────────────────────────────────────┘
             │
         TRAIN SET
             │
┌────────────▼────────────────────────────────────────────┐
│ 8. HANDLE CLASS IMBALANCE (SMOTE)                        │
│    └─ Only apply to training data                        │
│    └─ Create synthetic minority samples                  │
│    └─ Balance to 50:50 ratio                             │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│ 9. COMPUTE CLASS WEIGHTS                                 │
│    └─ For imbalanced data                                │
│    └─ Weight minority class higher                       │
│    └─ Use in model training                              │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────┐
│ 10. SAVE ARTIFACTS & DATASETS                            │
│     └─ scaler.pkl (StandardScaler)                       │
│     └─ encoders.pkl (Encoder dictionary)                 │
│     └─ train.csv (Preprocessed training data)            │
│     └─ test.csv (Preprocessed test data)                 │
└────────────┬────────────────────────────────────────────┘
             │
         READY FOR ML
             │
┌────────────▼────────────────────────────────────────────┐
│ Model Training & Evaluation                              │
│ └─ Train on X_train with y_train                         │
│ └─ Evaluate on X_test with y_test                        │
│ └─ Use class_weights for imbalanced data                 │
└─────────────────────────────────────────────────────────┘
"""


# ============================================================================
# SECTION 8: QUICK REFERENCE COMMANDS
# ============================================================================

"""
QUICK REFERENCE

Run Full Pipeline:
    python preprocessing_pipeline.py

Check Preprocessing Log:
    cat preprocessing.log

Load Preprocessed Data:
    import pandas as pd
    train = pd.read_csv('./data/processed/train.csv')
    test = pd.read_csv('./data/processed/test.csv')

Load Preprocessing Artifacts:
    import joblib
    scaler = joblib.load('./models/scaler.pkl')
    encoders = joblib.load('./models/encoders.pkl')

Verify Preprocessing:
    python -c "
    import pandas as pd
    train = pd.read_csv('./data/processed/train.csv')
    print(f'Training shape: {train.shape}')
    print(f'Fraud rate: {train[\"is_fraud\"].mean()*100:.2f}%')
    print(f'Features: {train.shape[1]-1}')
    "
"""

print(__doc__)
