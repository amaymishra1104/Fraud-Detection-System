# Preprocessing Pipeline - Quick Start

## 📋 Files Created

| File | Purpose |
|------|---------|
| **preprocessing_pipeline.py** | Main preprocessing pipeline with all stages |
| **preprocessing_utils.py** | Production utilities for real-time predictions |
| **PREPROCESSING_GUIDE.md** | Usage examples and troubleshooting |
| **PREPROCESSING_DOCUMENTATION.md** | Detailed explanation of all transformations |

## 🚀 Quick Start (30 seconds)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing
```bash
python preprocessing_pipeline.py
```

### 3. Check Results
```bash
ls -la data/processed/
ls -la models/
```

## 📊 Pipeline Stages

The preprocessing pipeline performs **9 major stages**:

```
Raw Data
   ↓
1. Load & Explore      (Data validation, statistics)
2. Drop ID Columns     (transaction_id, user_id)
3. Time Features       (Extract hour, day, month, etc.)
4. Categorical Encode  (One-hot + label encoding)
5. Scale Numericals    (StandardScaler: mean=0, std=1)
6. Train-Test Split    (80:20 stratified split)
7. SMOTE Balancing     (Upsample minority class)
8. Compute Weights     (For imbalanced training)
9. Save Artifacts      (Scaler, encoders, data)
   ↓
Preprocessed Data Ready for ML
```

## 📁 Input Data Structure

**Expected columns in `fraud_detection_elite_v3.csv`:**

### ID Columns (will be removed)
- `transaction_id` - Unique transaction ID
- `user_id` - Unique user ID

### Numerical Features
- `amount` - Transaction amount ($)
- `transaction_frequency` - How often user transacts
- `avg_user_amount` - User's average transaction
- `deviation_from_avg` - Deviation from user average
- `transaction_gap_seconds` - Time since last transaction
- `account_age_days` - Account age
- `failed_attempts` - Failed transaction attempts

### Categorical Features
- `transaction_type` - transfer, payment, withdrawal
- `merchant_category` - electronics, fashion, travel, gaming, grocery
- `device_type` - mobile, web
- `location` - USA, UK, Germany, India, UAE

### Binary Features
- `is_foreign_transaction` - 0/1
- `unusual_amount_flag` - 0/1
- `velocity_flag` - 0/1
- `new_device_flag` - 0/1
- `location_change_flag` - 0/1
- `night_transaction_flag` - 0/1

### Timestamp
- `timestamp` - Date and time of transaction

### Target
- `is_fraud` - 0=Legitimate, 1=Fraud

## 📤 Output Data Structure

After preprocessing, you get:

### Saved Files

**./data/processed/**
- `train.csv` - Training data (80% of samples)
- `test.csv` - Test data (20% of samples)

**./models/**
- `scaler.pkl` - StandardScaler for feature normalization
- `encoders.pkl` - Categorical encoders
- `preprocessing_metadata.pkl` - Metadata

### Feature Count

| Type | Count | Examples |
|------|-------|----------|
| Numerical (scaled) | 13 | amount, hour, account_age_days |
| Binary flags | 7 | is_foreign_transaction, is_weekend |
| One-hot encoded | 15 | transaction_type_transfer, location_USA |
| **Total Features** | **36** | (+ 1 target column) |

## 🔧 Usage Examples

### Basic Usage

```python
from preprocessing_pipeline import run_preprocessing_pipeline

# Run complete pipeline
results = run_preprocessing_pipeline(
    input_file='./data/fraud_detection_elite_v3.csv',
    output_dir='./data/processed',
    test_size=0.2,
    apply_smote=True,
    random_state=42
)

# Access results
X_train = results['X_train']
X_test = results['X_test']
y_train = results['y_train']
y_test = results['y_test']
scaler = results['scaler']
class_weights = results['class_weights']
```

### Training a Model

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train with class weights to handle imbalance
model = RandomForestClassifier(
    n_estimators=100,
    class_weight=results['class_weights'],
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")
```

### Production Inference

```python
from preprocessing_utils import ProductionPreprocessor
import pandas as pd
import joblib

# Load preprocessor with saved artifacts
preprocessor = ProductionPreprocessor(models_dir='./models')

# Load model
model = joblib.load('./models/fraud_model.pkl')

# Single transaction
new_transaction = pd.DataFrame([{
    'transaction_id': 'TXN_12345',
    'user_id': 'U001',
    'amount': 500.0,
    # ... other columns
}])

# Preprocess
df_processed = preprocessor.transform(new_transaction)

# Predict
prediction = model.predict(df_processed)
probability = model.predict_proba(df_processed)[0][1]

print(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Legitimate'}")
print(f"Fraud Probability: {probability:.4f}")
```

## 🎯 Key Transformations

### 1. Categorical Encoding

**One-Hot Encoding (Low Cardinality ≤ 10)**
```
transaction_type:
  transfer → [1, 0, 0]
  payment → [0, 1, 0]
  withdrawal → [0, 0, 1]
```

**Label Encoding (High Cardinality > 10)**
```
location:
  USA → 0
  UK → 1
  Germany → 2
  India → 3
  UAE → 4
```

### 2. Time Feature Extraction

```
timestamp: 2026-05-03 16:47:17
  ↓
hour: 16
day_of_week: 4 (Friday)
month: 5 (May)
is_weekend: 0 (False)
is_night_time: 0 (False)
...
```

### 3. Numerical Scaling

```
Amount: 2500.0
  ↓
StandardScaler: (2500 - mean) / std
  ↓
Scaled Amount: 0.25 (normalized between -3 to 3)
```

### 4. SMOTE Balancing (Training Only)

```
Before:
  Legitimate: 7,840 (98%)
  Fraud: 160 (2%)
  Ratio: 49:1

After SMOTE:
  Legitimate: 7,840 (50%)
  Fraud: 7,840 (50%) ← Synthetic fraud samples added
  Ratio: 1:1
```

## 📊 Expected Statistics

| Metric | Value |
|--------|-------|
| Input samples | ~10,000 |
| Input features | 21 |
| Output features | 36 |
| Features removed | 2 (IDs) |
| Features added | 8 (time) + 15 (one-hot) = 23 |
| Train samples | ~8,000 |
| Train samples (after SMOTE) | ~16,000 |
| Test samples | ~2,000 |
| Train fraud rate (original) | ~2% |
| Train fraud rate (after SMOTE) | ~50% |
| Test fraud rate (unchanged) | ~2% |

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| File not found | Ensure CSV is in `./data/` directory |
| Import error (imblearn) | `pip install imbalanced-learn` |
| Memory error | Reduce data size or process in chunks |
| Scaler not found | Run `preprocessing_pipeline.py` first |
| Inconsistent predictions | Use ProductionPreprocessor with saved artifacts |

## 📝 Detailed Documentation

- **PREPROCESSING_GUIDE.md** - Step-by-step usage guide with code examples
- **PREPROCESSING_DOCUMENTATION.md** - Detailed explanation of all transformations
- **preprocessing_pipeline.py** - Heavily commented source code

## ✅ Verification Checklist

After running preprocessing:

- [ ] `./data/processed/train.csv` exists
- [ ] `./data/processed/test.csv` exists
- [ ] `./models/scaler.pkl` exists
- [ ] `./models/encoders.pkl` exists
- [ ] `./preprocessing.log` created with details
- [ ] Train data has no missing values
- [ ] Test data has no missing values
- [ ] Features properly scaled (mean ≈ 0, std ≈ 1)
- [ ] Train data balanced (~50% fraud after SMOTE)
- [ ] Test data representative (~2% fraud, real-world ratio)

## 🚢 Production Deployment

1. **Save preprocessing artifacts**
   ```
   ./models/
   ├── scaler.pkl
   ├── encoders.pkl
   └── preprocessing_metadata.pkl
   ```

2. **Use in production**
   ```python
   from preprocessing_utils import ProductionPreprocessor
   preprocessor = ProductionPreprocessor()
   ```

3. **Ensure consistency**
   - Always use saved artifacts
   - Never create new scaler/encoders
   - Apply transformations in same order

## 📞 Support

For detailed help, refer to:
1. Source code comments in `preprocessing_pipeline.py`
2. `PREPROCESSING_GUIDE.md` for usage examples
3. `PREPROCESSING_DOCUMENTATION.md` for technical details
4. `preprocessing.log` for execution logs

---


