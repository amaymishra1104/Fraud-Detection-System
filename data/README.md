# Data Directory

This directory contains all datasets used in the fraud detection system.

## Structure

### raw/
Original, unprocessed data files:
- `transactions.csv` - Raw transaction data with all columns
- `fraud_labels.csv` - Fraud labels for supervised learning

### processed/
Cleaned and feature-engineered datasets:
- `train.csv` - Training dataset (typically 70% of data)
- `test.csv` - Testing dataset (typically 15% of data)
- `validation.csv` - Validation dataset (typically 15% of data)

## Data Preparation Workflow

```
raw/ → EDA → Feature Engineering → processed/
```

1. **Load raw data** from `raw/` directory
2. **Exploratory Data Analysis** (EDA notebooks)
3. **Data Cleaning** (handle missing values, outliers)
4. **Feature Engineering** (create new features, scale)
5. **Train-Test Split** (stratified split for imbalanced data)
6. **Save processed data** to `processed/` directory

## Dataset Format

All CSV files should include:
- Header row with column names
- Consistent data types
- Proper handling of missing values
- No duplicate rows

### Expected Columns
```
transaction_id, amount, merchant, timestamp, location, 
card_type, customer_age, transaction_type, fraud_label
```

## Data Privacy

⚠️ **Important**: 
- Remove or anonymize PII (Personally Identifiable Information)
- Never commit real customer data to version control
- Use `.gitignore` to exclude large data files

## File Size Guidelines

- `raw/` - Original data (can be large)
- `processed/` - Optimized for training (compressed when possible)

## Loading Data

```python
import pandas as pd

# Load training data
train_data = pd.read_csv('data/processed/train.csv')

# Load test data
test_data = pd.read_csv('data/processed/test.csv')
```

## Data Documentation

For each dataset, maintain:
1. **Data Dictionary** - Column descriptions
2. **Missing Value Report** - Percentage of missing data
3. **Statistical Summary** - Mean, median, std dev
4. **Class Distribution** - Fraud vs legitimate ratio
