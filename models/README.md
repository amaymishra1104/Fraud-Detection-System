# Models Directory

This directory contains trained machine learning models and associated artifacts.

## Files

### Trained Models
- `fraud_model.pkl` - Serialized XGBoost classifier for fraud detection
- `fraud_model_v2.pkl` - Alternative model version (if needed)

### Preprocessing
- `scaler.pkl` - Feature scaler (StandardScaler or MinMaxScaler)
- `feature_names.pkl` - List of feature names used during training
- `feature_importance.pkl` - Feature importance scores

### Documentation
- `model_metadata.json` - Model training metadata and hyperparameters
- `model_performance.txt` - Training/validation metrics

## Model Naming Convention

- `fraud_model_v{version}_{date}.pkl` - Versioned models with training date
- Example: `fraud_model_v1_2024-05-17.pkl`

## Serialization

Models are serialized using joblib:
```python
import joblib

# Save model
joblib.dump(model, 'fraud_model.pkl')

# Load model
model = joblib.load('fraud_model.pkl')
```

## Model Information

### Current Model (fraud_model.pkl)
- **Algorithm**: XGBoost Classifier
- **Training Date**: [Update after training]
- **Accuracy**: [Update after training]
- **Precision**: [Update after training]
- **Recall**: [Update after training]
- **ROC-AUC**: [Update after training]

## Version Control

When training new models:
1. Test the new model thoroughly
2. Document performance metrics
3. Archive old models (keep last 2-3 versions)
4. Update deployment to use new model
