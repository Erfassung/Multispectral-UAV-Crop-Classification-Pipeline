
# Example usage for inference with the saved PCA Random Forest model:

import joblib
import json
import numpy as np

# Load the trained model and components
model = joblib.load("rf_pca_model.joblib")
pca = joblib.load("pca_transformer.joblib")
scaler = joblib.load("scaler.joblib")
imputer = joblib.load("imputer.joblib")

# Load metadata
with open("model_metadata.json", "r") as f:
    metadata = json.load(f)

print(f"Model info: {metadata['pca_components']} PCA components, "
      f"Test Accuracy: {metadata['test_accuracy']:.3f}")

# For new data prediction:
# 1. Apply imputation: X_new = imputer.transform(X_new)
# 2. Apply scaling: X_new = scaler.transform(X_new)
# 3. Apply PCA: X_new = pca.transform(X_new)
# 4. Predict: predictions = model.predict(X_new)
# 5. Get probabilities: probabilities = model.predict_proba(X_new)

def predict_crops(X_new):
    """Complete inference pipeline for new data"""
    X_processed = imputer.transform(X_new)
    X_scaled = scaler.transform(X_processed)
    X_pca = pca.transform(X_scaled)
    predictions = model.predict(X_pca)
    probabilities = model.predict_proba(X_pca)
    return predictions, probabilities
