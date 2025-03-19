import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from models.local_outlier_factor_model import LocalOutlierFactorModel

@pytest.fixture
def lof_model():
    # Use corrected hyperparameters (for example, those chosen via grid search)
    return LocalOutlierFactorModel(n_neighbors=100, contamination='auto', novelty=True)

def test_lof_fit_predict(dataset, lof_model):
    # Use the dataset fixture that returns a dictionary
    X_train = dataset["X_train_scaled"]
    X_test = dataset["X_test_scaled"]
    y_test = dataset["y_test"]
    
    lof_model.fit(X_train)
    preds = lof_model.predict(X_test)
    
    # Optionally, if your implementation sets an attribute for fitted status
    if hasattr(lof_model, "is_fitted"):
        assert lof_model.is_fitted, "Model should be marked as fitted"
    
    assert preds.shape == y_test.shape, "Prediction shape mismatch"
    assert set(np.unique(preds)).issubset({-1, 1}), "Predictions should be either -1 or 1"

def test_lof_evaluate(dataset, lof_model):
    X_train = dataset["X_train_scaled"]
    X_test = dataset["X_test_scaled"]
    y_test = dataset["y_test"]
    
    lof_model.fit(X_train)
    eval_results = lof_model.evaluate(X_test, y_test)
    
    # Ensure key metrics exist and are within [0, 1]
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        value = eval_results.get(metric)
        assert value is not None, f"{metric} calculation failed"
        assert 0 <= value <= 1, f"{metric} is outside valid range"
