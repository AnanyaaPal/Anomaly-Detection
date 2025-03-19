import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from sklearn.metrics import accuracy_score
from models.isolation_forest_model import IsolationForestModel

@pytest.fixture
def isolation_forest_model():
    # Using hyperparameters from our corrected approach
    return IsolationForestModel(
        n_estimators=300,
        max_samples=0.8,
        contamination="auto",
        max_features=1.0,
        random_state=42
    )

def test_isolation_forest_fit_predict(dataset, isolation_forest_model):
    # Use the dictionary returned by our dataset fixture
    X_train = dataset["X_train_scaled"]
    X_test = dataset["X_test_scaled"]
    y_test = dataset["y_test"]
    
    isolation_forest_model.fit(X_train)
    preds = isolation_forest_model.predict(X_test)
    
    assert preds.shape == y_test.shape, "Prediction shape mismatch"
    assert set(np.unique(preds)).issubset({-1, 1}), "Predictions should only contain -1 and 1"

def test_isolation_forest_evaluate(dataset, isolation_forest_model):
    X_train = dataset["X_train_scaled"]
    X_test = dataset["X_test_scaled"]
    y_test = dataset["y_test"]
    
    isolation_forest_model.fit(X_train)
    metrics = isolation_forest_model.evaluate(X_test, y_test)
    
    # Ensure key metrics exist and are within the [0, 1] range
    assert "accuracy" in metrics and "f1_score" in metrics, "Missing key metrics"
    for key, value in metrics.items():
        assert 0 <= value <= 1, f"{key} out of valid range"
