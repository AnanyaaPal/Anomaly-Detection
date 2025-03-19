import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from models.isolation_forest_model import IsolationForestModel

@pytest.fixture
def isolation_forest_model():
    return IsolationForestModel(n_estimators=100, contamination=0.1, random_state=42)

def test_isolation_forest_fit_predict(dataset, isolation_forest_model):
    X_train, X_test, y_test = dataset
    isolation_forest_model = isolation_forest_model()
    isolation_forest_model.fit(X_train)
    preds = isolation_forest_model.predict(X_test)

    assert preds.shape == y_test.shape, "Prediction shape mismatch"
    assert set(preds).issubset({-1, 1}), "Predictions should only contain -1 and 1"

def test_isolation_forest_evaluate(dataset, isolation_forest_model):
    X_train, X_test, y_test = dataset
    isolation_forest_model.fit(X_train)
    metrics = isolation_forest_model.evaluate(X_test, y_test)

    assert all(0 <= v <= 1 for v in metrics.values()), "Metrics out of valid range (0-1)"
    assert "accuracy" in metrics and "f1_score" in metrics, "Missing key metrics"
