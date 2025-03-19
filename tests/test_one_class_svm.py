import pytest
import numpy as np
from models.one_class_svm import OneClassSVMModel

@pytest.fixture
def one_class_svm_model():
    return OneClassSVMModel(kernel='rbf', nu=0.1, gamma='scale')

def test_one_class_svm_fit_predict(dataset, one_class_svm_model):
    X_train, X_test, y_test = dataset
    one_class_svm_model.fit(X_train)
    preds = one_class_svm_model.predict(X_test)
    
    assert preds.shape == y_test.shape, "Prediction shape mismatch."
    assert set(pred for pred in np.unique(one_class_svm_model.predict(X_test))).issubset({-1, 1}), "Predictions should be 1 or -1"

def test_one_class_svm_evaluate(dataset, one_class_svm_model):
    X_train, X_test, y_test = dataset
    one_class_svm_model.fit(X_train)
    metrics = one_class_svm_model.evaluate(X_test, y_test)
    assert all(metric in metrics for metric in ["accuracy", "precision", "recall", "f1_score"]), "Metrics keys missing."
    for metric, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{metric} out of valid range"
