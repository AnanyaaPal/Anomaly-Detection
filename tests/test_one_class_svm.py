import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from models.one_class_svm import OneClassSVMWrapper

@pytest.fixture
def one_class_svm_model():
    # Instantiate the One-Class SVM wrapper with initial hyperparameters.
    return OneClassSVMWrapper(kernel='rbf', nu=0.1, gamma='scale')

def test_one_class_svm_fit_predict(dataset, one_class_svm_model):
    # Retrieve the training and test splits from the dataset dictionary
    X_train = dataset["X_train_scaled"]
    X_test = dataset["X_test_scaled"]
    y_test = dataset["y_test"]

    # Fit the model on the training data
    one_class_svm_model.fit(X_train)
    preds = one_class_svm_model.predict(X_test)
    
    # Check that the prediction shape matches y_test's shape
    assert preds.shape == y_test.shape, "Prediction shape mismatch."
    # Verify that predictions only contain the valid labels (-1 and 1)
    assert set(np.unique(preds)).issubset({-1, 1}), "Predictions should be either -1 or 1"

def test_one_class_svm_evaluate(dataset, one_class_svm_model):
    X_train = dataset["X_train_scaled"]
    X_test = dataset["X_test_scaled"]
    y_test = dataset["y_test"]

    one_class_svm_model.fit(X_train)
    metrics = one_class_svm_model.evaluate(X_test, y_test)
    
    # Ensure that key metrics exist
    required_keys = ["accuracy", "precision", "recall", "f1_score"]
    assert all(key in metrics for key in required_keys), "Metrics keys missing."
    
    # Check that each metric is within the valid range [0, 1]
    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{key} out of valid range"
