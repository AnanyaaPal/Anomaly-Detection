import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from models.one_class_svm import OneClassSVMModel

def test_oneclass_shapes():
    # Mock a small set of normal data
    X_train = np.random.randn(50, 2)  # shape (50, 2)
    # Mock a small test set with labels
    X_test = np.random.randn(20, 2)
    y_test = np.array([1]*15 + [-1]*5)  # e.g., 15 normal, 5 anomalies

    model = OneClassSVMModel(kernel='rbf', nu=0.1, gamma='scale')
    model.fit(X_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == (20,), "Prediction shape mismatch"
    # You can further assert any property you expect
