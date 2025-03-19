import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from models.iav_flap_anomaly_detection import make_data
from sklearn.preprocessing import StandardScaler

@pytest.fixture(scope="session")
def dataset():
    X_train, X_test, y_test = make_data(n_train=500, n_test=500)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_test
