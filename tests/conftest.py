import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.iav_flap_anomaly_detection import make_data

@pytest.fixture(scope="session")
def dataset():
    # Generate dataset (using smaller sizes for testing)
    X_train, X_test, test_ground_truth = make_data(n_train=500, n_test=500)
    
    # Split X_test into final test set (80%) and validation set (20%) for final evaluation
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, test_ground_truth,
        test_size=0.2,
        random_state=42
    )
    
    # Split X_train into tuning subsets for hyperparameter tuning only
    X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
        X_train, np.ones(len(X_train)),
        test_size=0.2,
        random_state=42  # Validation from train only
    )
    
    # Normalize the data (fit only on the tuning training set)
    scaler = StandardScaler()
    X_train_tune_scaled = scaler.fit_transform(X_train_tune)
    X_val_tune_scaled   = scaler.transform(X_val_tune)
    X_train_scaled      = scaler.transform(X_train)  # Full training set for final training
    X_val_scaled        = scaler.transform(X_val)      # Evaluation set from test split
    X_test_scaled       = scaler.transform(X_test)     # Final test set

    # Combine tuning data for hyperparameter tuning
    X_all = np.vstack([X_train_tune_scaled, X_val_tune_scaled])
    y_all = np.hstack([np.ones(len(X_train_tune_scaled)), y_val_tune])
    
    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "X_val_scaled": X_val_scaled,
        "X_train_tune_scaled": X_train_tune_scaled,
        "X_val_tune_scaled": X_val_tune_scaled,
        "X_all": X_all,
        "y_all": y_all
    }
