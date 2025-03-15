import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

def make_data(n_train=10000, n_test=10000):
    """
    Create the dataset.
    
    Parameters
    ----------
    
    n_train: int
        The number of training data points. Default: 10000
    n_test: int
        The number of test data points. Default: 10000
    
    Returns
    -------
    X_train: array_like
        Training data.
    X_test: array_like
        Test data.
    test_ground_truth_labels: array_like
        Ground truth labels for the test data.
    """
    n_normal = n_train + n_test//2
    n_anomaly = n_train + n_test - n_normal
    
    rng = np.random.default_rng(42)
    
    X, y = make_moons(n_samples=(n_normal, n_anomaly), noise=0.05, random_state=23, shuffle=False)
    y_normalized = np.where(y==0, 1, -1) # Adapt labels to OneClassSVM labels # 1 is normal inlier data # -1 is anomalous outlier data

    X_normal = rng.permutation(X[:n_normal, :], axis=0), y_normalized[:n_normal]
    X_anomaly, y_anomaly = X[n_normal:, :], y_normalized[n_normal:]

    X_normal, y_normal = rng.permutation(X[:n_normal, :], axis=0), y_normalized[:n_normal]
    X_anomaly, y_anomaly = X[n_normal:, :], y_normalized[n_normal:]
    
    X_train = X_normal[:n_train, :]

    X_test = np.vstack([X_normal[n_train:, :], X_anomaly])
    y_test = np.hstack([y_normalized[n_train:], y_anomaly])

    shuffled_indices = np.arange(X_test.shape[0], dtype=np.uint64)
    rng.shuffle(shuffled_indices)
    X_test = X_test[shuffled_indices, :]
    test_ground_truth_labels = y_test[shuffled_indices]
    
    return X_train, X_test, test_ground_truth_labels

def plot_data(X_train, X_test, test_ground_truth):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(12.8,4.8))
    ax1.scatter(X_train[:, 0], X_train[:, 1], label="Training data")
    ax1.set_xlim(-1.2, 2.2)
    ax1.set_ylim(-0.7, 1.2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Training data")
    ax1.legend()
    ax2.scatter(X_train[:, 0], X_train[:, 1], label="Training data (ok)", alpha=0.7)
    ax2.scatter(X_test[:, 0], X_test[:, 1], label="Test data (ok + not ok)", alpha=0.7)
    ax2.set_xlim(-1.2, 2.2)
    ax2.set_ylim(-0.7, 1.2)
    ax2.set_title("Training vs. test data")
    ax2.legend()
    
    inlier_mask = test_ground_truth == 1
    outlier_mask = np.logical_not(inlier_mask)
    
    ax3.scatter(X_test[inlier_mask, 0], X_test[inlier_mask, 1], label="Ok data", alpha=0.7)
    ax3.scatter(X_test[outlier_mask, 0], X_test[outlier_mask, 1], label="Not ok data", alpha=0.7)
    ax3.set_xlim(-1.2, 2.2)
    ax3.set_ylim(-0.7, 1.2)
    ax3.set_title("What detection should ideally look like")
    ax3.legend()
    plt.show()