import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
class IsolationForestModel:
    """
    A simple anomaly detection model using IsolationForest.
    
    Attributes:
    -----------
    n_estimators : int
        The number of base estimators in the ensemble.
    max_samples : int, float or 'auto'
        The number of samples to draw from X to train each base estimator.
    contamination : 'auto' or float
        The proportion of outliers in the data set.
    max_features : int or float
        The number of features to draw from X to train each base estimator.
    random_state : int or None
        Controls the random seed for reproducibility.
    model : IsolationForest
        The underlying scikit-learn IsolationForest model.
    is_fitted : bool
        Indicates whether the model has been fit.
    """

    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto',
                 max_features=1.0, random_state=None):
        """
        Initialize the IsolationForest model with user-defined or default parameters.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        
        # Create the IsolationForest instance
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state
        )
        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Fit the IsolationForest model on training data (which should be mostly normal).
        """
        self.model.fit(X)
        self.is_fitted = True

    def predict(self, X):
        """
        Predict if each sample in X is an outlier or not.
        
        Returns:
            y_pred : array of shape (n_samples,)
                     +1 for inliers, -1 for outliers.
        """
        if not self.is_fitted:
            raise RuntimeError("You must fit the model before calling predict().")
        return self.model.predict(X)

    def score_samples(self, X):
        """
        Return the anomaly score for each sample in X.
        The lower, the more abnormal. Negative scores represent outliers.
        
        Returns:
            scores : array of shape (n_samples,)
                     The anomaly score of the input samples.
        """
        if not self.is_fitted:
            raise RuntimeError("You must fit the model before calling score_samples().")
        return self.model.score_samples(X)

    def evaluate(self, X, y_true):
        """
        Evaluate the model on a labeled dataset.
        
        :param X: array-like, shape (n_samples, n_features)
                  The data to evaluate.
        :param y_true: array-like, shape (n_samples,)
                       Ground truth labels (1 for normal, -1 for anomaly).
                       If 0/1, they will be remapped to 1/-1.
        :return: A dict with accuracy, precision, recall, f1_score
        """
        if not self.is_fitted:
            raise RuntimeError("You must fit the model before calling evaluate().")
        
        # Convert 0/1 labels to 1/-1 if needed
        unique_labels = set(np.unique(y_true))
        if unique_labels == {0, 1}:
            # Map 0 -> 1 (inlier) and 1 -> -1 (outlier)
            y_true = np.where(y_true == 0, 1, -1)
        
        y_pred = self.predict(X)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=-1)
        rec = recall_score(y_true, y_pred, pos_label=-1)
        f1 = f1_score(y_true, y_pred, pos_label=-1)
        
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }

    # Optional: Provide get_params and set_params to allow GridSearchCV or RandomizedSearchCV
    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "max_features": self.max_features,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Reinitialize the underlying IsolationForest with updated params
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state
        )
        return self
    
    def plot_decision_boundary(self, X_train, X_test=None, X_outliers=None):
        """
        Plot the decision boundary learned by the Isolation Forest model,
        along with data points.
        
        Parameters:
            X_train: Normal training data (2D).
            X_test: Optional test data (2D).
            X_outliers: Optional set of data points predicted as anomalies (2D).
            
        This method is for 2D data only. It plots:
        - Training data (blue)
        - Test data (orange)
        - Anomalies (green)
        and shows the decision function contour from the IsolationForest model.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        if X_train.shape[1] != 2:
            raise ValueError("Visualization is supported only for 2D data.")
        
        # Combine all points to determine the plot range
        X_all = X_train
        if X_test is not None:
            X_all = np.vstack([X_all, X_test])
        if X_outliers is not None:
            X_all = np.vstack([X_all, X_outliers])
        x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
        y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
        
        # Create a grid to evaluate the decision function.
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                            np.linspace(y_min, y_max, 300))
        # Compute the decision function: lower values indicate more abnormal.
        Z = self.model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(6, 5))
        # Shade the outlier region (where decision function < 0)
        plt.contourf(xx, yy, Z, levels=[Z.min(), 0], colors='orange', alpha=0.3)
        # Plot the decision boundary (where decision function equals 0)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
        
        # Plot training data
        plt.scatter(X_train[:, 0], X_train[:, 1],
                    color='blue', alpha=0.7, label='Training data (ok)')
        # Plot test data if available
        if X_test is not None:
            plt.scatter(X_test[:, 0], X_test[:, 1],
                        color='orange', alpha=0.7, label='Test data (ok)')
        # Plot anomalies if provided
        if X_outliers is not None:
            plt.scatter(X_outliers[:, 0], X_outliers[:, 1],
                        color='green', alpha=0.7, label='Anomalies')
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc='upper left')
        plt.title("Isolation Forest Decision Boundary")
        plt.show()


