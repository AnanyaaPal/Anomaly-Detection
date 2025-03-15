import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
