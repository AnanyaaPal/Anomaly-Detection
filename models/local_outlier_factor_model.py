import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LocalOutlierFactorModel:
    """
    A LocalOutlierFactor-based anomaly detection model.
    """

    def __init__(self, n_neighbors=20, contamination='auto', novelty=False):
        """
        Initialize the LocalOutlierFactor model.
        
        Parameters:
        - n_neighbors: int, number of neighbors (default 20)
        - contamination: 'auto' or float in (0, 0.5], proportion of outliers
        - novelty: bool, if True, use LOF in novelty detection mode
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.novelty = novelty
        # The LOF model is created, but note that if novelty=False, we typically use fit_predict
        # to get outlier labels directly on training data.
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=self.novelty
        )
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """
        Fit the LOF model on training data.
        
        If novelty=False, you typically won't call predict on new data with LOF. 
        If novelty=True, you can do so, but results differ from standard LOF.
        """
        if self.novelty:
            # In novelty mode, we call .fit() so we can later use .predict() on unseen data
            self.model.fit(X)
        else:
            # In outlier detection mode, we can do fit_predict on the training set itself
            # This automatically labels the training data as inlier/outlier.
            self.model.fit_predict(X)
        self.is_fitted = True

    def predict(self, X):
        """
        Predict outliers vs. inliers on new data if novelty=True, or on training data if novelty=False.
        
        Returns: 
          Array of +1 (inlier) or -1 (outlier).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        if self.novelty:
            # Only valid when novelty=True
            return self.model.predict(X)
        else:
            # In outlier detection mode, LOF doesn't allow .predict on new data
            # but we can do fit_predict on the same data.
            return self.model.fit_predict(X)

    def score_samples(self, X):
        """
        Return the LOF score for each sample (the higher, the more normal).
        Only available if novelty=True in scikit-learn >= 0.20.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring samples.")
        
        if not self.novelty:
            raise RuntimeError("score_samples is only available when novelty=True.")
        
        return self.model.score_samples(X)

    def evaluate(self, X, y_true):
        """
        Evaluate the model on a labeled dataset.
        
        :param X: feature array
        :param y_true: ground truth labels (1 for normal, -1 for anomaly).
        :return: dict of metrics: accuracy, precision, recall, f1_score
        """
        y_pred = self.predict(X)
        # Ensure y_true is in {1, -1}
        unique_labels = set(np.unique(y_true))
        if unique_labels == {0, 1}:
            # Map 0->1 and 1->-1 if needed
            y_true = np.where(y_true == 0, 1, -1)

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

    def plot_lof_predictions(self, X_train, X_test=None, X_outliers=None, title="LocalOutlierFactor Results"):
        """
        Plot LOF results for 2D data. This function will scatter-plot:
        - X_train as 'Training data (ok)'
        - X_test as 'Test data (ok)'
        - X_outliers as 'Not ok data'
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, 2)
            Normal training data used to fit the LOF model.
        X_test : array-like of shape (n_samples, 2), optional
            Test data (also normal, if you have a separate normal set).
        X_outliers : array-like of shape (n_samples, 2), optional
            Known or labeled outliers (anomalies) to plot separately.
        title : str
            Title of the plot.
        """
        import matplotlib.pyplot as plt
        
        # Check if data is 2D
        if X_train.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")
        
        plt.figure(figsize=(6, 5))
        
        # Plot training data
        plt.scatter(X_train[:, 0], X_train[:, 1],
                    color='blue', alpha=0.7)
        
        # Plot normal test data
        if X_test is not None:
            plt.scatter(X_test[:, 0], X_test[:, 1],
                        color='blue', alpha=0.7, label='Train + Test Ok data')
        
        # Plot outliers/anomalies
        if X_outliers is not None:
            plt.scatter(X_outliers[:, 0], X_outliers[:, 1],
                        color='orange', alpha=0.7, label='Not ok data')
        
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc='upper left')
        plt.show()
    
    def get_params(self, deep=True):
        """
        Return the parameters as a dict, so that the estimator can be cloned.
        """
        return {"n_neighbors": self.n_neighbors,
                "contamination": self.contamination,
                "novelty": self.novelty}

    def set_params(self, **params):
        """
        Set parameters from a dict. Reinitialize the underlying LOF model accordingly.
        """
        for key, value in params.items():
            setattr(self, key, value)
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=self.novelty
        )
        return self
