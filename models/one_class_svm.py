from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class OneClassSVMModel:
    def __init__(self, kernel='rbf', nu=0.5, gamma='scale'):
        """
        Initialize One-Class SVM model for anomaly detection.
        Parameters:
        - kernel: Kernel type for OneClassSVM (default 'rbf').
        - nu: An upper bound on the fraction of anomalies (float between 0 and 1).
        - gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' (float, 'scale', or 'auto').
        """
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
    
    def fit(self, X, y=None):
        """
        Train the One-Class SVM on normal data.
        :param X: Array-like of shape (n_samples, n_features) containing only normal (inlier) data.
        """
        self.model.fit(X)
        return self  # return self to allow method chaining if desired
    
    def predict(self, X):
        """
        Predict whether each sample in X is an inlier or outlier.
        :param X: Array-like, shape (n_samples, n_features).
        :return: Array of predictions (1 for normal/inlier, -1 for anomaly/outlier).
        """
        return self.model.predict(X)
    
    def tune_parameters(self, X_train, X_val_norm, X_val_anom, nu_values, gamma_values, metric='f1'):
        """
        Perform grid search to tune nu and gamma using a validation set.
        Trains on X_train (normal data) for each combination of parameters and evaluates on validation data.
        
        :param X_train: Normal data for training (array-like).
        :param X_val_norm: Normal data for validation (array-like).
        :param X_val_anom: Anomalous data for validation (array-like).
        :param nu_values: List of candidate values for nu.
        :param gamma_values: List of candidate values for gamma.
        :param metric: Metric to optimize: 'accuracy', 'precision', 'recall', or 'f1' (default 'f1').
        :return: (best_nu, best_gamma) tuple of the best parameters found.
        """
        if X_val_norm is None or X_val_anom is None:
            raise ValueError("Both normal and anomalous validation sets are required for tuning.")
        
        # Prepare combined validation data and true labels (1 for normal, -1 for anomaly)
        import numpy as np
        X_val_combined = np.vstack([X_val_norm, X_val_anom])
        y_val_combined = np.hstack([np.ones(len(X_val_norm)), -np.ones(len(X_val_anom))])
        
        best_score = -1.0
        best_params = (None, None)
        for nu in nu_values:
            for gamma in gamma_values:
                # Train a model with the current parameters
                candidate_model = OneClassSVM(kernel=self.kernel, nu=nu, gamma=gamma)
                candidate_model.fit(X_train)
                # Predict on validation data
                y_pred = candidate_model.predict(X_val_combined)
                # Compute the chosen metric (treat anomalies as the "positive" class, i.e., pos_label=-1)
                if metric == 'accuracy':
                    score = accuracy_score(y_val_combined, y_pred)
                elif metric == 'precision':
                    score = precision_score(y_val_combined, y_pred, pos_label=-1)
                elif metric == 'recall':
                    score = recall_score(y_val_combined, y_pred, pos_label=-1)
                else:  # 'f1' or any other case defaults to F1-score
                    score = f1_score(y_val_combined, y_pred, pos_label=-1)
                # Update best params if current combination is better
                if score > best_score:
                    best_score = score
                    best_params = (nu, gamma)
        
        # Retrain the model on the full training set with best found parameters
        best_nu, best_gamma = best_params
        self.nu, self.gamma = best_nu, best_gamma
        self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.model.fit(X_train)
        return best_params
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on a test set with true labels.
        :param X_test: Array-like test dataset (can contain normal and anomaly samples).
        :param y_test: True labels for X_test (1 for normal, -1 for anomaly). 
                       If labels are given as 0/1, 0 will be treated as normal and 1 as anomaly.
        :return: Dictionary with accuracy, precision, recall, and F1-score.
        """
        import numpy as np
        y_true = np.array(y_test)
        # Convert 0/1 labels to 1/-1 if needed (assume 1 indicates anomaly in that case)
        if set(np.unique(y_true)) == {0, 1}:
            y_true = np.where(y_true == 0, 1, -1)
        # Predict using the trained model
        y_pred = self.model.predict(X_test)
        # Compute evaluation metrics (treat -1 as positive class for precision/recall/F1)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=-1)
        rec = recall_score(y_true, y_pred, pos_label=-1)
        f1 = f1_score(y_true, y_pred, pos_label=-1)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
    
    def plot_decision_boundary(self, X_train, X_test=None, X_outliers=None):
        """
        Plot the decision boundary learned by the One-Class SVM, along with data points.
        - X_train: Normal training data (2 features).
        - X_test: Optional normal test data to plot.
        - X_outliers: Optional anomalous data to plot.
        
        This method only works for 2D feature space. It will show the model's decision frontier (boundary) 
        and highlight which points are considered anomalies.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        if X_train.shape[1] != 2:
            raise ValueError("Visualization is supported only for 2D data.")
        
        # Combine all points to determine plot range
        X_all = X_train
        if X_test is not None:
            X_all = np.vstack([X_all, X_test])
        if X_outliers is not None:
            X_all = np.vstack([X_all, X_outliers])
        x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
        y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
        
        # Compute the decision function grid values for contour plotting
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                             np.linspace(y_min, y_max, 300))
        Z = self.model.decision_function(np.c_[xx.ravel(), yy.ravel()])  # anomaly score (negative for outliers)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary at anomaly score = 0
        plt.figure(figsize=(6, 5))
        # Shade the outlier region (where decision function < 0)
        plt.contourf(xx, yy, Z, levels=[Z.min(), 0], colors = 'white', alpha=0.3)
        # Plot the decision boundary line
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
        
        # Plot training, test, and outlier points
        plt.scatter(X_train[:, 0], X_train[:, 1], color = 'blue', alpha = 0.7)
        if X_test is not None:
            plt.scatter(X_test[:, 0], X_test[:, 1], color = 'blue', label='Train + Test Ok data', alpha = 0.7)
        if X_outliers is not None:
            plt.scatter(X_outliers[:, 0], X_outliers[:, 1], color = 'orange', label='Anomalies/ Not ok data')
        plt.legend(loc='upper left')
        plt.title("One-Class SVM Decision Boundary")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show() 

# Create a wrapper class that extends your custom model and adds get_params/set_params
class OneClassSVMWrapper(OneClassSVMModel):
    def get_params(self, deep=True):
        return {"kernel": self.kernel, "nu": self.nu, "gamma": self.gamma}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        return self
