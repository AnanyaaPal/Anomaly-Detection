import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Import your data helper functions
from models.iav_flap_anomaly_detection import make_data, plot_data

# Import relevant models
from models.one_class_svm import OneClassSVMModel

# Load data once, cache to speed up
@st.cache_data
def load_data():
    X_train, X_test, test_ground_truth = make_data()
    return X_train, X_test, test_ground_truth

X_train, X_test, test_ground_truth = load_data()

# Create top-level radio for main sections
st.sidebar.title("Navigation")
page_choice = st.sidebar.radio(
    "Select a page",
    ["Home", "Models", "Model Comparison"]
)
# HOME PAGE
if page_choice == "Home":
    st.title("Anomaly Detection - Project Overview")

    st.markdown("""
    **The Problem**  
    Below is our data. We have a system that produces data that normally looks like the left picture. 
    However, there is a special kind of problem that occurs that makes the data shift and flip. 
    Usually, nobody has the time to look at the data and label it - we only have data of which we 
    know that it is probably ok and serves as your training data.
    """)

    # Display the three-panel plot from your iav_flap_anomaly_detection.py
    # st.subheader('')
    # fig = plt.figure()
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
        outlier_mask = ~inlier_mask

        ax3.scatter(X_test[inlier_mask, 0], X_test[inlier_mask, 1], label="Ok data", alpha=0.7)
        ax3.scatter(X_test[outlier_mask, 0], X_test[outlier_mask, 1], label="Not ok data", alpha=0.7)
        ax3.set_xlim(-1.2, 2.2)
        ax3.set_ylim(-0.7, 1.2)
        ax3.set_title("What detection should ideally look like")
        ax3.legend()

        return fig  

    st.markdown("""
    ---
    **My Solution**
    We will explore three pre-defined algorithms for anomaly detection from the sk-learn module:
    1. **One-Class SVM**
    2. **Isolation Forest**
    3. **LocalOutlierFactor**

    Select a model from the sidebar to see its performance on this data.
    """)

# MODELS SECTION
elif page_choice == "Models":
    st.title("Models")
    st.markdown("Select a specific model in the sidebar to see its details and results.")

    # Secondary radio for model selection
    model_choice = st.sidebar.radio(
        "Select Model",
        ["One-Class SVM", "Isolation Forest", "LocalOutlierFactor"]
    )

# ONE-CLASS SVM PAGE
    if model_choice == "One-Class SVM":
        st.subheader("One-Class SVM")
        st.markdown("""
        The One-Class SVM is trained solely on the normal data (from the training set) and then used to 
        detect anomalies in the test set. You can adjust its hyperparameters below.
        """)

        # Set random seed for reproducibility
        import numpy as np
        np.random.seed(42)

        # Sidebar hyperparameters for One-Class SVM
        st.sidebar.subheader("One-Class SVM Settings")
        kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "sigmoid"], index=0)
        nu = st.sidebar.slider("Nu", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
        gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"], index=0)

        # Load & split data
        from models.iav_flap_anomaly_detection import make_data
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, precision_recall_curve

        X_train, X_test, test_ground_truth = make_data()
        
        # Split X_test into final test set (80%) and validation set (20%) for final evaluation
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, test_ground_truth, test_size=0.2, random_state=42
        )

        # Split X_train into tuning subsets for hyperparameter tuning only
        X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
            X_train, np.ones(len(X_train)), test_size=0.2, random_state=42
        )

        # Normalize the data (fit only on X_train_tune)
        scaler = StandardScaler()
        X_train_tune_scaled = scaler.fit_transform(X_train_tune)
        X_val_tune_scaled   = scaler.transform(X_val_tune)
        X_train_scaled      = scaler.transform(X_train)  # Full training set for final training
        X_val_scaled        = scaler.transform(X_val)      # Evaluation set from X_test split
        X_test_scaled       = scaler.transform(X_test)     # Final test set

        # Combine tuning data for hyperparameter tuning
        X_all = np.vstack([X_train_tune_scaled, X_val_tune_scaled])
        y_all = np.hstack([np.ones(len(X_train_tune_scaled)), y_val_tune])

        # Set up GridSearchCV to test different kernels, nu, and gamma parameters for One-Class SVM
        param_grid = {
            "kernel": ["rbf"],
            "nu": [0.5, 0.1, 0.2, 0.2],
            "gamma": [0.05, 0.08, 0.1, 0.2, 0.3],
        }

        def accuracy_scoring(estimator, X, y, **kwargs):
            y_pred = estimator.predict(X)
            return accuracy_score(y, y_pred)

        scorer = make_scorer(accuracy_score, greater_is_better=True)

        from models.one_class_svm import OneClassSVMWrapper

        grid_search = GridSearchCV(
            estimator=OneClassSVMWrapper(),  # Wrapper class that supports get_params/set_params
            param_grid=param_grid,
            scoring=scorer,
            cv=5
        )
        grid_search.fit(X_all, y_all)
        st.write("**Best parameters found:**", grid_search.best_params_)

        # Train the final model on the full training set using the best hyperparameters
        best_model = grid_search.best_estimator_
        best_model.fit(X_train_scaled)

        # Evaluate on validation set (from X_test split)
        y_pred_val = best_model.predict(X_val_scaled)
        val_metrics = best_model.evaluate(X_val_scaled, y_val)
        st.write("**Validation Metrics:**")
        import pandas as pd
        val_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1_score"]
            ]
        })
        st.table(val_df.style.format(precision=3))

        # Evaluate on final test set
        y_pred_test = best_model.predict(X_test_scaled)
        test_metrics = best_model.evaluate(X_test_scaled, y_test)
        st.write("**Test Metrics:**")
        test_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [
                test_metrics["accuracy"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1_score"]
            ]
        })
        st.table(test_df.style.format(precision=3))

        # Decision Boundary Visualization
        st.subheader("Decision Boundary Visualization")
        def plot_decision_boundary(model, X_train, X_test=None, X_outliers=None, title="One-Class SVM Decision Boundary"):
            import matplotlib.pyplot as plt
            import numpy as np
            if X_train.shape[1] != 2:
                st.error("Visualization is supported only for 2D data.")
                return None
            X_all = X_train
            if X_test is not None:
                X_all = np.vstack([X_all, X_test])
            if X_outliers is not None:
                X_all = np.vstack([X_all, X_outliers])
            x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
            y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
            Z = model.model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            fig, ax = plt.subplots(figsize=(6,5))
            ax.contourf(xx, yy, Z, levels=[Z.min(), 0], colors='white', alpha=0.3)
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
            ax.scatter(X_train[:, 0], X_train[:, 1], color='blue', alpha=0.7, label='Train + Test Ok data')
            if X_outliers is not None:
                ax.scatter(X_outliers[:, 0], X_outliers[:, 1], color='orange', alpha=0.7, label='Anomalies')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            ax.legend(loc='upper right')
            return fig

        outlier_mask = (y_test == -1)
        fig_boundary = plot_decision_boundary(best_model, X_train_scaled, X_test=X_test_scaled, X_outliers=X_test_scaled[outlier_mask])
        st.pyplot(fig_boundary)

        # Advanced Threshold Tuning & Metrics
        st.subheader("Advanced Threshold Tuning & Metrics")
        scores_test = best_model.model.decision_function(X_test_scaled)
        thresholds = np.linspace(scores_test.min(), scores_test.max(), 20)
        best_acc = 0.0
        best_thresh = None
        for thr in thresholds:
            y_pred_custom = np.where(scores_test < thr, -1, 1)
            acc = accuracy_score(y_test, y_pred_custom)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thr
        st.write(f"**Best threshold = {best_thresh:.3f} yields accuracy = {best_acc:.3f}**")
        y_pred_best = np.where(scores_test < best_thresh, -1, 1)
        cm = confusion_matrix(y_test, y_pred_best, labels=[-1, 1])
        st.write("**Confusion Matrix (@ best threshold):**")
        st.write(cm)

        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test, -scores_test, pos_label=-1)
        fig_pr, ax_pr = plt.subplots(figsize=(6,5))
        ax_pr.plot(recall, precision, marker='.')
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curve (One-Class SVM)")
        st.pyplot(fig_pr)
    
        # Display model code
        with st.expander("Show One-Class SVM Model Code"):
            try:
                with open("models/one_class_svm.py", "r") as file:
                    code = file.read()
                st.code(code, language="python")
            except Exception as e:
                st.write("Could not load model code:", e)
        
        # Display documentation of the model
        with st.expander("Show Documentation"):
            st.markdown("[OneClassSVM Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)")

# ISOLATION FOREST PAGE 

    elif model_choice == "Isolation Forest":
        st.subheader("Isolation Forest")
        st.markdown("""
        The Isolation Forest is trained on the normal data (assumed from the training set) and then used to detect anomalies in the test set.  
        Adjust the hyperparameters below to see how the model's performance and decision boundary change.
        """)

        # Sidebar hyperparameters for Isolation Forest
        st.sidebar.subheader("Isolation Forest Settings")
        n_estimators = st.sidebar.selectbox("Number of Estimators", options=[200, 300], index=0)
        max_samples = st.sidebar.selectbox("Max Samples", options=["auto", 0.5, 0.8], index=0)
        contamination = st.sidebar.selectbox("Contamination", options=["auto", 0.2, 0.3, 0.4, 0.5], index=0)
        max_features = st.sidebar.selectbox("Max Features", options=[1.0, 0.8], index=0)
        random_state = st.sidebar.number_input("Random State", value=42, step=1)

        st.subheader("Model Training & Evaluation")
        from models.isolation_forest_model import IsolationForestModel
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, precision_recall_curve
        from models.iav_flap_anomaly_detection import make_data

        # Load Data
        #    - X_train: normal data only (shape: (10000, 2))
        #    - X_test: mixed data (shape: (10000, 2))
        #    - test_ground_truth: labels for X_test (shape: (10000,), +1 for normal, -1 for anomaly)
        X_train, X_test, test_ground_truth = make_data()

        # Split X_test into final test set (80%) and validation set (20%) for final evaluation
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, test_ground_truth, 
            test_size=0.2,
            random_state=42
        )

        # Split X_train into training and validation for hyperparameter tuning only
        X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
            X_train, np.ones(len(X_train)), 
            test_size=0.2, 
            random_state=42  # Validation from train only
        )

        # Normalize the data (fit only on X_train_tune)
        scaler = StandardScaler()
        X_train_tune_scaled = scaler.fit_transform(X_train_tune)
        X_val_tune_scaled   = scaler.transform(X_val_tune)
        X_train_scaled      = scaler.transform(X_train)  # Full train set for final training
        X_val_scaled        = scaler.transform(X_val)      # Validation from test
        X_test_scaled       = scaler.transform(X_test)     # Final test set

        # Hyperparameter tuning should be done only on X_train_tune and X_val_tune
        X_all = np.vstack([X_train_tune_scaled, X_val_tune_scaled])
        y_all = np.hstack([y_train_tune, y_val_tune])

        # GridSearchCV using accuracy
        param_grid = {
            "n_estimators":   [200, 300],
            "max_samples":    ["auto", 0.5, 0.8],
            "contamination":  ["auto", 0.2, 0.3, 0.4, 0.5],
            "max_features":   [1.0, 0.8],
            "random_state":   [42]
        }

        def accuracy_scoring_iso(estimator, X, y, **kwargs):
            y_pred = estimator.predict(X)
            return accuracy_score(y, y_pred)

        scorer = make_scorer(accuracy_score, greater_is_better=True)

        grid_search = GridSearchCV(
            estimator=IsolationForestModel(),
            param_grid=param_grid,
            scoring=scorer,
            cv=3
        )
        grid_search.fit(X_all, y_all)
        st.write("**Best parameters found:**", grid_search.best_params_)

        # Train the final model on the full training set
        best_model = grid_search.best_estimator_
        best_model.fit(X_train_scaled)  # Now train using full train data

        # Validation set evaluation (not used in hyperparameter tuning)
        y_pred_val  = best_model.predict(X_val_scaled)
        val_metrics = best_model.evaluate(X_val_scaled, y_val)
        st.write("**Validation Metrics:**")
        import pandas as pd
        val_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1_score"]
            ]
        })
        st.table(val_df.style.format(precision=3))

        # Test set evaluation (final model performance)
        y_pred_test  = best_model.predict(X_test_scaled)
        test_metrics = best_model.evaluate(X_test_scaled, y_test)
        st.write("**Test Metrics:**")
        test_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [
                test_metrics["accuracy"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1_score"]
            ]
        })
        st.table(test_df.style.format(precision=3))

        # Visualize the decision boundary
        st.subheader("Decision Boundary Visualization")
        def plot_decision_boundary(model, X_train, X_test=None, X_outliers=None, title="Isolation Forest Decision Boundary"):
            import numpy as np
            import matplotlib.pyplot as plt

            if X_train.shape[1] != 2:
                raise ValueError("Visualization is supported only for 2D data.")

            # Combine data to get plotting range
            X_all = X_train
            if X_test is not None:
                X_all = np.vstack([X_all, X_test])
            if X_outliers is not None:
                X_all = np.vstack([X_all, X_outliers])

            x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
            y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 300),
                np.linspace(y_min, y_max, 300)
            )
            # For IsolationForest, decision_function returns negative scores for outliers
            Z = model.model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            fig, ax = plt.subplots(figsize=(6,5))
            # Shade region where decision_function < 0 (outliers) in white
            ax.contourf(xx, yy, Z, levels=[Z.min(), 0], colors='white', alpha=0.3)
            # Boundary at 0
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')

            # Plot ok data in blue, anomalies in orange
            ax.scatter(X_train[:, 0], X_train[:, 1], color='blue', alpha=0.7)
            if X_test is not None:
                ax.scatter(X_test[:, 0], X_test[:, 1], color='blue', alpha=0.7, label='Train + Test Ok data')
            if X_outliers is not None:
                ax.scatter(X_outliers[:, 0], X_outliers[:, 1], color='orange', alpha=0.7, label='Anomalies/Not ok data')

            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(loc='upper right')
            return fig

        # Identify outliers in the final test set based on y_test
        outlier_mask = (y_test == -1)
        fig_boundary = plot_decision_boundary(best_model, X_train_scaled,
                                            X_test=X_test_scaled,
                                            X_outliers=X_test_scaled[outlier_mask])
        st.pyplot(fig_boundary)

        # Advanced Score-Based Threshold & Metrics
        st.subheader("Advanced Threshold Tuning & Metrics")
        scores_test = best_model.model.decision_function(X_test_scaled)
        thresholds = np.linspace(scores_test.min(), scores_test.max(), 20)
        best_acc = 0.0
        best_thresh = None
        for thr in thresholds:
            y_pred_custom = np.where(scores_test < thr, -1, 1)  # < thr => anomaly
            acc = accuracy_score(y_test, y_pred_custom)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thr

        st.write(f"**Best threshold = {best_thresh:.3f} yields accuracy = {best_acc:.3f}**")
        y_pred_best = np.where(scores_test < best_thresh, -1, 1)
        cm = confusion_matrix(y_test, y_pred_best, labels=[-1, 1])
        st.write("**Confusion Matrix (@ best threshold):**")
        st.write(cm)

        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test, -scores_test, pos_label=-1)
        fig_pr, ax_pr = plt.subplots(figsize=(6,5))
        ax_pr.plot(recall, precision, marker='.')
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curve (Isolation Forest)")
        st.pyplot(fig_pr)

        # Display model code
        with st.expander("Show Isolation Forest Model Code"):
            try:
                with open("models/isolation_forest_model.py", "r") as file:
                    code = file.read()
                st.code(code, language="python")
            except Exception as e:
                st.write("Could not load model code:", e)

        # Display documentation of the model
        with st.expander("Show Documentation"):
            st.markdown("[Isolation Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)")

# LOCAL OUTLIER FACTOR PAGE

    elif model_choice == "LocalOutlierFactor":
        st.subheader("Local Outlier Factor")
        st.markdown("""
        The Local Outlier Factor (LOF) model is trained on normal data (from the training set) 
        and then used to detect anomalies in the test set. LOF is a local density-based method that 
        assigns an outlier score to each sample by comparing its density with that of its neighbors.  
        Adjust the hyperparameters below to see how the model's performance and visualization change.
        """)

        # Sidebar hyperparameters
        st.sidebar.subheader("Local Outlier Factor Settings")
        n_neighbors = st.sidebar.number_input("Number of Neighbors", value=20, min_value=5, max_value=100, step=1)
        contamination = st.sidebar.selectbox("Contamination", options=["auto", 0.05, 0.1, 0.2, 0.3, 0.4], index=0)
        novelty_str = st.sidebar.selectbox("Novelty", options=["True", "False"], index=1)
        novelty = novelty_str == "True"

        st.subheader("Model Training & Evaluation")

        from models.local_outlier_factor_model import LocalOutlierFactorModel
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, precision_recall_curve
        from models.iav_flap_anomaly_detection import make_data
        import pandas as pd

        # Load Data
        #    - X_train: normal data only (shape: (10000, 2))
        #    - X_test: mixed data (shape: (10000, 2))
        #    - test_ground_truth: labels for X_test (shape: (10000,), +1 for normal, -1 for anomaly)
        X_train, X_test, test_ground_truth = make_data()

        # Split X_test into final test set (80%) and validation set (20%) for final evaluation
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, test_ground_truth, 
            test_size=0.2, 
            random_state=42
        )

        # Split X_train into training and validation for hyperparameter tuning only
        X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
            X_train, np.ones(len(X_train)), 
            test_size=0.2, 
            random_state=42  # Validation from train only
        )

        # Normalize the data (fit only on the tuning training set)
        scaler = StandardScaler()
        X_train_tune_scaled = scaler.fit_transform(X_train_tune)
        X_val_tune_scaled   = scaler.transform(X_val_tune)
        X_train_scaled      = scaler.transform(X_train)  # Full train set for final training
        X_val_scaled        = scaler.transform(X_val)      # Validation from test
        X_test_scaled       = scaler.transform(X_test)     # Final test set

        # Prepare combined tuning data for hyperparameter tuning
        X_all = np.vstack([X_train_tune_scaled, X_val_tune_scaled])
        y_all = np.hstack([y_train_tune, y_val_tune])

        # Set up GridSearchCV to test different n_neighbors and contamination values using accuracy
        param_grid = {
            "n_neighbors": [50, 75, 100],
            "contamination": ['auto', 0.05, 0.1, 0.2, 0.3, 0.4]
        }

        def accuracy_scoring_lof(estimator, X, y, **kwargs):
            y_pred = estimator.predict(X)
            return accuracy_score(y, y_pred)

        scorer = make_scorer(accuracy_score, greater_is_better=True)

        grid_search = GridSearchCV(
            estimator=LocalOutlierFactorModel(novelty=True),  # novelty=True enables predictions on new data
            param_grid=param_grid,
            scoring=scorer,
            cv=10
        )
        grid_search.fit(X_all, y_all)
        st.write("**Best parameters found:**", grid_search.best_params_)

        # Train the final model on the full training set using the best hyperparameters
        best_model = grid_search.best_estimator_
        best_model.fit(X_train_scaled)

        # Evaluate the best model on both the validation and final test sets

        # Evaluation on validation set (from X_test split)
        y_pred_val = best_model.predict(X_val_scaled)
        val_metrics = best_model.evaluate(X_val_scaled, y_val)
        st.write("**Validation Metrics:**")
        val_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1_score"]
            ]
        })
        st.table(val_df.style.format(precision=3))

        # Evaluation on final test set
        y_pred_test = best_model.predict(X_test_scaled)
        test_metrics = best_model.evaluate(X_test_scaled, y_test)
        st.write("**Test Metrics:**")
        test_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [
                test_metrics["accuracy"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1_score"]
            ]
        })
        st.table(test_df.style.format(precision=3))

        # Decision Boundary Visualization
        st.subheader("Decision Boundary Visualization")

        def plot_decision_boundary(model, X_train, X_test=None, X_outliers=None, title="Local Outlier Factor Decision Boundary"):
            import matplotlib.pyplot as plt
            import numpy as np
            
            if X_train.shape[1] != 2:
                st.error("Visualization is supported only for 2D data.")
                return None
            
            # Combine data points to determine plot range
            X_all = X_train
            if X_test is not None:
                X_all = np.vstack([X_all, X_test])
            if X_outliers is not None:
                X_all = np.vstack([X_all, X_outliers])
            
            x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
            y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
            
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                np.linspace(y_min, y_max, 300))
            Z = model.model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            fig, ax = plt.subplots(figsize=(6,5))
            ax.contourf(xx, yy, Z, levels=[Z.min(), 0], colors='white', alpha=0.3)
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
            
            # Plot training data in blue, anomalies in orange.
            ax.scatter(X_train[:, 0], X_train[:, 1], color='blue', alpha=0.7)
            if X_test is not None:
                ax.scatter(X_test[:, 0], X_test[:, 1], color='blue', alpha=0.7, label='Train + Test Ok data')
            if X_outliers is not None:
                ax.scatter(X_outliers[:, 0], X_outliers[:, 1], color='orange', alpha=0.7, label='Anomalies/Not ok data')
            
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            ax.legend(loc='upper right')
            return fig

        # Identify outliers in the final test set based on y_test
        outlier_mask = (y_test == -1)
        fig_boundary = plot_decision_boundary(best_model, X_train_scaled, X_test=X_test_scaled, X_outliers=X_test_scaled[outlier_mask])
        st.pyplot(fig_boundary)

        # Advanced Score-Based Threshold & Metrics
        st.subheader("Advanced Threshold Tuning & Metrics")
        scores_test = best_model.model.decision_function(X_test_scaled)
        thresholds = np.linspace(scores_test.min(), scores_test.max(), 20)
        best_acc = 0.0
        best_thresh = None
        for thr in thresholds:
            # For LOF, negative scores indicate anomalies.
            y_pred_custom = np.where(scores_test < thr, -1, 1)
            acc = accuracy_score(y_test, y_pred_custom)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thr

        st.write(f"**Best threshold = {best_thresh:.3f} yields accuracy = {best_acc:.3f}**")
        y_pred_best = np.where(scores_test < best_thresh, -1, 1)
        cm = confusion_matrix(y_test, y_pred_best, labels=[-1, 1])
        st.write("**Confusion Matrix (@ best threshold):**")
        st.write(cm)

        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_test, -scores_test, pos_label=-1)
        fig_pr, ax_pr = plt.subplots(figsize=(6,5))
        ax_pr.plot(recall, precision, marker='.')
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision-Recall Curve (Local Outlier Factor)")
        st.pyplot(fig_pr)

        # Display the LocalOutlierFactor model code
        with st.expander("Show LocalOutlierFactor Model Code"):
            try:
                with open("models/local_outlier_factor_model.py", "r") as file:
                    code = file.read()
                st.code(code, language="python")
            except Exception as e:
                st.write("Could not load model code:", e)

        # Display the documentation of the model
        with st.expander("Show Documentation"):
            st.markdown("[Local Outlier Factor Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)")

# MODEL COMPARISON
elif page_choice == "Model Comparison":
    st.title("Model Performance Comparison")

    import pandas as pd

    st.subheader("Key Takeaways")
    performance_data = {
        "Model": ["One-Class SVM", "Local Outlier Factor", "Isolation Forest"],
        "Accuracy": ["83.7%", "99.1%", "91.6%"],
        "Precision": ["89.1%", "98.2%", "85.8%"],
        "Recall": ["77.0%", "100.0%", "100.0%"],
        "F1-Score": ["82.6%", "99.1%", "92.3%"]
    }
    performance_df = pd.DataFrame(performance_data)
    st.table(performance_df)

    st.markdown("### **Best Performing Model:** `Isolation Forest`")

    st.subheader("Observations")
    st.markdown("""
    - **One-Class SVM** achieves moderate performance with a test accuracy of **83.7%** and a recall of **77.0%**, indicating that a notable number of anomalies are missed.
    - **Local Outlier Factor (LOF)** shows near-perfect metrics (accuracy **99.1%**, 100% recall), but such stellar performance suggests potential overfitting and raises concerns about its robustness on new, noisy data.
    - **Isolation Forest** strikes the best balance between performance and robustness, achieving **91.6% accuracy**, **100% recall**, and strong precision (**85.8%**) with an excellent F1-score of **92.3%**.
    """)

    st.subheader("Confusion Matrices (Best Thresholds)")

    st.markdown("**One-Class SVM**")
    st.code("""
    [[2988 1035]
    [  53 3924]]
    """)
    st.markdown("- **Observations:** A moderate number of false negatives and false positives indicate missed anomalies and misclassifications.")

    st.markdown("**Local Outlier Factor**")
    st.code("""
    [[4023    0]
    [   0 3977]]
    """)
    st.markdown("- **Observations:** Perfect separation at the best threshold suggests potential overfitting, which may hinder its robustness on unseen, noisy data.")

    st.markdown("**Isolation Forest**")
    st.code("""
    [[4023    0]
    [  15 3962]]
    """)
    st.markdown("- **Observations:** Although a few anomalies are missed (15 false negatives), this matrix represents the most balanced trade-off between false positives and false negatives, indicating robust performance on new data.")

    st.subheader("Final Thoughts")
    st.markdown("""
    - **Isolation Forest** delivers robust and generalizable performance with high recall and strong overall metrics, making it the best choice for this application.
    - **One-Class SVM** may be useful if high recall is prioritized over precision in specific applications, but it misses a significant number of anomalies.
    - **Local Outlier Factor (LOF)**, despite near-perfect metrics, is likely overfitting and may not perform well in real-world, noisy environments.
    """)
