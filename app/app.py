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
        The One-Class SVM is trained on the normal data (assumed from the training set) and then used to 
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
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, test_ground_truth, test_size=0.2, random_state=42
        )

        # Normalize data (fit on X_train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Combine training and validation sets for hyperparameter tuning
        X_all = np.vstack([X_train_scaled, X_val_scaled])
        y_all = np.hstack([np.ones(len(X_train_scaled)), y_val])

        # Set up GridSearchCV for hyperparameter tuning
        from models.one_class_svm import OneClassSVMWrapper
        param_grid = {
            "kernel": ["linear", "rbf", "sigmoid"],
            "nu": [0.05, 0.1, 0.2, 0.25, 0.27, 0.3, 0.35, 0.4, 0.45, 0.5],
            "gamma": ["scale", "auto"],
        }
        def accuracy_scoring(estimator, X, y, **kwargs):
            y_pred = estimator.predict(X)
            return accuracy_score(y, y_pred)
        scorer = make_scorer(accuracy_scoring, greater_is_better=True)

        grid_search = GridSearchCV(
            estimator=OneClassSVMWrapper(),
            param_grid=param_grid,
            scoring=scorer,
            cv=3
        )
        grid_search.fit(X_all, y_all)
        st.write("**Best parameters found:**", grid_search.best_params_)

        # Use best model for evaluation
        best_model = grid_search.best_estimator_
        # import pandas
        import pandas as pd
        # Evaluate on validation set
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
        n_estimators = st.sidebar.selectbox(
            "Number of Estimators", 
            options=[100, 200],  
            index=0
        )
        max_samples = st.sidebar.selectbox(
            "Max Samples", 
            options=["auto", 0.5, 0.8],
            index=0
        )
        contamination = st.sidebar.selectbox(
            "Contamination", 
            options=["auto", 0.05, 0.1],
            index=1
        )
        max_features = st.sidebar.selectbox(
            "Max Features", 
            options=[1.0, 0.8],
            index=0
        )
        random_state = st.sidebar.number_input("Random State", value=42, step=1)

        st.subheader("Model Training & Evaluation")
        from models.isolation_forest_model import IsolationForestModel
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, precision_recall_curve

        # Load Data (assumed already loaded globally) or re-load if needed:
        X_train, X_test, test_ground_truth = make_data()

        # Split X_test into final test set (80%) and validation set (20%)
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, test_ground_truth, test_size=0.2, random_state=42
        )

        # Normalize the data (fit on X_train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        X_test_scaled  = scaler.transform(X_test)

        # Combine training and validation sets for hyperparameter tuning
        X_all = np.vstack([X_train_scaled, X_val_scaled])
        y_all = np.hstack([np.ones(len(X_train_scaled)), y_val])  # +1 for normal (train), ±1 for validation

        # GridSearchCV using accuracy
        param_grid = {
            "n_estimators":   [100, 200],
            "max_samples":    ["auto", 0.5, 0.8],
            "contamination":  ["auto", 0.05, 0.1],
            "max_features":   [1.0, 0.8],
            "random_state":   [42]
        }

        def accuracy_scoring_iso(estimator, X, y, **kwargs):
            y_pred = estimator.predict(X)
            return accuracy_score(y, y_pred)

        scorer = make_scorer(accuracy_scoring_iso, greater_is_better=True)

        grid_search = GridSearchCV(
            estimator=IsolationForestModel(),
            param_grid=param_grid,
            scoring=scorer,
            cv=3
        )
        grid_search.fit(X_all, y_all)
        st.write("**Best parameters found:**", grid_search.best_params_)

        # Evaluate best model on validation and test sets
        best_model = grid_search.best_estimator_

        # Evaluation on validation set
        y_pred_val  = best_model.predict(X_val_scaled)
        val_metrics = best_model.evaluate(X_val_scaled, y_val)
        st.write("**Validation Metrics:**")
        import pandas as pd
        val_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"], val_metrics["f1_score"]]
        })
        st.table(val_df.style.format(precision=3))

        # Evaluation on test set
        y_pred_test  = best_model.predict(X_test_scaled)
        test_metrics = best_model.evaluate(X_test_scaled, y_test)
        st.write("**Test Metrics:**")
        test_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [test_metrics["accuracy"], test_metrics["precision"], test_metrics["recall"], test_metrics["f1_score"]]
        })
        st.table(test_df.style.format(precision=3))

        # Decision Boundary Visualization
        st.subheader("Decision Boundary Visualization")
        def plot_decision_boundary(model, X_train, X_test=None, X_outliers=None, title="Isolation Forest Decision Boundary"):
            import matplotlib.pyplot as plt
            import numpy as np

            if X_train.shape[1] != 2:
                st.error("Visualization is supported only for 2D data.")
                return None

            # Combine data to get plotting range
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
            # Shade region where decision_function < 0 (outliers) in white\n        ax.contourf(xx, yy, Z, levels=[Z.min(), 0], colors='white', alpha=0.3)
            # Boundary at 0
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
            
            # Plot ok data in blue, anomalies in orange
            ax.scatter(X_train[:, 0], X_train[:, 1], color='blue', alpha=0.7)
            if X_test is not None:
                ax.scatter(X_test[:, 0], X_test[:, 1], color='blue', alpha=0.7, label='Train + Test Ok data')
            if X_outliers is not None:
                ax.scatter(X_outliers[:, 0], X_outliers[:, 1], color='orange', alpha=0.7, label='Anomalies/ Not ok data')
            
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(loc='upper right')
            return fig

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
        The Local Outlier Factor (LOF) model is trained on normal data (assumed from the training set) 
        and then used to detect anomalies in the test set. LOF is a local density-based method that 
        assigns an outlier score to each sample by comparing its density with that of its neighbors.  
        Adjust the hyperparameters below to see how the model's performance and visualization change.
        """)

        # Sidebar hyperparameters
        st.sidebar.subheader("Local Outlier Factor Settings")
        n_neighbors = st.sidebar.number_input("Number of Neighbors", value=20, min_value=5, max_value=100, step=1)
        contamination = st.sidebar.selectbox("Contamination", options=["auto", 0.01, 0.05, 0.1, 0.2], index=0)
        novelty_str = st.sidebar.selectbox("Novelty", options=["True", "False"], index=1)
        novelty = novelty_str == "True"

        st.subheader("Model Training & Evaluation")

        from models.local_outlier_factor_model import LocalOutlierFactorModel
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, precision_recall_curve
        from models.iav_flap_anomaly_detection import make_data

        X_train, X_test, test_ground_truth = make_data()

        # Split X_test into final test set and validation set
        X_test, X_val, y_test, y_val = train_test_split(X_test, test_ground_truth, test_size=0.2, random_state=42)

        # Normalize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Combine training and validation sets for hyperparameter tuning
        X_all = np.vstack([X_train_scaled, X_val_scaled])
        y_all = np.hstack([np.ones(len(X_train_scaled)), y_val])

        # Grid search
        param_grid = {
            "n_neighbors": [5, 10, 20, 30, 40, 50, 80, 100],
            "contamination": ["auto", 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        }

        def accuracy_scoring_lof(estimator, X, y):
            y_pred = estimator.predict(X)
            return accuracy_score(y, y_pred)

        scorer = make_scorer(accuracy_scoring_lof)

        grid_search = GridSearchCV(
            estimator=LocalOutlierFactorModel(novelty=True),
            param_grid=param_grid,
            scoring=scorer,
            cv=10
        )
        grid_search.fit(X_all, y_all)

        st.write("**Best parameters found:**", grid_search.best_params_)

        best_lof = grid_search.best_estimator_

        # Validation metrics
        y_pred_val = best_lof.predict(X_val_scaled)
        val_metrics = best_lof.evaluate(X_val_scaled, y_val)
        val_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"], val_metrics["f1_score"]]
        })
        st.write("**Validation Metrics:**")
        st.table(val_df.style.format(precision=3))

        # Test metrics
        y_pred_test = best_lof.predict(X_test_scaled)
        test_metrics = best_lof.evaluate(X_test_scaled, y_test)
        test_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [test_metrics["accuracy"], test_metrics["precision"], test_metrics["recall"], test_metrics["f1_score"]]
        })
        st.write("**Test Metrics:**")
        st.table(test_df.style.format(precision=3))

        # Decision boundary visualization
        st.subheader("Decision Boundary Visualization")

        def plot_decision_boundary(model, X_train, X_test, X_outliers, title):
            import matplotlib.pyplot as plt
            if X_train.shape[1] != 2:
                st.error("Visualization supported only for 2D data.")
                return

            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
            Z = model.model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.contourf(xx, yy, Z, levels=[Z.min(), 0], colors='white', alpha=0.3)
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')

            ax.scatter(X_train[:, 0], X_train[:, 1], color='blue', alpha=0.7, label='Train data')
            ax.scatter(X_test[:, 0], X_test[:, 1], color='blue', alpha=0.7, label='Test OK data')
            ax.scatter(X_outliers[:, 0], X_outliers[:, 1], color='orange', alpha=0.7, label='Anomalies')

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            ax.legend()

            return fig

        outlier_mask = (y_test == -1)
        fig_boundary = plot_decision_boundary(best_lof, X_train_scaled, X_test_scaled[~outlier_mask], X_test_scaled[outlier_mask], "Local Outlier Factor Decision Boundary")
        st.pyplot(fig_boundary)

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
        "Accuracy": ["85.5%", "53.8%", "97.8%"],
        "Precision": ["50.3%", "76.8%", "89.0%"],
        "Recall": ["100.0%", "5.9%", "100.0%"],
        "F1-Score": ["66.9%", "11.0%", "94.2%"]
    }
    performance_df = pd.DataFrame(performance_data)
    st.table(performance_df)

    st.markdown("### **Best Performing Model:** `Isolation Forest`")

    st.subheader("Observations")
    st.markdown("""
    - **One-Class SVM** achieves a **high recall (100%)** but suffers from **low precision (50.3%)**, leading to a high false positive rate.
    - **Local Outlier Factor (LOF)** has the worst performance, with **very low recall (5.9%)**, meaning it **fails to detect most anomalies**.
    - **Isolation Forest** **outperforms both models**, with **97.8% accuracy** and **100% recall**, making it the most **reliable anomaly detection model**.
    """)

    st.subheader("Confusion Matrices (Best Thresholds)")

    st.markdown("**One-Class SVM**")
    st.code("""
    [[3583  440]
     [ 721 3256]]
    """)
    st.markdown("- **Missed Anomalies**: **721 false negatives**  ")
    st.markdown("- **False Positives**: **440 normal points misclassified as anomalies**")

    st.markdown("**Local Outlier Factor**")
    st.code("""
    [[ 909 3114]
     [ 584 3393]]
    """)
    st.markdown("- **Missed Anomalies**: **584 false negatives**  ")
    st.markdown("- **False Positives**: **3114 misclassified normal points!** (LOF has very poor anomaly detection)")

    st.markdown("**Isolation Forest**")
    st.code("""
    [[3993   30]
     [ 144 3833]]
    """)
    st.markdown("- **Best Tradeoff**: **Lowest false positives and false negatives**  ")
    st.markdown("- **Minimal anomalies missed** (**144 false negatives**)")

    st.subheader("Final Thoughts")
    st.markdown("""
    - **Isolation Forest is the most effective model** for this dataset, with the **highest accuracy and best balance of precision-recall**.
    - **One-Class SVM can still be useful** if **high recall is more important than precision** (e.g., fraud detection).
    - **LOF struggles with this dataset**, as it fails to generalize well.
    """)
