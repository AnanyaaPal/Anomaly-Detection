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
# ---------------------------------
# Load data once, cache to speed up
# ---------------------------------
@st.cache_data
def load_data():
    X_train, X_test, test_ground_truth = make_data()
    return X_train, X_test, test_ground_truth

X_train, X_test, test_ground_truth = load_data()

# -------------------------------
# 1) Create top-level radio for main sections
# -------------------------------
st.sidebar.title("Navigation")
page_choice = st.sidebar.radio(
    "Select a page",
    ["Home", "Models", "Model Comparison"]
)

# ---------------------------------
# HOME PAGE
# ---------------------------------
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
    fig = plot_data(X_train, X_test, test_ground_truth)  # This function directly plots
    st.pyplot(fig)

    st.markdown("""
    ---
    ## My Solution
    We will explore **three models** for anomaly detection:
    1. **One-Class SVM**
    2. **Isolation Forest**
    3. **LocalOutlierFactor**

    Select a model from the sidebar to see its performance on this data.
    """)
# -------------------------------
# MODELS SECTION
# -------------------------------
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

        # Sidebar hyperparameters for One-Class SVM
        st.sidebar.subheader("One-Class SVM Settings")
        kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
        nu = st.sidebar.slider("Nu", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"], index=0)

        # Train the model
        st.subheader("Model Training & Evaluation")
        model = OneClassSVMModel(kernel=kernel, nu=nu, gamma=gamma)
        model.fit(X_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Evaluate
        metrics = model.evaluate(X_test, test_ground_truth)
        st.write("**Evaluation Metrics:**")
        for k, v in metrics.items():
            st.write(f"**{k}:** {v:.3f}")

        # Decision boundary visualization
        st.subheader("Decision Boundary Visualization")
        def plot_decision_boundary(model, X_train, X_test=None, X_outliers=None, title="One-Class SVM Decision Boundary"):
            """
            Plots:
            - Training data (blue)
            - Test data (orange)
            - Anomalies (green)
            - The decision boundary (red contour) and outlier region (orange shading)
            """
            import matplotlib.pyplot as plt
            import numpy as np
            
            if X_train.shape[1] != 2:
                st.error("Visualization is supported only for 2D data.")
                return None
            
            # Combine points to set up the plotting range
            X_all = X_train
            if X_test is not None:
                X_all = np.vstack([X_all, X_test])
            if X_outliers is not None:
                X_all = np.vstack([X_all, X_outliers])
            
            x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
            y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
            
            # Create a mesh grid for contour plotting
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                np.linspace(y_min, y_max, 300))
            Z = model.model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            # Shade the outlier region (decision_function < 0)
            ax.contourf(xx, yy, Z, levels=[Z.min(), 0], colors='orange', alpha=0.3)
            # Plot the boundary line where decision_function = 0
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
            
            # Plot the data
            ax.scatter(X_train[:, 0], X_train[:, 1], color='blue', alpha=0.7, label='Training Data (ok)')
            if X_test is not None:
                ax.scatter(X_test[:, 0], X_test[:, 1], color='orange', alpha=0.7, label='Test Data (ok)')
            if X_outliers is not None:
                ax.scatter(X_outliers[:, 0], X_outliers[:, 1], color='green', alpha=0.7, label='Anomalies')
            
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            ax.legend(loc='upper left')
            return fig

        outlier_mask = (test_ground_truth == -1)
        fig_boundary = plot_decision_boundary(model, X_train, X_test=X_test, X_outliers=X_test[outlier_mask])
        st.pyplot(fig_boundary)

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
            options=[100, 200, 500, 1000], 
            index=0
        )
        # For max_samples, you can either use 'auto' or specify an integer (since your training set has 10,000 samples)
        max_samples = st.sidebar.selectbox(
            "Max Samples", 
            options=["auto", 5000, 7500, 10000], 
            index=0
        )
        # Contamination: if you have an approximate fraction of anomalies, or use 'auto'
        contamination = st.sidebar.selectbox(
            "Contamination", 
            options=["auto", 0.01, 0.05, 0.1, 0.2], 
            index=2  # default to 0.05 or change as appropriate
        )
        # For max_features, you can choose 'auto' (all features) or a fraction (since there are only 2 features)
        max_features = st.sidebar.selectbox(
            "Max Features", 
            options=[1.0, 0.8, 0.5],
            index=0
        )
        random_state = st.sidebar.number_input("Random State", value=42, step=1)

        st.subheader("Model Training & Evaluation")
        # Import code for model training and evaluation
        from models.isolation_forest_model import IsolationForestModel
        iso_model = IsolationForestModel(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state
        )
        iso_model.fit(X_train)

        # Predict on test data and evaluate
        y_pred = iso_model.predict(X_test)
        metrics = iso_model.evaluate(X_test, test_ground_truth)
        st.write("**Evaluation Metrics:**")
        for key, val in metrics.items():
            st.write(f"**{key}:** {val:.3f}")

        st.subheader("Decision Boundary Visualization")
        # Plot the decision boundary using a function similar to the one defined for One-Class SVM
        def plot_decision_boundary(model, X_train, X_test=None, X_outliers=None, title="Isolation Forest Decision Boundary"):
            import matplotlib.pyplot as plt
            import numpy as np
            if X_train.shape[1] != 2:
                st.error("Visualization is supported only for 2D data.")
                return None

            # Combine points to set the plot range
            X_all = X_train
            if X_test is not None:
                X_all = np.vstack([X_all, X_test])
            if X_outliers is not None:
                X_all = np.vstack([X_all, X_outliers])
            x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
            y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1

            # Create a grid for contour plotting
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                np.linspace(y_min, y_max, 300))
            Z = model.model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            fig, ax = plt.subplots(figsize=(6,5))
            ax.contourf(xx, yy, Z, levels=[Z.min(), 0], colors='orange', alpha=0.3)
            ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')

            # Plot data points with three labels: Training (blue), Test (orange), and Anomalies (green)
            ax.scatter(X_train[:, 0], X_train[:, 1], color='blue', alpha=0.7, label='Training Data (ok)')
            if X_test is not None:
                ax.scatter(X_test[:, 0], X_test[:, 1], color='orange', alpha=0.7, label='Test Data (ok)')
            if X_outliers is not None:
                ax.scatter(X_outliers[:, 0], X_outliers[:, 1], color='green', alpha=0.7, label='Anomalies')
            
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            ax.legend(loc='upper left')
            return fig

        # Determine outlier indices based on ground truth for visualization
        outlier_mask = (test_ground_truth == -1)
        fig_boundary = plot_decision_boundary(iso_model, X_train, X_test=X_test, X_outliers=X_test[outlier_mask])
        st.pyplot(fig_boundary)

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

    elif model_choice == "Local Outlier Factor":
        st.subheader("LocalOutlierFactor")
        st.markdown("""
        The Local Outlier Factor (LOF) model is trained on normal data (assumed from the training set) 
        and then used to detect anomalies in the test set. LOF is a local density-based method that 
        assigns an outlier score to each sample by comparing its density with that of its neighbors.  
        Adjust the hyperparameters below to see how the model's performance and visualization change.
        """)

        # Sidebar hyperparameters for Local Outlier Factor
        st.sidebar.subheader("Local Outlier Factor Settings")
        n_neighbors = st.sidebar.number_input("Number of Neighbors", value=20, min_value=5, max_value=100, step=1)
        contamination = st.sidebar.selectbox("Contamination", options=["auto", 0.01, 0.05, 0.1, 0.2], index=0)
        novelty_str = st.sidebar.selectbox("Novelty", options=["True", "False"], index=1)
        novelty = True if novelty_str == "True" else False

        st.subheader("Model Training & Evaluation")
        from models.local_outlier_factor_model import LocalOutlierFactorModel
        lof_model = LocalOutlierFactorModel(n_neighbors=n_neighbors, contamination=contamination, novelty=novelty)
        lof_model.fit(X_train)

        # For novelty=True, we can use predict on new data.
        y_pred = lof_model.predict(X_test)
        metrics = lof_model.evaluate(X_test, test_ground_truth)
        st.write("**Evaluation Metrics:**")
        for key, val in metrics.items():
            st.write(f"**{key}:** {val:.3f}")

        st.subheader("LOF Prediction Visualization")
        def plot_lof_predictions(X_train, X_test=None, X_outliers=None, title="LocalOutlierFactor Results"):
            import matplotlib.pyplot as plt
            if X_train.shape[1] != 2:
                st.error("Visualization is supported only for 2D data.")
                return None
            fig, ax = plt.subplots(figsize=(6,5))
            # Plot training data in blue
            ax.scatter(X_train[:, 0], X_train[:, 1], color='blue', alpha=0.7, label='Training data (ok)')
            # Plot test data in orange
            if X_test is not None:
                ax.scatter(X_test[:, 0], X_test[:, 1], color='orange', alpha=0.7, label='Test data (ok)')
            # Plot anomalies in green
            if X_outliers is not None:
                ax.scatter(X_outliers[:, 0], X_outliers[:, 1], color='green', alpha=0.7, label='Not ok data')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title)
            ax.legend(loc='upper left')
            return fig

        # Separate test data into "ok" and "not ok" based on ground truth
        outlier_mask = (test_ground_truth == -1)
        X_test_ok = X_test[~outlier_mask]
        X_test_not_ok = X_test[outlier_mask]
        
        fig_lof = plot_lof_predictions(X_train, X_test=X_test_ok, X_outliers=X_test_not_ok, title="LocalOutlierFactor Results")
        st.pyplot(fig_lof)

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

# ---------------------------------
# MODEL COMPARISON
# ---------------------------------
elif page_choice == "Model Comparison ":
    st.title("Model Comparison (coming soon)")
    st.markdown("Information about the comparison. (coming soon)")
