import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Import your data helper functions
from models.iav_flap_anomaly_detection import make_data, plot_data

# Import the One-Class SVM model
from models.one_class_svm import OneClassSVMModel

# ---------------------------------
# Load data once, cache to speed up
# ---------------------------------
@st.cache_data
def load_data():
    X_train, X_test, test_ground_truth = make_data()
    return X_train, X_test, test_ground_truth

X_train, X_test, test_ground_truth = load_data()

# ---------------------------------
# Sidebar: Model Selection
# ---------------------------------
st.sidebar.title("Navigation")
page_choice = st.sidebar.selectbox(
    "Select a page",
    ["Home", "One-Class SVM", "Isolation Forest", "LocalOutlierFactor"]
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

# ---------------------------------
# ONE-CLASS SVM PAGE
# ---------------------------------
elif page_choice == "One-Class SVM":
    st.title("One-Class SVM Model")
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

    # Optionally, display model code
    with st.expander("Show One-Class SVM Model Code"):
        try:
            with open("models/one_class_svm.py", "r") as file:
                code = file.read()
            st.code(code, language="python")
        except Exception as e:
            st.write("Could not load model code:", e)

# ---------------------------------
# ISOLATION FOREST PAGE (Placeholder)
# ---------------------------------
elif page_choice == "Isolation Forest (coming soon)":
    st.title("Isolation Forest - Coming Soon")
    st.markdown("Integration of Isolation Forest code will be provided here in the future.")

# ---------------------------------
# LOCAL OUTLIER FACTOR PAGE (Placeholder)
# ---------------------------------
elif page_choice == "LocalOutlierFactor (coming soon)":
    st.title("LocalOutlierFactor - Coming Soon")
    st.markdown("Integration of LOF code will be provided here in the future.")
