[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://192.168.0.102:8501)
![Tests](https://github.com/AnanyaaPal/Anomaly-Detection/actions/workflows/python-tests.yml/badge.svg)

<p align="center">
  <img src="assets/IAV_logo.png" alt="IAV Logo" width="200">
</p>

# Anomaly Detection
This project involves developing, evaluating, and deploying various anomaly detection algorithms—including One-Class SVM, Isolation Forest, and Local Outlier Factor—to accurately identify anomalous patterns in data. The models are rigorously compared based on performance metrics to select the most effective approach for practical anomaly detection tasks.

---

## Description
The motivation behind this project was to challenge my analytical skills by addressing a realistic anomaly detection task, particularly involving unlabeled or partially labeled datasets where anomalies cause data to shift and flip. To tackle this, three powerful unsupervised machine learning algorithms from scikit-learn—One-Class SVM, Local Outlier Factor, and Isolation Forest—were implemented and systematically compared. Each model makes distinct assumptions about the underlying data distribution, providing valuable visual insights to support anomaly detection.

This project enabled me to deepen my understanding of anomaly detection methodologies, leverage statistical and machine learning knowledge, and effectively evaluate and interpret model performance through comprehensive visualizations and metrics.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Analytical Strategy](#analyticalstrategy)
- [Results](#results)
- [Credits](#credits)
- [License](#license)
- [Tests](#tests)

---

## Usage
Follow these steps to start analyzing anomalies in your datasets:

1. **Setup & Installation:**
- Clone the repository and navigate to the root directory:
```bash
git clone https://github.com/AnanyaaPal/Anomaly-Detection.git
cd Anomaly-Detection
```
- Install dependencies from the provided requirements file:
```bash
pip install -r requirements.txt
```
2. **Data Preparation & Model Analysis:**
- Open and run task.ipynb Jupyter Notebook to:
-- Train and tune anomaly detection models (One-Class SVM, Local Outlier Factor, Isolation Forest).
-- Evaluate models using accuracy, precision, recall, and F1-score metrics.
-- Visualize decision boundaries and analyze model performances.

3. **Deployment & Interactive Visualization:**
- Install streamlit:
```bash
pip install streamlit
streamlit hello
```
If this opens the _Streamlit Hello_ app in your browser, you're all set! If not, head over to [the documentation](https://docs.streamlit.io/get-started) for specific installs.

- Navigate to the path of the folder and launch the interactive app:
```bash
cd Anomaly-Detection/app
python streamlit run app.py
```
-The Streamlit UI provides:
-- Automatic hyperparameter tuning for optimal model configuration.
-- Interactive sidebar controls to manually adjust hyperparameters and observe changes.
-- Clear visualizations of decision boundaries and anomaly detection results.

---

## Analytical Strategy
This project explores three anomaly detection models—One-Class SVM, Isolation Forest, and Local Outlier Factor (LOF)—by performing hyperparameter tuning using GridSearchCV to identify optimal configurations. Each model is trained and evaluated on standardized training, validation, and test datasets, with model performance compared through accuracy, precision, recall, and F1-score. Decision boundaries and Precision-Recall curves are visualized to illustrate model effectiveness and facilitate informed model selection for anomaly detection.

---

## Results
### Model Performance Comparison

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| **One-Class SVM**     | 85.5%    | 50.3%     | 100.0% | 66.9%    |
| **Local Outlier Factor** | 53.8%    | 76.8%     | 5.9%   | 11.0%    |
| **Isolation Forest**  | 97.8%    | 89.0%     | 100.0% | 94.2%    |

### Best Performing Model: **Isolation Forest**

### Precision-Recall Curve (Isolation Forest)
![Isolation Forest Precision-Recall Curve](assets/Isolation_Forest_Precision-Recall%20Curve.png)

### Final Thoughts
- **Isolation Forest** clearly emerges as the most effective model for anomaly detection in this dataset, showcasing superior accuracy (97.8%) and an optimal balance between precision (89.0%) and recall (100%).
- **One-Class SVM** remains valuable in scenarios where recall is prioritized over precision, such as fraud detection.
- **Local Outlier Factor (LOF)** performs poorly on this dataset, indicating its limitations in generalizing well to the provided data structure.

---

## Credits

Streamlit for building the interactive web application.
<img src="https://user-images.githubusercontent.com/7164864/217936487-1017784e-68ec-4e0d-a7f6-6b97525ddf88.gif" alt="Streamlit Hello" width=500 href="none"></img>

---

## License

The MIT License applies to this repository. See [LICENSE](LICENSE) for more details.

---

## Features

- **Interactive Streamlit Application:** Intuitive user interface for anomaly detection.
- **Multiple Models:** Compare results of One-Class SVM, Isolation Forest, and Local Outlier Factor.
- **Automated Hyperparameter Tuning:** GridSearchCV integrated for optimal model parameters.
- **Data Visualization:** Clearly visualize decision boundaries and precision-recall curves.
- **Detailed Model Evaluation:** Comprehensive performance metrics (accuracy, precision, recall, F1-score) and confusion matrices.
- **Scalable & Reproducible:** Structured code allowing easy replication or extension for other datasets.

---

## How to Contribute

Contributions are warmly welcomed to further improve and expand anomaly detection techniques demonstrated in this project. To contribute:

- Fork the repository and create your branch (`git checkout -b feature/YourFeature`).
- Submit your changes through a clear and detailed pull request explaining the improvements.
- Open an issue if you encounter bugs, have feature requests, or general questions.

---

## Tests

Automated tests run via GitHub Actions upon each push or pull request to the `main` branch.  
To execute tests locally:

1. Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```
2. Run tests using:
```
python -m pytest --maxfail=1 --disable-warnings
```
The testing workflow configuration is defined in [`.github/workflows/python-tests.yml`](.github/workflows/python-tests.yml).
