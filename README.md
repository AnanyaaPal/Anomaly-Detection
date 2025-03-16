[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://192.168.0.102:8501)

    ```md
    ![IAV Logo](assets/images/IAV_logo.png)
    ```
# Anomaly-Detection
This project focuses on developing and testing models to detect anomalies and choose the model with the best results. 

## Description
The motivation to build this project was simply to challenge my analytical skills to come up with a solution to solve the anomaly detection task, where we have unlabelled/ partially labelled data, and the system has a problem that makes the data shift and flip. 
It solves the problem of detecting outliers in the data with a visual support aid. Three unsupervised learning algorithms from the sk-learn library have been utilised - One-Class SVM, Local Outlier Factor and Isolation Fores - each fitting the training dataset with a unique assumption. 
The project allowed me to delve deeper into the field of anomaly detection using pre-defined algorithms and utilise my core knowledge about statistical and machine learning concepts to build the aforementioned models. 

## Table of Contents

If your README is long, add a table of contents to make it easy for users to find what they need.

- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)

## Installation
Open a terminal and run the following:

Clone the repository:
```bash
git clone https://github.com/AnanyaaPal/Anomaly-Detection.git
cd Anomaly-Detection
```
Install the dependencies:
```bash
pip install -r requirements.txt
```

Install streamlit to run model deployment:
```bash
pip install streamlit
streamlit hello
```
If this opens the _Streamlit Hello_ app in your browser, you're all set! If not, head over to [the documentation](https://docs.streamlit.io/get-started) for specific installs.

Run the example:
```bash
cd Anomaly-Detection/app
python streamlit run app.py
```
<img src="https://user-images.githubusercontent.com/7164864/217936487-1017784e-68ec-4e0d-a7f6-6b97525ddf88.gif" alt="Streamlit Hello" width=500 href="none"></img>

## Usage
1. **Installation:**
   Follow the instructions. 

2. **Training and Evaluation:**
   - Run the models/ folder to generate the dataset and run the models.
   - Run .ipynb evaluate the trained models using the test set and analyze the performance metrics.

4. **Deployment:**
   - Run the example using the streamlit command prompt (from installation) to evaluate metrics and visualise the detected anomalies. 


## Credits

List your collaborators, if any, with links to their GitHub profiles.

If you used any third-party assets that require attribution, list the creators with links to their primary web presence in this section.

If you followed tutorials, include links to those here as well.

## License

The last section of a high-quality README file is the license. This lets other developers know what they can and cannot do with your project. If you need help choosing a license, refer to [https://choosealicense.com/](https://choosealicense.com/).

---

## Badges
![Static Badge](https://img.shields.io/badge/MIT%20License)


Badges aren't necessary, per se, but they demonstrate street cred. Badges let other developers know that you know what you're doing. Check out the badges hosted by [shields.io](https://shields.io/). You may not understand what they all represent now, but you will in time.

## Features

If your project has a lot of features, list them here.

## How to Contribute

If you created an application or package and would like other developers to contribute it, you can include guidelines for how to do so. The [Contributor Covenant](https://www.contributor-covenant.org/) is an industry standard, but you can always write your own if you'd prefer.

## Tests

Go the extra mile and write tests for your application. Then provide examples on how to run them here.
