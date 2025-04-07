---
permalink: /portfolio/
title: "Portfolio"
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"

---
*Updated: 04/01/2025*
## 🧠Machine Learning Models Applications
### Automatic Model Hyperparameters GridSearch for Classification Models Using Python and Streamlit

* This project is a modular machine learning pipeline built using Python and Streamlit. Through an intuitive web interface, users can preprocess data, explore datasets, train machine learning models, and make predictions on unseen data. The pipeline supports a machine learning workflow, from data cleaning to model evaluation and prediction.
[![View the python scripts on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Gazmaths/ML_classification_app)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Visit_App-red?logo=streamlit&logoColor=white)](https://mlclassificationpipeline.streamlit.app/)

**Features**
* Preprocessing: Handle missing values and outliers with user-defined options.
* Exploratory Data Analysis: Analyze target correlation and generate pairplots for insights.
* Model Training: Train and tune multiple models (Random Forest, Logistic Regression, XGBoost, SVC) using GridSearchCV.
* Model Selection and Evaluation: Automatically pick the best model and evaluate its performance.
* Prediction: Predict outcomes for unseen data with a trained model.
* Visualization: View feature contributions using SHAP.

**Files and Structure**
preprocess.py: Functions for data preprocessing (missing values, outliers).
explore.py: Functions for exploratory data analysis.
train.py: Functions for training and tuning machine learning models.
model.py: Functions for selecting the best model and evaluating feature importance.
prediction.py: Functions for preprocessing and predicting outcomes on unseen data.
classification_app.py: The main Streamlit app that integrates all modules.
requirements.txt: List of dependencies for the project.

**How to Run**
* Install dependencies:
* Copy code
* pip install -r requirements.txt
* Start the Streamlit app:
* bash
* Copy code
* streamlit run classification_app.py
* Follow the steps in the app to upload data, preprocess, explore, train models, and make predictions.

**Deployment**
This app can be deployed to Streamlit Cloud for easy sharing and accessibility.

**Technologies Used**
* Python
* Streamlit
* Scikit-learn
* XGBoost
* Seaborn
* Matplotlib

<img src="https://raw.githubusercontent.com/Gazmaths/ML_classification_app/main/streamlit.png" width="580">


### 📊 Interactive Data Analysis App Using Python and Streamlit

This web-based app allows users to easily upload CSV files and perform exploratory data analysis and linear regression modeling. Built with Streamlit, the app features:

📁 CSV file upload and preview

📈 Summary statistics and correlation heatmaps

📌 Interactive scatter plots with trendlines (via Plotly)

📊 Histograms for data distribution

📉 Linear regression modeling with RMSE and coefficient outputs

📈 Visualization of actual vs. predicted values
[![View the python scripts on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Gazmaths/data_analysis_app)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Visit_App-red?logo=streamlit&logoColor=white)](https://gazal1app.streamlit.app/)

### Optimizing Landslide Susceptibility Models through Ensemble Classification Techniques
## 👁️‍🗨️ Computer Vision Methods
## 🤖 Building Chatbots with Pre-trained LLM Models
## 📝 Natural Language Processing Methods
