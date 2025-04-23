---
permalink: /portfolio/
title: "Portfolio"
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"

---
*Updated: 04/01/2025*
## 📊Data Science and Analytics
### 🚗 CarMax Hackathon Using Python Visualization, Machine Learning Tools, and Streamlit
**Problem Statement**
Many customers decide to trade in their current vehicle when purchasing a car at CarMax. 
What does a customer's appraised vehicle tell us about the vehicle they will ultimately purchase? How can CarMax use this information to improve business operations and provide customers with a personalized shopping experience?

**Solution**
I created an interactive app that visualizes CarMax's data, highlighting the distribution of cars across various states, makes, and models. Additionally, I built a Random Forest classifier using encoded data to predict customers' likelihood of switching car makes, helping CarMax optimize its offerings and customer experience. The model performs well in identifying customers who switch car makes, but has lower precision for those who stay with the same make due to an imbalanced dataset. Future improvements could involve applying techniques like oversampling, undersampling, or SMOTE to mitigate the effects of class imbalance and improve model performance for customers who remain with the same make. This insight could guide targeted marketing strategies.

<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://lookerstudio.google.com/reporting/d1cb1221-836f-40eb-b019-15e23b693c08" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/93ea07059672493cb156ba00e0d2db31?sid=e5a9f021-cd57-453a-9aa8-51f4e1b4894d" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

## 🧠Machine Learning Models Applications
### 🔍 Automatic Model Hyperparameters GridSearch for Classification Models Using Python and Streamlit

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

### ✅ Machine Learning Course Project: Detection of Anomalies in Credit Card Transactions Using Supervised and Unsupervised Learning Techniques
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/Gazmaths/Credit-card-anomaly-detection-using-machine-learning-/blob/main/Anomalies_detection_in_credit_card_transactions.ipynb)
[![Read the full paper project submission](https://img.shields.io/badge/Google%20Drive-Read_Paper_on_Google_Drive-blue?logo=GoogleDrive)](https://drive.google.com/file/d/161G3yPrFjx5xYRpna_SIb3poLg6aMGTv/view?usp=drive_link)

Abstract -- In the financial sector, financial fraud is a rising problem with serious effects. The identification of credit card fraud in online transactions was greatly helped by big data analysis. Due to two main factors—first, the profiles of legitimate and fraudulent behavior change often, and second, credit card
fraud data sets are extremely skewed—detection of credit card fraud, a data mining challenge, becomes difficult. The dataset sampling strategy, variable choice, and detection methods employed all have significant impacts on the effectiveness of fraud detection in credit card transactions. The performance of isolation forest (iForest), local outliers’ factor(LOF), and Logistic regression (LR) on highly skewed credit card fraud data are examined in this study. The model was constructed using real data from European cardholders, and under-sampling methods were also applied. The three models were implemented in Python, and the effectiveness of the methods is assessed based on accuracy, recall, precision, and the Kappa score coefficient. The outcomes indicate that Isolation Forest has the best accuracy at 99.8% in detecting outliers.

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

### 🚀 Optimizing Landslide Susceptibility Models through Ensemble Classification Techniques
This study aims to assess how well machine learning models perform on landslide susceptibility mapping. For Polk County, Western North Carolina, seven distinct models were examined for landslide susceptibility modeling: Support Vector Machine (SVM), Logistic Regression (LR), Linear Discriminant Analysis (LDA), Random Forest (RF), and Extreme Gradient Boosting (XGBoost) and ensemble techniques (Stacking and Weighted Average). A dataset of 1215 historical landslide events and 1215 non-landslide sites, as well as fourteen geographic data layers, is used to evaluate these models.
Metrics including accuracy, F1-score, Kappa score, and AUC-ROC are used to assess these models' performance, with a focus on how non-landslide random sampling affects model outcomes.

The buffer-based scenario's AUC-ROC was 91.8%, and the slope threshold scenario's was 99.4%. The weighted average ensemble of the five models yielded the best results. This demonstrates the effectiveness of machine learning in landslide susceptibility mapping, offering a strong instrument for pinpointing high-risk regions and guiding plans for disaster risk reduction. 
For more details, refer to the full paper:  
<img src="https://gazmaths.github.io/assets/images/optimizingmetric.png" width="580">{: .align-center}
<img src="https://gazmaths.github.io/assets/images/optimizingWA.png" width="580">{: .align-center}
[Optimizing landslide susceptibility mapping using machine learning and geospatial techniques](https://www.sciencedirect.com/science/article/pii/S1574954124001250)

## 👁️‍🗨️ Computer Vision Methods
### 👁️‍🗨️  Automatic Landslide Detection using a Fine-Tuned YOLOv11 Model
I employed the YOLOv11 model from Ultralytics, utilizing a transfer learning approach with a high-resolution Roboflow landslide dataset, to improve landslide detection accuracy in Polk County, NC, and to provide valuable insights for developing effective hazard mitigation strategies.
<img src="https://gazmaths.github.io/assets/images/Yolo11detection.png" width="680">{: .align-center}

### ⚠️ Landslide Identification using Segmentation Models
The objective of this project is to investigate the potential of the application of deep learning algorithms to analyze remote sensing data for landslide detection, utilizing an open-source dataset obtained from Meena et al. 2023.
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Gazmaths/Landslide-detection-using-semantic-segmentation-algorithms/blob/main/Landslide%20Detection%20Using%20Semantic%20Segmentation%20Algorithms.ipynb)
<img src="https://gazmaths.github.io/assets/images/segmentationmodel.png" width="680">{: .align-center}
## 🤖 Generative AI Applications and Building Chatbots with Pre-trained LLM Models
### 💡 Creating EDU-Navigator for HBCU students(HP Future of Work Academy (FOWA) Challenge)

I worked as a software developer on a team named "20," consisting of 5 members, to develop the EDU-Navigator chatbot for HBCU students. The chatbot, powered by the Chat-GPT turbo engine, was designed to help students navigate their educational journey. Our team received an honorable mention in the HP Future of Work Academy (FOWA) Challenge.

### 📚 Google 5-Day Generative AI Intensive Training

I participated in the Google 5-Day Generative AI intensive training, which included a hands-on assignment each day on Kaggle.

**Day 1 - Prompting** 
I worked through example prompts and techniques using the Python SDK and AI Studio, with further insights provided in the Prompting whitepaper. I also explored various apps built with the Gemini family of models for inspiration. ![View on Kaggle](https://img.shields.io/badge/Notebook-Kaggle-yellow)[Notebook](https://www.kaggle.com/code/gazaliagboola/day-1-prompting)

**Day 1 - Evaluation and Structured Output**
I focused on evaluating the performance of LLMs in real-world tasks, specifically learning how to assess and measure their outputs effectively. ![View on Kaggle](https://img.shields.io/badge/Notebook-Kaggle-yellow)[Notebook](https://www.kaggle.com/code/gazaliagboola/day-1-evaluation-and-structured-output)

**Day 2 - Classifying Embeddings with Keras**
I learned how to classify embeddings using Keras and the Gemini API, enhancing my understanding of model classification. ![View on Kaggle](https://img.shields.io/badge/Notebook-Kaggle-yellow)[Notebook](https://www.kaggle.com/code/gazaliagboola/day-2-classifying-embeddings-with-keras)

**Day 2 - Embeddings and Similarity Scores**
I explored embeddings and similarity scores with the Gemini API, using its embedding endpoint to analyze similarity scores between different data points. ![View on Kaggle](https://img.shields.io/badge/Notebook-Kaggle-yellow)[Notebook](https://www.kaggle.com/code/gazaliagboola/day-2-embeddings-and-similarity-scores)

**Day 2 - Document Q&A with RAG**
I learned how to implement Retrieval Augmented Generation (RAG) with Chroma to enhance Document Q&A. I created a vector database with the Gemini API to retrieve relevant information and generate tailored answers, overcoming LLM limitations by processing external, unseen data and using Chroma for efficient document searching and embedding. ![View on Kaggle](https://img.shields.io/badge/Notebook-Kaggle-yellow)[Notebook](https://www.kaggle.com/code/gazaliagboola/day-2-document-q-a-with-rag)

**Day 3 - Function Calling with the Gemini API**
I explored function calling with the Gemini API, learning to build a chat interface over a local database using the API’s automatic function-calling feature. Though the example was a simple project without real-world safety constraints, it demonstrated how to integrate AI chat capabilities into existing applications. ![View on Kaggle](https://img.shields.io/badge/Notebook-Kaggle-yellow)[Notebook](https://www.kaggle.com/code/gazaliagboola/day-3-function-calling-with-the-gemini-api)

**Day 4 - Fine-Tuning a Custom Model**
I delved into fine-tuning a custom model using the Gemini API, focusing on training the model to classify newsgroup posts into their respective categories. The fine-tuning process enabled me to tailor the model for specific tasks like entity extraction, summarization, and creative generation. I also discovered that AI Studio supports creating and monitoring tuned models via a web UI using data from Google Sheets, Drive, or my own files. ![View on Kaggle](https://img.shields.io/badge/Notebook-Kaggle-yellow)[Notebook](https://www.kaggle.com/code/gazaliagboola/day-4-fine-tuning-a-custom-model)




## 📝 Natural Language Processing Methods

