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
This study aims to assess how well machine learning models perform on landslide susceptibility mapping. For Polk County, Western North Carolina, seven distinct models were examined for landslide susceptibility modeling: Support Vector Machine (SVM), Logistic Regression (LR), Linear Discriminant Analysis (LDA), Random Forest (RF), and Extreme Gradient Boosting (XGBoost) and ensemble techniques (Stacking and Weighted Average). A dataset of 1215 historical landslide events and 1215 non-landslide sites, and fourteen geographic data layers, is used to evaluate these models.
Metrics including accuracy, F1-score, Kappa score, and AUC-ROC are used to assess these models' performance, with a focus on how non-landslide random sampling affects model outcomes.

With an AUC-ROC of 91.8% for the buffer-based scenario and 99.4% for the slope threshold scenario, the weighted average ensemble of the five models yielded the best results. This demonstrates the effectiveness of machine learning in landslide susceptibility mapping, offering a strong instrument for pinpointing high-risk regions and guiding plans for disaster risk reduction. 
For more details, refer to the full paper:  
<img src="https://gazmaths.github.io/assets/images/optimizingmetric.png" width="580">{: .align-center}
<img src="https://gazmaths.github.io/assets/images/optimizingWA.png" width="580">{: .align-center}
[Optimizing landslide susceptibility mapping using machine learning and geospatial techniques](https://www.sciencedirect.com/science/article/pii/S1574954124001250)

## 👁️‍🗨️ Computer Vision Methods
### Automatic Landslide Detection using a Fine-Tuned YOLOv11 Model
I employed the YOLOv4 model from Ultralytics, utilizing a transfer learning approach with a high-resolution Roboflow landslide dataset, to improve landslide detection accuracy in Polk County, NC, and to provide valuable insights for developing effective hazard mitigation strategies.
<img src="https://gazmaths.github.io/assets/images/Yolo11detection.png" width="680">{: .align-center}

### Landslide Identification using Segmentation Models
The objective of this project is to investigate the potential of the application of deep learning algorithms to analyze remote sensing data for landslide detection, utilizing an open-source dataset obtained from Meena et al. 2023.
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/Gazmaths/Landslide-detection-using-semantic-segmentation-algorithms/blob/main/Landslide%20Detection%20Using%20Semantic%20Segmentation%20Algorithms.ipynb)
<img src="https://gazmaths.github.io/assets/images/segmentationmodel.png" width="680">{: .align-center}
## 🤖 Generative AI Applications and Building Chatbots with Pre-trained LLM Models
### Creating EDU-Navigator for HBCU students(HP Future of Work Academy (FOWA) Challenge)

I worked as a software developer on a team named "20," consisting of 5 members, to develop the EDU-Navigator chatbot for HBCU students. The chatbot, powered by the Chat-GPT turbo engine, was designed to help students navigate their educational journey. Our team received an honorable mention in the HP Future of Work Academy (FOWA) Challenge.

### Google 5-Day Generative AI Intensive Training

I participated in the Google 5-Day Generative AI intensive training, which included a hands-on assignment each day on Kaggle.

#### Day 1 - Prompting
I worked through example prompts and techniques using the Python SDK and AI Studio, with further insights provided in the Prompting whitepaper. I also explored various apps built with the Gemini family of models for inspiration.

#### Day 1 - Evaluation and Structured Output
I focused on evaluating the performance of LLMs in real-world tasks, specifically learning how to assess and measure their outputs effectively.

#### Day 2 - Classifying Embeddings with Keras
I learned how to classify embeddings using Keras and the Gemini API, enhancing my understanding of model classification.

#### Day 2 - Embeddings and Similarity Scores
I explored embeddings and similarity scores with the Gemini API, using its embedding endpoint to analyze similarity scores between different data points.

#### Day 2 - Document Q&A with RAG
I learned how to implement Retrieval Augmented Generation (RAG) with Chroma to enhance Document Q&A. I created a vector database with the Gemini API to retrieve relevant information and generate tailored answers, overcoming LLM limitations by processing external, unseen data and using Chroma for efficient document searching and embedding.

#### Day 3 - Function Calling with the Gemini API
I explored function calling with the Gemini API, learning to build a chat interface over a local database using the API’s automatic function-calling feature. Though the example was a simple project without real-world safety constraints, it demonstrated how to integrate AI chat capabilities into existing applications.

#### Day 4 - Fine-Tuning a Custom Model
I delved into fine-tuning a custom model using the Gemini API, focusing on training the model to classify newsgroup posts into their respective categories. The fine-tuning process enabled me to tailor the model for specific tasks like entity extraction, summarization, and creative generation. I also discovered that AI Studio supports creating and monitoring tuned models via a web UI using data from Google Sheets, Drive, or my own files.


## 📝 Natural Language Processing Methods

