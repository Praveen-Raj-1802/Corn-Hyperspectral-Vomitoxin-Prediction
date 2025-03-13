# Hyperspectral Vomitoxin Prediction

## Overview
This repository contains code and models for predicting vomitoxin levels in corn using hyperspectral imaging. 
The dataset is preprocessed, and different machine learning models, including Random Forest, CNN, and XGBoost, are evaluated for their predictive performance.

## Repository Structure
📂 Corn-Hyperspectral-Vomitoxin-Prediction │-- 
📜 Corn-Hyperspectral-Vomitoxin-Prediction.ipynb # Jupyter Notebook for model training & evaluation │-- 
📜 app.py # Streamlit web application for prediction │-- 📜 pca_transformer.pkl # Saved PCA transformer model │-- 
📜 random_forest_model.pkl # Trained Random Forest model │-- 📜 scaler.pkl # Standard scaler for feature scaling |-- 
📜 images/ # Folder for screenshots |-- 📜 README.md # Project documentation 

## Installation
Clone this repository:
```bash
git clone https://github.com/Praveen-Raj-1802/Corn-Hyperspectral-Vomitoxin-Prediction.git
cd Corn-Hyperspectral-Vomitoxin-Prediction
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
streamlit run app.py
```

## Model Performance
The Random Forest model achieved the best results with:
MAE: 2106.86
RMSE: 5099.63
R²: 0.91

## Future Improvements
1. Hyperparameter tuning for better model performance.
2. Ensemble models for improved accuracy.
3. Feature extraction enhancements before PCA transformation.

## Streamlit Web App

Below is a preview of the web application:

![Streamlit App Screenshot](images/streamlit_app1.png)

![Streamlit App Screenshot](images/streamlit_app2.png)



