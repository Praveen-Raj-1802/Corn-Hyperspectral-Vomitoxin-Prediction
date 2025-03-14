# Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Hyperspectral Data Prediction using Random Forest")

uploaded_file = st.file_uploader("Upload a CSV file with spectral data", type=["csv"])

# Load pre-trained model and transformers
with open('random_forest_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)
with open('pca_transformer.pkl', 'rb') as pca_file:
    pca = pickle.load(pca_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df_uploaded.head())

    # Preprocess uploaded data
    spectral_data_uploaded = df_uploaded.iloc[:, 1:]  # Assuming first column is ID
    spectral_data_scaled = scaler.transform(spectral_data_uploaded)
    spectral_data_pca = pca.transform(spectral_data_scaled)
    
    # Make predictions
    predictions = rf_model.predict(spectral_data_pca)
    df_uploaded['Predicted_vomitoxin_ppb'] = predictions
    
    st.write("Predictions:")
    st.dataframe(df_uploaded[['hsi_id', 'Predicted_vomitoxin_ppb']])
     
    # Download results
    csv = df_uploaded.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")