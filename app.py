import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import statsmodels.api as sm

# Load the model and scaler
model_filename = 'bestgrad.py'
scaler_filename = 'minmax.py'

with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open(scaler_filename, 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Title of the app
st.title("Graduate Admission Predictor")

# User inputs
GRE_Score = st.number_input("GRE Score", min_value=0, max_value=340, value=300)
TOEFL_Score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
University_Rating = st.number_input("University Rating", min_value=1, max_value=5, value=3)
SOP = st.number_input("SOP (Statement of Purpose Strength)", min_value=1.0, max_value=5.0, value=3.0)
LOR = st.number_input("LOR (Letter of Recommendation Strength)", min_value=1.0, max_value=5.0, value=3.0)
CGPA = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0)
Research = st.selectbox("Research Experience", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Predict button
if st.button("Predict"):
    # Create a dataframe with user inputs
    input_data = pd.DataFrame({
        'GRE Score': [GRE_Score],
        'TOEFL Score': [TOEFL_Score],
        'University Rating': [University_Rating],
        'SOP': [SOP],
        'LOR': [LOR],
        'CGPA': [CGPA],
        'Research': [Research]
    })

    # Ensure columns are in the correct order and format
    input_data = input_data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
    input_data = input_data.astype({'GRE Score': 'float64', 'TOEFL Score': 'float64', 'University Rating': 'float64',
                                    'SOP': 'float64', 'LOR': 'float64', 'CGPA': 'float64', 'Research': 'float64'})

    # Scale the input data
    scaled_input = loaded_scaler.transform(input_data)

    # Make prediction
    prediction = loaded_model.predict(scaled_input)

    # Display the result
    st.write(f"The predicted chance of admission is: {prediction[0]:.2f}")