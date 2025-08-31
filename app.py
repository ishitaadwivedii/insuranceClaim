import streamlit as st
import joblib
from utils import preprocess_input

# Load model
model_path = r"models/rf_best_model_rs_selected.joblib"
model = joblib.load(model_path)

# App title
st.title("🏥 Healthcare Insurance Claim Prediction")

st.write("Enter the details below to estimate the insurance claim amount.")

# Input fields
smoker = st.selectbox("Smoker", ["Yes", "No"])
bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
bloodpressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
age_group = st.selectbox("Age Group", ["Adult", "Middle-Aged", "Senior"])
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
hereditary_heart = st.selectbox("Hereditary Heart Disease", ["Yes", "No"])
is_male = st.selectbox("Gender", ["Male", "Female"])
hereditary_alzheimer = st.selectbox("Hereditary Alzheimer", ["Yes", "No"])

# Predict button
if st.button("Predict Claim Amount"):
    # Preprocess input
    input_df = preprocess_input(smoker, bmi_category, bloodpressure, age_group, no_of_dependents,
                                diabetes, hereditary_heart, is_male, hereditary_alzheimer)
    
    # Prediction
    prediction = model.predict(input_df)[0]
    
    st.success(f"Estimated Claim Amount: ${prediction:,.2f}")
