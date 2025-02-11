import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# Load the trained model
model_path = "model.pkl"
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        print("Pickle file loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
else:
    st.error("⚠️ Model file not found! Please train the model first.")
    model = None

# Define input fields
st.title("🎓 Student Adaptability Prediction")
st.sidebar.header("📌 Enter Student Details")

gender = st.sidebar.selectbox("Select Gender", ["Boy", "Girl"])
age = st.sidebar.number_input("Enter Age", min_value=10, max_value=100, value=20, step=1)
education_level = st.sidebar.selectbox("Education Level", ["School", "College", "University"])
institution_type = st.sidebar.selectbox("Institution Type", ["Government", "Non Government"])
it_student = st.sidebar.radio("Are you an IT student?", ["Yes", "No"])
location = st.sidebar.radio("Do you live in a city?", ["Yes", "No"])
load_shedding = st.sidebar.radio("Load-shedding Level", ["High", "Low"])
financial_condition = st.sidebar.selectbox("Financial Condition", ["Rich", "Mid", "Poor"])
internet_type = st.sidebar.radio("Internet Type", ["Mobile Data", "Wifi"])
network_type = st.sidebar.radio("Network Type", ["4G", "3G"])
class_duration = st.sidebar.slider("Class Duration (in hours)", min_value=1, max_value=10, value=4, step=1)
self_lms = st.sidebar.radio("Do you use Self LMS?", ["Yes", "No"])
device = st.sidebar.radio("Device Used", ["Mobile", "Computer"])

# Convert inputs into the required format
input_data = {
    "Gender_Girl": 1 if gender == "Girl" else 0,
    "Age": age,
    "Education_Level_School": 1 if education_level == "School" else 0,
    "Education_Level_College": 1 if education_level == "College" else 0,
    "Education_Level_University": 1 if education_level == "University" else 0,
    "Institution_Type_Non_Government": 1 if institution_type == "Non Government" else 0,
    "IT_Student_Yes": 1 if it_student == "Yes" else 0,
    "Location_Yes": 1 if location == "Yes" else 0,
    "Load_Shedding_High": 1 if load_shedding == "High" else 0,
    "Financial_Condition_Rich": 1 if financial_condition == "Rich" else 0,
    "Financial_Condition_Mid": 1 if financial_condition == "Mid" else 0,
    "Financial_Condition_Poor": 1 if financial_condition == "Poor" else 0,
    "Internet_Type_Wifi": 1 if internet_type == "Wifi" else 0,
    "Network_Type_4G": 1 if network_type == "4G" else 0,
    "Class_Duration": class_duration,
    "Self_LMS_Yes": 1 if self_lms == "Yes" else 0,
    "Device_Computer": 1 if device == "Computer" else 0
}

# Convert input dictionary to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure column order matches model's expected input
if model is not None:
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
    else:
        st.error("⚠️ Model does not have feature names. Make sure it is properly trained.")

# Predict adaptability level
if st.sidebar.button("Predict Adaptability") and model is not None:
    try:
        prediction = model.predict(input_df)
        adaptability_level = "High" if int(prediction.flatten()[0]) == 1 else "Low"
        st.success(f"🎯 **Predicted Adaptability Level: {adaptability_level}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
