import numpy as np
import pandas as pd
import pickle
import os
import streamlit as st
from streamlit_lottie import st_lottie


from streamlit_lottie import st_lottie

def load_lottie_url(url):
    import requests
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a Lottie animation for High Adaptability Level
lottie_high = load_lottie_url("https://lottie.host/4c96ed6e-342f-4d23-9c15-75e57c18781d/Gv6SbnUw2P.json")  # Example JSON animation
lottie_moderate = load_lottie_url("https://lottie.host/2d4f6a74-ffb8-4e93-9cf6-3a50e5cc5fdd/kWADlJFOgE.json")  # Moderate adaptability (Motivated)
lottie_low = load_lottie_url("https://lottie.host/bcd3c79d-fc90-40f0-8ae8-29a9e5923b59/J7HkjgV9hb.json")  # Low adaptability (Trying to improve)

# Load the trained model
with open("model.pkl", "rb") as model_file:
    xgb_model = pickle.load(model_file)

def get_user_input():
    st.sidebar.title("User Input")
    
    gender = st.sidebar.selectbox("Select Gender", ["Boy", "Girl"])
    age = st.sidebar.slider("Enter Age", 5, 25, 18)
    education_level = st.sidebar.selectbox("Education Level", ["School", "College", "University"])
    institution_type = st.sidebar.selectbox("Institution Type", ["Government", "Non Government"])
    it_student = st.sidebar.radio("Are you an IT student?", ["Yes", "No"])
    location = st.sidebar.radio("Do you live in a city?", ["Yes", "No"])
    load_shedding = st.sidebar.radio("Load-shedding Level", ["High", "Low"])
    financial_condition = st.sidebar.selectbox("Financial Condition", ["Rich", "Mid", "Poor"])
    internet_type = st.sidebar.radio("Internet Type", ["Mobile Data", "Wifi"])
    network_type = st.sidebar.selectbox("Network Type", ["4G", "3G", "2G"])
    class_duration = st.sidebar.slider("Class Duration (hours)", 1, 6, 3)
    self_lms = st.sidebar.radio("Do you use Self LMS?", ["Yes", "No"])
    device = st.sidebar.selectbox("Device Used", ["Mobile", "Computer", "Tab"])
    
    data = {
        "Gender": gender,
        "Age": age,
        "Education Level": education_level,
        "Institution Type": institution_type,
        "IT Student": it_student,
        "Location": location,
        "Load-shedding": load_shedding,
        "Financial Condition": financial_condition,
        "Internet Type": internet_type,
        "Network Type": network_type,
        "Class Duration": class_duration,
        "Self Lms": self_lms,
        "Device": device
    }
    
    return data

def encode_features(data):
    age_categories = {
        "Age_1_5": 1 if 1 <= data["Age"] <= 5 else 0,
        "Age_6_10": 1 if 6 <= data["Age"] <= 10 else 0,
        "Age_11_15": 1 if 11 <= data["Age"] <= 15 else 0,
        "Age_16_20": 1 if 16 <= data["Age"] <= 20 else 0,
        "Age_21_25": 1 if 21 <= data["Age"] <= 25 else 0,
        "Age_26_30": 1 if 26 <= data["Age"] <= 30 else 0
    }
    
    class_duration_categories = {
        "Class_Duration_0": 1 if data["Class Duration"] == 0 else 0,
        "Class_Duration_1_3": 1 if 1 <= data["Class Duration"] <= 3 else 0,
        "Class_Duration_4_6": 1 if 4 <= data["Class Duration"] <= 6 else 0
    }
    
    input_data = {
        "Gender_Girl": 1 if data["Gender"] == "Girl" else 0,
        "Institution_Type_Non_Government": 1 if data["Institution Type"] == "Non Government" else 0,
        "IT_Student_Yes": 1 if data["IT Student"] == "Yes" else 0,
        "Location_Yes": 1 if data["Location"] == "Yes" else 0,
        "Load_shedding_Low": 1 if data["Load-shedding"] == "Low" else 0,
        "Internet_Type_Wifi": 1 if data["Internet Type"] == "Wifi" else 0,
        "Self_LMS_Yes": 1 if data["Self Lms"] == "Yes" else 0,
        **age_categories,
        "Education_Level_College": 1 if data["Education Level"] == "College" else 0,
        "Education_Level_School": 1 if data["Education Level"] == "School" else 0,
        "Education_Level_University": 1 if data["Education Level"] == "University" else 0,
        "Financial_Condition_Rich": 1 if data["Financial Condition"] == "Rich" else 0,
        "Financial_Condition_Mid": 1 if data["Financial Condition"] == "Mid" else 0,
        "Financial_Condition_Poor": 1 if data["Financial Condition"] == "Poor" else 0,
        "Network_Type_2G": 1 if data["Network Type"] == "2G" else 0,
        "Network_Type_3G": 1 if data["Network Type"] == "3G" else 0,
        "Network_Type_4G": 1 if data["Network Type"] == "4G" else 0,
        **class_duration_categories,
        "Device_Computer": 1 if data["Device"] == "Computer" else 0,
        "Device_Mobile": 1 if data["Device"] == "Mobile" else 0,
        "Device_Tab": 1 if data["Device"] == "Tab" else 0
    }
    return input_data

# Streamlit UI
st.title("Adaptability Level Prediction")
data = get_user_input()

if st.sidebar.button("Predict Adaptability Level"):
    encoded_data = encode_features(data)
    
    # Predict adaptability level
    p = xgb_model.predict([list(encoded_data.values())])

    # Convert prediction to one-hot encoding using argmax
    p_s = np.zeros(3, dtype=int)  # Ensure output is always in [0, 1, 0] format
    p_s[np.argmax(p)] = 1  # Assign correct class index

    # Display result
    if np.array_equal(p_s, [0, 1, 0]):
        st.success("Adaptability Level: Moderate")
        st.success("🎉 Congratulations! You have **Moderate Adaptability Level**!")
        st.image("https://media.giphy.com/media/NJvoYP1O8AePdUUgIP/giphy.gif?cid=790b76115vpcz8sas53vhtxz6ez8xzeguja2c0ihfiwrfpj4&ep=v1_gifs_search&rid=giphy.gif&ct=g", width=300)
        st.write("You're doing great! Keep pushing forward and improving. 🚀")
    elif np.array_equal(p_s, [0, 0, 1]):
        st.success("🏆 Amazing! You have **High Adaptability Level**! 🌟")
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOWl6cHZjbWl0dTlrcWhtcTVseTg5cWh2ZHBobjF3ZmNueDExdDVpeSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/xT5LMHxhOfscxPfIfm/giphy.gif", width=300)
        st.write("You are highly adaptable! Keep up the excellent work. 💪")
    elif np.array_equal(p_s, [1, 0, 0]):
        st.success("Adaptability Level: Low")
        st.warning("🛠️ Don't Worry! You have **Low Adaptability Level**, but you can improve! 📈")
        st.image("https://media.giphy.com/media/3o6MbmqpP08p8hinRe/giphy.gif?cid=790b76119pgfofkle4xai0tngwbx1iz2d96knwy77l09b72d&ep=v1_gifs_search&rid=giphy.gif&ct=g", width=300)
        st.write("Stay motivated and take small steps to boost your adaptability. You got this! 💡")
    else:
        st.error("Error: Unable to determine adaptability level.")
