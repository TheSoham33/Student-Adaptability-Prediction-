# Student Adaptability Prediction Web App

[![Watch](https://img.shields.io/github/watchers/TheSoham33/Student-Adaptability-Prediction-?style=social)](https://github.com/TheSoham33/Student-Adaptability-Prediction-/watchers)
[![Fork](https://img.shields.io/github/forks/TheSoham33/Student-Adaptability-Prediction-?style=social)](https://github.com/TheSoham33/Student-Adaptability-Prediction-/network/members)
[![Stars](https://img.shields.io/github/stars/TheSoham33/Student-Adaptability-Prediction-?style=social)](https://github.com/TheSoham33/Student-Adaptability-Prediction-/stargazers)
[![Python](https://img.shields.io/badge/Python-100.0%25-blue)](https://www.python.org/)

## Overview

The **Student Adaptability Prediction Web App** is a Streamlit application designed to forecast a student's likelihood of adapting to online learning environments. By taking in student-specific information through a user-friendly form, the app employs a pre-trained machine learning model to categorize their adaptability level as either **'High'** or **'Low'**.

This application serves as a valuable tool for:

*   **Educators**: To proactively identify students who may require additional support in online learning settings.
*   **Educational Institutions**:  To understand and address the factors influencing student adaptability, optimizing online learning programs.
*   **Policymakers**: To gain insights into the broader trends affecting online learning adaptability and inform educational strategies.

## Features

*   **Intuitive Web Interface:**  Leveraging Streamlit, the application provides a simple and accessible web-based interface for all users.
*   **Detailed Input Form:** A sidebar form collects essential student data through interactive fields:
    *   **Demographic Factors:** Gender, Age, Education Level (School, College, University)
    *   **Institutional & Technological Context:** Institution Type (Government, Non-Government), IT Student Status (Yes/No)
    *   **Environmental & Socio-economic Conditions:** Location (City/Non-City), Load Shedding Level (High/Low), Financial Condition (Rich, Mid, Poor)
    *   **Digital Access & Proficiency:** Internet Type (Mobile Data, Wifi), Network Type (4G, 3G), Device Used (Mobile, Computer), Self-Learning Management System (LMS) Usage (Yes/No)
    *   **Learning Engagement:** Class Duration (in hours)
*   **Instant Adaptability Prediction:**  Upon submission of student details, the app instantly predicts the student's adaptability level using the loaded machine learning model.
*   **Clear and Direct Results:**  The predicted adaptability level (**High** or **Low**) is displayed prominently, making the assessment results readily understandable.
*   **Error Handling & Model Integrity:**  Implements robust error handling to manage scenarios like missing model files or prediction errors, ensuring a stable and informative user experience.

## Tech Stack

This project is powered by:

*   **Python:** The core language for application logic and machine learning.
*   **Streamlit:**  For creating the interactive web application interface.
*   **Pandas:**  Used for efficient data manipulation and structuring of input data.
*   **NumPy:**  For numerical computations and data handling within the application.
*   **Scikit-learn (sklearn):**  The machine learning library utilized for training the adaptability