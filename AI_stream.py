import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load model, scaler, and encoder
model = pickle.load(open('student_grade_predictor.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# App title and description
st.title("🎓 Student Grade Predictor")
st.write("""
This app predicts the **final grade of a student** based on personal and academic attributes.  
Fill in the form below to get your grade prediction.
""")

# --- USER INPUT FORM ---
st.subheader("🔍 Enter Student Information")

student_age = st.selectbox("Age Group", ['18', '19-22', '23-27'])
sex = st.selectbox("Gender", ['Male', 'Female'])
high_school_type = st.selectbox("High School Type", ['State', 'Private', 'Other'])
scholarship = st.slider("Scholarship Percentage", 0, 100)
additional_work = st.selectbox("Additional Work", ['Yes', 'No'])
sports_activity = st.selectbox("Sports Activity", ['Yes', 'No'])
transportation = st.selectbox("Transportation", ['Private', 'Bus'])
weekly_study_hours = st.slider("Weekly Study Hours", 0, 40)
attendance = st.selectbox("Attendance", ['Always', 'Never', 'Sometimes'])
reading = st.selectbox("Reading Habit", ['Yes', 'No'])
notes = st.selectbox("Takes Notes", ['Yes', 'No'])
listening_in_class = st.selectbox("Listens in Class", ['Yes', 'No'])
project_work = st.selectbox("Does Project Work", ['Yes', 'No'])

# Create input DataFrame
user_input = pd.DataFrame([{
    'Student_Age': student_age,
    'Sex': sex,
    'High_School_Type': high_school_type,
    'Scholarship': scholarship / 100,
    'Additional_Work': additional_work,
    'Sports_activity': sports_activity,
    'Transportation': transportation,
    'Weekly_Study_Hours': weekly_study_hours,
    'Attendance': attendance,
    'Reading': reading,
    'Notes': notes,
    'Listening_in_Class': listening_in_class,
    'Project_work': project_work
}])

# --- PREDICT BUTTON ---
if st.button("🎯 Predict Grade"):
    try:
        # Separate numeric and categorical columns
        numeric_cols = ['Scholarship', 'Weekly_Study_Hours']
        categorical_cols = [col for col in user_input.columns if col not in numeric_cols]

        # Scale numeric features
        scaled_numeric = scaler.transform(user_input[numeric_cols])

        # Encode categorical features
        encoded_categorical = encoder.transform(user_input[categorical_cols])

        # Combine all features
        import numpy as np
        final_input = np.hstack((scaled_numeric, encoded_categorical))

        # Predict
        prediction = model.predict(final_input)

        # Display result
        st.success(f"📘 **Predicted Grade**: {prediction[0]}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
