import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open('student_grade_predictor.pkl', 'rb'))

# Title of the app
st.title("Student Grade Predictor")

# Description of the app
st.write("""
This app predicts the final grade of a student based on various attributes such as study hours, attendance, and extracurricular activities. 
Please fill out the information below, and the model will predict the grade.
""")

# User input section
st.subheader("Student Information")

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

# Create a dictionary for the user input
user_input = {
    'Student_Age': student_age,
    'Sex': sex,
    'High_School_Type': high_school_type,
    'Scholarship': scholarship / 100,  # Convert to proportion
    'Additional_Work': additional_work,
    'Sports_activity': sports_activity,
    'Transportation': transportation,
    'Weekly_Study_Hours': weekly_study_hours,
    'Attendance': attendance,
    'Reading': reading,
    'Notes': notes,
    'Listening_in_Class': listening_in_class,
    'Project_work': project_work
}

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Prediction Button
if st.button("Predict Grade"):
    # Preprocess input
    user_input_processed = preprocess_user_input(user_input_df)  # Define your preprocessing function
    prediction = model.predict(user_input_processed)
    st.write(f"**Predicted Grade**: {prediction[0]}")
    
    # Optional: Show additional info (e.g., feature importance or model accuracy)
    # Feature importance or model accuracy can be added here

# Function to preprocess user input (this should match the preprocessing done during training)
def preprocess_user_input(input_df):
    # Add necessary preprocessing steps here (e.g., scaling, encoding)
    # Example: scaling the numerical features
    scaler = StandardScaler()
    input_df[['Scholarship', 'Weekly_Study_Hours']] = scaler.fit_transform(input_df[['Scholarship', 'Weekly_Study_Hours']])
    # Encoding categorical features can be added if necessary
    return input_df
