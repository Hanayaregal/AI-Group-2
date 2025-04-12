import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Grade mapping functions
def map_grade_letter_to_numeric(letter):
    mapping = {
        "F": 0.0, "D": 1.0, "C-": 1.75, "C": 2.0, "C+": 2.5,
        "B-": 2.75, "B": 3.0, "B+": 3.5, "A-": 3.75, "A": 4.0, "A+": 4.1, "AA": 4.0
    }
    return mapping.get(str(letter).upper(), 0.0)

def convert_numeric_to_letter(grade):
    if grade < 1.0:
        return "F"
    elif grade < 1.75:
        return "D"
    elif grade < 2.0:
        return "C-"
    elif grade < 2.5:
        return "C"
    elif grade < 2.75:
        return "C+"
    elif grade < 3.0:
        return "B-"
    elif grade < 3.5:
        return "B"
    elif grade < 3.75:
        return "B+"
    elif grade < 4.0:
        return "A-"
    elif grade <= 4.0:
        return "A"
    else:
        return "A+"

# Age conversion helper
def convert_age(age_value):
    if '-' in str(age_value):
        parts = age_value.split('-')
        return (float(parts[0]) + float(parts[1])) / 2
    try:
        return float(age_value)
    except:
        return None

# Load and preprocess data
data = pd.read_csv("Students _Performance _Prediction.csv")
data['Scholarship'] = data['Scholarship'].str.replace('%', '', regex=False).astype(float)
data['Student_Age'] = data['Student_Age'].apply(convert_age)
data['Grade'] = data['Grade'].apply(map_grade_letter_to_numeric)

# Encode categorical columns
categorical_columns = data.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
features = ['Student_Age', 'Sex', 'High_School_Type', 'Scholarship', 'Additional_Work',
            'Sports_activity', 'Transportation', 'Weekly_Study_Hours', 'Attendance',
            'Reading', 'Notes', 'Listening_in_Class', 'Project_work']
X = data[features]
y = data['Grade']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Streamlit interface
st.title("🎓 Student Grade Prediction App")

# Dropdown-based input form
with st.form("prediction_form"):
    student_age = st.selectbox("Student Age", ['18', '19-22', '23-27'])
    sex = st.selectbox("Sex", ['Male', 'Female'])
    high_school_type = st.selectbox("High School Type", ['Public', 'Private'])
    scholarship = st.selectbox("Scholarship (%)", ['50', '75', '100'])
    additional_work = st.selectbox("Additional Work", ['Yes', 'No'])
    sports_activity = st.selectbox("Sports Activity", ['Yes', 'No'])
    transportation = st.selectbox("Transportation", ['Private', 'Bus'])
    weekly_study_hours = st.slider("Weekly Study Hours", 0.0, 40.0, 10.0)
    attendance = st.slider("Attendance Score (1-3)", 1.0, 3.0, 2.0)
    reading = st.selectbox("Reading", ['Yes', 'No'])
    notes = st.selectbox("Takes Notes", ['1', '0'])
    listening_in_class = st.selectbox("Listens in Class", ['1', '0'])
    project_work = st.selectbox("Project Work", ['1', '0'])

    submitted = st.form_submit_button("Predict Grade")

# Perform prediction
if submitted:
    input_dict = {
        'Student_Age': convert_age(student_age),
        'Sex': label_encoders['Sex'].transform([sex])[0],
        'High_School_Type': label_encoders['High_School_Type'].transform([high_school_type])[0],
        'Scholarship': float(scholarship),
        'Additional_Work': label_encoders['Additional_Work'].transform([additional_work])[0],
        'Sports_activity': label_encoders['Sports_activity'].transform([sports_activity])[0],
        'Transportation': label_encoders['Transportation'].transform([transportation])[0],
        'Weekly_Study_Hours': weekly_study_hours,
        'Attendance': attendance,
        'Reading': label_encoders['Reading'].transform([reading])[0],
        'Notes': float(notes),
        'Listening_in_Class': float(listening_in_class),
        'Project_work': float(project_work)
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    letter_grade = convert_numeric_to_letter(prediction)

    st.success(f"Predicted Grade: {prediction:.2f} ➝ {letter_grade}")