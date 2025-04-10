
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 🎯 Grade Mapping Functions
# ----------------------------
def map_grade_to_numeric_letter(score):
    if score < 40:
        return 0.0, 'F'
    elif 40 <= score < 45:
        return 1.0, 'D'
    elif 45 <= score < 50:
        return 1.75, 'C-'
    elif 50 <= score < 60:
        return 2.0, 'C'
    elif 60 <= score < 65:
        return 2.5, 'C+'
    elif 65 <= score < 70:
        return 2.75, 'B-'
    elif 70 <= score < 75:
        return 3.0, 'B'
    elif 75 <= score < 80:
        return 3.5, 'B+'
    elif 80 <= score < 85:
        return 3.75, 'A-'
    elif 85 <= score < 90:
        return 4.0, 'A'
    else:
        return 4.1, 'A+'

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

# ----------------------------
# 📥 Load and Preprocess Data
# ----------------------------
data = pd.read_csv('Students _Performance _Prediction.csv')

# Encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Map grades
data[['Numeric_Grade', 'Letter_Grade']] = data['Grade'].apply(
    lambda x: pd.Series(map_grade_to_numeric_letter(x))
)

feature_columns = ['Student_Age', 'Sex', 'High_School_Type', 'Scholarship',
                   'Additional_Work', 'Sports_activity', 'Transportation',
                   'Weekly_Study_Hours', 'Attendance', 'Reading',
                   'Notes', 'Listening_in_Class', 'Project_work']

X = data[feature_columns]
y = data['Numeric_Grade']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=34)
model = RandomForestRegressor(random_state=34)
model.fit(X_train, y_train)

# ----------------------------
# 🌐 Streamlit UI for Batch Prediction
# ----------------------------
st.title("\U0001F4DA Student Performance Predictor")
st.markdown("Upload a CSV file of student details to predict grades")

# Example format download
with st.expander("\U0001F4C4 See required CSV format"):
    st.markdown("""
    **Required columns in the CSV file:**

    - Student_Age  
    - Sex  
    - High_School_Type  
    - Scholarship  
    - Additional_Work  
    - Sports_activity  
    - Transportation  
    - Weekly_Study_Hours  
    - Attendance  
    - Reading  
    - Notes  
    - Listening_in_Class  
    - Project_work  
    """)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Encode categorical inputs
        for col in input_df.columns:
            if col in label_encoders:
                encoder = label_encoders[col]
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except ValueError:
                    input_df[col] = input_df[col].apply(
                        lambda x: encoder.transform([encoder.classes_[0]])[0]
                        if x not in encoder.classes_ else encoder.transform([x])[0]
                    )

        input_df = input_df[X.columns]

        # Predict grades
        predicted_numeric = model.predict(input_df)
        predicted_letters = [convert_numeric_to_letter(score) for score in predicted_numeric]

        # Show results
        result_df = pd.read_csv(uploaded_file)
        result_df['Predicted Numeric Grade'] = predicted_numeric
        result_df['Predicted Letter Grade'] = predicted_letters

        st.success("\u2705 Predictions complete!")
        st.dataframe(result_df)

        # Downloadable results
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("\U0001F4E5 Download Results", csv, file_name="predicted_grades.csv", mime='text/csv')

    except Exception as e:
        st.error(f"\u274C Error: {e}")
