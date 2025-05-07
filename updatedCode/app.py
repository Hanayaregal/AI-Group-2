import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load the trained model, scaler, and encoder
with open('student_grade_predictor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Title of the App
st.title("Student Grade Prediction App")

# Upload sample input file or allow manual input
st.subheader("Upload your data or fill in the details:")

# File uploader for sample input
uploaded_file = st.file_uploader("Students _Performance _Prediction.csv", type=['csv'])

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    st.write(df.head())  # Show the first few rows of the dataset for confirmation

    # Preprocessing the data before prediction
    def preprocess_data(data):
        # Preprocess numerical and categorical features using the trained scaler and encoder
        numeric_features = ['Scholarship', 'Weekly_Study_Hours', 'Age']
        categorical_features = ['Student_Age', 'Sex', 'High_School_Type', 'Additional_Work', 
                                'Sports_activity', 'Transportation', 'Attendance', 'Reading', 
                                'Notes', 'Listening_in_Class', 'Project_work']

        # Apply scaling and encoding
        data[numeric_features] = scaler.transform(data[numeric_features])
        encoded_data = encoder.transform(data[categorical_features])
        
        return np.hstack([data[numeric_features], encoded_data.toarray()])

    # Preprocess the uploaded file
    preprocessed_data = preprocess_data(df)

    # Predict the grades
    predictions = model.predict(preprocessed_data)

    # Display the predictions
    df['Predicted_Grade'] = predictions
    st.write(df[['Student_ID', 'Predicted_Grade']])

# Manual input form for individual student prediction
st.subheader("Or manually enter student details:")

student_id = st.text_input("Student ID")
scholarship = st.selectbox("Scholarship", options=[0, 0.25, 0.5, 0.75, 1])
study_hours = st.number_input("Weekly Study Hours", min_value=0)
age = st.number_input("Age", min_value=0)
attendance = st.selectbox("Attendance", options=['Always', 'Sometimes', 'Never'])
# Add more fields as per your dataset (Sex, High_School_Type, etc.)

# Create a dictionary with manual input
manual_input = {
    'Student_ID': [student_id],
    'Scholarship': [scholarship],
    'Weekly_Study_Hours': [study_hours],
    'Age': [age],
    'Attendance': [attendance],
    # Add the rest of the features
}

input_df = pd.DataFrame(manual_input)

# Preprocess manual input
preprocessed_manual_data = preprocess_data(input_df)

# Prediction
if st.button('Predict Grade'):
    manual_prediction = model.predict(preprocessed_manual_data)
    st.write(f"Predicted Grade: {manual_prediction[0]}")

#st.image('mau_blue.png', width=150)



"""
# Student Grade Prediction App using RandomForestClassifier and Streamlit

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import streamlit as st
import pickle

# Load the dataset
df = pd.read_csv('Students _Performance _Prediction.csv')

# Data Cleaning and Preprocessing
# Handle '6' in Notes column
df['Notes'] = df['Notes'].replace('6', 'No')

# Convert Scholarship to numerical
df['Scholarship'] = df['Scholarship'].str.rstrip('%').replace('None', '0').astype(float) / 100

# Convert Student_Age to categorical
df['Student_Age'] = df['Student_Age'].apply(lambda x: '18' if x == '18' else '19-22' if x == '19-22' else '23-27')

# Define features and target
X = df.drop(['Grade', 'Student_ID'], axis=1)
y = df['Grade']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline for numeric and categorical data
numeric_features = ['Scholarship', 'Weekly_Study_Hours']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Student_Age', 'Sex', 'High_School_Type', 'Additional_Work', 
                        'Sports_activity', 'Transportation', 'Attendance', 'Reading', 
                        'Notes', 'Listening_in_Class', 'Project_work']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer to apply different transformations on numeric and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model pipeline that includes preprocessing and RandomForestClassifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the trained model using pickle
with open('student_grade_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)

# Streamlit function to input data and predict the grade
def predict_grade():
    st.title("Student Grade Prediction")

    st.write("""
    This app predicts the grade of students based on their study habits, 
    attendance, scholarship status, and other relevant factors.
    """)

    # Input fields for student details
    features = {
        'Student_Age': st.selectbox("Age group", ['18', '19-22', '23-27']),
        'Sex': st.radio("Gender", ['Male', 'Female']),
        'High_School_Type': st.selectbox("High School Type", ['State', 'Private', 'Other']),
        'Scholarship': st.slider("Scholarship percentage (0-100)", 0, 100, step=1),
        'Additional_Work': st.selectbox("Additional Work", ['Yes', 'No']),
        'Sports_activity': st.selectbox("Sports Activity", ['Yes', 'No']),
        'Transportation': st.selectbox("Transportation", ['Private', 'Bus']),
        'Weekly_Study_Hours': st.slider("Weekly Study Hours", 0, 40, step=1, value=10),
        'Attendance': st.selectbox("Attendance", ['Always', 'Never', 'Sometimes']),
        'Reading': st.selectbox("Reading Habit", ['Yes', 'No']),
        'Notes': st.selectbox("Takes Notes", ['Yes', 'No']),
        'Listening_in_Class': st.selectbox("Listens in Class", ['Yes', 'No']),
        'Project_work': st.selectbox("Does Project Work", ['Yes', 'No'])
    }

    # Create a DataFrame from the user input
    input_df = pd.DataFrame([features])

    # Load the saved model for prediction
    model = pickle.load(open('student_grade_predictor.pkl', 'rb'))

    # Predict the grade
    prediction = model.predict(input_df)
    
    st.write(f"Predicted Grade: {prediction[0]}")

# Run the Streamlit app
if __name__ == "__main__":
    predict_grade()


"""