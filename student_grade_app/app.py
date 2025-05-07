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
