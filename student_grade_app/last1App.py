import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Cache the model to load only once
@st.cache_resource
def train_model():
    # Load and preprocess data
    df = pd.read_csv('Students _Performance _Prediction.csv')
    
    # Data cleaning
    df['Notes'] = df['Notes'].replace('6', 'No')
    df['Scholarship'] = df['Scholarship'].str.rstrip('%').replace('None', '0').astype(float) / 100
    df['Student_Age'] = df['Student_Age'].apply(lambda x: '18' if x == '18' else '19-22' if x == '19-22' else '23-27')
    
    X = df.drop(['Grade', 'Student_ID'], axis=1)
    y = df['Grade']
    
    # Preprocessing pipeline
    numeric_features = ['Scholarship', 'Weekly_Study_Hours']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['Student_Age', 'Sex', 'High_School_Type', 'Additional_Work', 
                            'Sports_activity', 'Transportation', 'Attendance', 'Reading', 
                            'Notes', 'Listening_in_Class', 'Project_work']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create and train model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X, y)
    return model

# Create Streamlit app
def main():
    st.title("Student Performance Predictor")
    st.write("This app predicts student grades based on academic and demographic factors")
    
    # Load model
    model = train_model()
    
    # Create input form
    with st.form("student_info"):
        st.header("Student Information")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.selectbox("Age Group", ["18", "19-22", "23-27"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            school_type = st.selectbox("High School Type", ["State", "Private", "Other"])
            scholarship = st.number_input("Scholarship (%)", 0, 100, 50)
            additional_work = st.selectbox("Additional Work", ["Yes", "No"])
            
        with col2:
            sports = st.selectbox("Sports Activity", ["Yes", "No"])
            transport = st.selectbox("Transportation", ["Private", "Bus"])
            study_hours = st.number_input("Weekly Study Hours", 0, 168, 0)
            attendance = st.selectbox("Attendance", ["Always", "Sometimes", "Never"])
            reading = st.selectbox("Reading Habit", ["Yes", "No"])
        
        notes = st.selectbox("Takes Notes", ["Yes", "No"])
        listening = st.selectbox("Listens in Class", ["Yes", "No"])
        project_work = st.selectbox("Project Work", ["Yes", "No"])
        
        submitted = st.form_submit_button("Predict Grade")
        
    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame([{
            'Student_Age': age,
            'Sex': gender,
            'High_School_Type': school_type,
            'Scholarship': scholarship / 100,
            'Additional_Work': additional_work,
            'Sports_activity': sports,
            'Transportation': transport,
            'Weekly_Study_Hours': study_hours,
            'Attendance': attendance,
            'Reading': reading,
            'Notes': notes,
            'Listening_in_Class': listening,
            'Project_work': project_work
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Grade: {prediction}")
        
        # Show feature importance (optional)
        if st.checkbox("Show explanation"):
            st.subheader("Prediction Explanation")
            st.write("The prediction is based on a Random Forest classifier trained on historical student data. ")
            st.write("Main factors influencing the grade include:")
            st.write("- Study hours and attendance")
            st.write("- Scholarship percentage")
            st.write("- Participation in academic activities")
            st.write("- Note-taking and reading habits")

if __name__ == "__main__":
    main()