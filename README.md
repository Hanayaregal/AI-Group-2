## Student Performance Prediction

This project analyzes student performance data and builds a machine learning model to predict academic grades based on various student characteristics and behaviors.

## Dataset Overview
The dataset contains information about 145 students with the following attributes:
Student_ID: Unique identifier for each student
Student_Age: Age group of the student (18, 19-22, 23-27)
Sex: Gender of the student (Male/Female)
High_School_Type: Type of high school attended (State, Private, Other)
Scholarship: Scholarship percentage (None, 25%, 50%, 75%, 100%)
Additional_Work: Whether the student has additional work (Yes/No)
Sports_activity: Participation in sports (Yes/No)
Transportation: Mode of transportation (Private, Bus)
Weekly_Study_Hours: Hours spent studying per week (0, 2, 8, 12)
Attendance: Class attendance frequency (Always, Sometimes, Never)
Reading: Whether the student reads outside class (Yes/No)
Notes: Whether the student takes notes (Yes/No)
Listening_in_Class: Attention level in class (Yes/No)
Project_work: Participation in project work (Yes/No)
Grade: Final academic grade (AA, BA, BB, CB, CC, DC, DD, Fail)

## Project Structure
student-performance-prediction/

├── data/
│ 
└── student_performance.csv # Raw dataset
├── notebooks/
│ 
└── Student_Performance_Analysis.ipynb # Jupyter notebook with analysis
├── src/
│
├── preprocess.py # Data preprocessing script
│ 
├── train_model.py # Model training script
│
└── predict.py # Prediction script
├── models/
│
└── random_forest_model.pkl # Trained model
├── README.md # This file
└── requirements.txt # Python dependencies
