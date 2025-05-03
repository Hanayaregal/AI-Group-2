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


## Key Findings

1. **Important Predictors**:
   - Weekly study hours have the strongest correlation with academic performance
   - Regular attendance significantly improves grades
   - Students who take notes and participate in project work tend to perform better

2. **Model Performance**:
   - The Random Forest classifier achieved XX% accuracy
   - The model is particularly good at predicting high-performing (AA) and at-risk (Fail/DD) students

## How to Use

### Prerequisites

- Python 3.8+
- Required packages (install with `pip install -r requirements.txt`)

### Running the Analysis

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook notebooks/Student_Performance_Analysis.ipynb`

### Making Predictions

To predict a student's performance:

python
from src.predict import predict_grade

student_data = {
    'Student_Age': '19-22',
    'Sex': 'Male',
    'High_School_Type': 'State',
    'Scholarship': '50%',
    'Additional_Work': 'No',
    'Sports_activity': 'No',
    'Transportation': 'Private',
    'Weekly_Study_Hours': 12,
    'Attendance': 'Always',
    'Reading': 'Yes',
    'Notes': 'No',
    'Listening_in_Class': 'Yes',
    'Project_work': 'Yes'
}

predicted_grade = predict_grade(student_data)
print(f"Predicted Grade: {predicted_grade}")
## Potential Applications
Early Intervention: Identify at-risk students for targeted support

Resource Allocation: Optimize tutoring and academic support resources

Curriculum Design: Understand which student behaviors correlate with success

Scholarship Decisions: Inform merit-based scholarship allocations

## Limitations
Dataset size is relatively small (145 students)

Grades may be influenced by factors not captured in the data

Model performance may vary across different educational institutions

## Future Work
Collect more data from additional institutions

Incorporate more features (e.g., family background, previous academic history)

Develop a web interface for easy prediction

Implement time-series analysis for tracking performance over semesters

## License
This project is licensed under the MIT License - see the LICENSE file for details.

This README provides:
1. Clear overview of the dataset
2. Project structure explanation
3. Key findings from the analysis
4. Instructions for running the code
5. Potential applications and limitations
6. Future work suggestions

You can customize it further by adding:
- Specific accuracy metrics from your model
- Visualizations of important findings
- Contributor guidelines if it's an open-source project
- Citation information if the dataset comes from a published source
