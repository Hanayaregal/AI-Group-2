import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
df = pd.read_csv('Students _Performance _Prediction.csv')

# Clean and preprocess
df['Notes'] = df['Notes'].replace('6', 'No')
df['Scholarship'] = df['Scholarship'].str.rstrip('%').replace('None', '0').astype(float) / 100
df['Student_Age'] = df['Student_Age'].apply(lambda x: '18' if x == '18' else '19-22' if x == '19-22' else '23-27')

# Define X and y
X = df.drop(['Grade', 'Student_ID'], axis=1)
y = df['Grade']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing
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

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Build final pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Save pipeline
joblib.dump(model, 'student_grade_predictor.pkl')

print("✅ Model saved as student_grade_predictor.pkl")
