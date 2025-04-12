import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 🎯 Grade Mapping Functions
# ----------------------------
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

# ----------------------------
# 🔁 Age Conversion Helper
# ----------------------------
def convert_age(age_value):
    if '-' in str(age_value):
        parts = age_value.split('-')
        return (float(parts[0]) + float(parts[1])) / 2
    try:
        return float(age_value)
    except:
        return None

# ----------------------------
# 📥 Load and Preprocess Data
# ----------------------------
data = pd.read_csv("Students _Performance _Prediction.csv")

data['Scholarship'] = data['Scholarship'].str.replace('%', '', regex=False).astype(float)
data['Student_Age'] = data['Student_Age'].apply(convert_age)
data['Grade'] = data['Grade'].apply(map_grade_letter_to_numeric)

# Encode categoricals
categorical_columns = data.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Features & Target
features = ['Student_Age', 'Sex', 'High_School_Type', 'Scholarship', 'Additional_Work',
            'Sports_activity', 'Transportation', 'Weekly_Study_Hours', 'Attendance',
            'Reading', 'Notes', 'Listening_in_Class', 'Project_work']
X = data[features]
y = data['Grade']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 🌐 Streamlit Interface
# ----------------------------
st.title("🎓 Student Grade Predictor")
st.markdown("Upload student data CSV to predict their grades.")

uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)

        input_df['Scholarship'] = input_df['Scholarship'].str.replace('%', '', regex=False).astype(float)
        input_df['Student_Age'] = input_df['Student_Age'].apply(convert_age)

        for col in input_df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0])

        input_features = input_df[features]
        predicted_numeric = model.predict(input_features)
        predicted_letters = [convert_numeric_to_letter(p) for p in predicted_numeric]

        input_df['Predicted Grade (Numeric)'] = predicted_numeric
        input_df['Predicted Grade (Letter)'] = predicted_letters

        st.success("✅ Prediction complete!")
        st.dataframe(input_df)

        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv, "predicted_results.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")