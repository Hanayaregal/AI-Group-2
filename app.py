import streamlit as st # type: ignore
import pandas as pd # type: ignore

# Create a DataFrame with your data
data = {
    "Student_ID": ["STUDENT1", "STUDENT2", "STUDENT3", "STUDENT4", "STUDENT5","STUDENT1", "STUDENT2", "STUDENT3", "STUDENT4", "STUDENT5"],
    "Student_Age": ["19-22", "19-22", "19-22", "18", "19-22","19-22", "19-22", "19-22", "18", "19-22"],
    "Sex": ["Male", "Male", "Male", "Female", "Male","Male", "Male", "Male", "Female", "Male"],
    "High_School_Type": ["Other", "Other", "State", "Private", "Private","Other", "Other", "State", "Private", "Private"],
    "Scholarship": ["50%", "50%", "50%", "50%", "50%","50%", "50%", "50%", "50%", "50%"],
    "Additional_Work": ["Yes", "Yes", "No", "Yes", "No","Yes", "Yes", "No", "Yes", "No"],
    "Sports_activity": ["No", "No", "No", "No", "No","No", "No", "No", "No", "No"],
    "Transportation": ["Private", "Private", "Private", "Bus", "Bus","Private", "Private", "Private", "Bus", "Bus"],
    "Weekly_Study_Hours": [0, 0, 2, 2, 12,0, 0, 2, 2, 12],
    "Attendance": ["Always", "Always", "Never", "Always", "Always","Always", "Always", "Never", "Always", "Always"],
    "Reading": ["Yes", "Yes", "No", "No", "Yes","No", "Yes", "Yes", "No", "Yes"],
    "Notes": ["Yes", "No", "No", "Yes", "No","No", "Yes", "Yes", "No", "Yes"],
    "Listening_in_Class": ["No", "Yes", "No", "No", "Yes","No", "Yes", "Yes", "No", "Yes"],
    "Project_work": ["No", "Yes", "Yes", "No", "Yes","No", "Yes", "Yes", "No", "Yes"],
    "Grade": ["AA", "AA", "AA", "AA", "AA","BB", "BB", "BB", "BB", "BB"]

}



# Convert to DataFrame
df = pd.DataFrame(data)

# Streamlit app
st.title("Student Performance Data")
st.subheader("Student Data Overview")

# Display the DataFrame
st.dataframe(df)

# Filter by Grade
grade_filter = st.selectbox("Filter by Grade", df['Grade'].unique())
filtered_df = df[df['Grade'] == grade_filter]

# Display the filtered DataFrame
st.subheader(f"Filtered by Grade: {grade_filter}")
st.dataframe(filtered_df)

# Sort by Weekly Study Hours
st.subheader("Sort by Weekly Study Hours")
sort_by_study_hours = st.selectbox("Sort by", ['Descending', 'Ascending'])

if sort_by_study_hours == 'Descending':
    sorted_df = df.sort_values(by='Weekly_Study_Hours', ascending=False)
else:
    sorted_df = df.sort_values(by='Weekly_Study_Hours', ascending=True)

# Display the sorted DataFrame
st.dataframe(sorted_df)

# Summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())
