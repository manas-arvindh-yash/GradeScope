# app.py — Final Clean Version
import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ------------------------
# Page setup
# ------------------------
st.set_page_config(page_title="GradeScope", layout="centered")

# ------------------------
# Background: blurred image + white text
# ------------------------
def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        /* Blurred background */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
            filter: blur(10px);
            transform: scale(1.05);
            z-index: -1;
        }}

        /* Keep content visible */
        .stApp, .block-container, .main {{
            position: relative;
            z-index: 1;
        }}

        /* Make all text white */
        h1, h2, h3, label, p, span, div {{
            color: #ffffff !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("A celebratory backgr.png")

# ------------------------
# Load dataset
# ------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cstperformance01.csv")

df = load_data()

required_columns = [
    'Gender', 'Age', 'Department', 'Attendance (%)',
    'Midterm_Score', 'Final_Score', 'Assignments_Avg',
    'Projects_Score', 'Study_Hours_per_Week',
    'Extracurricular_Activities','Quizzes_Avg',
    'Internet_Access_at_Home', 'Parent_Education_Level',
    'Family_Income_Level', 'Stress_Level (1-10)',
    'Sleep_Hours_per_Night', 'Total_Score'
]
if not all(col in df.columns for col in required_columns):
    st.error("Dataset missing required columns.")
    st.stop()

df = df[required_columns]

# ------------------------
# Encode categorical columns
# ------------------------
cat_cols = [
    'Gender', 'Department', 'Extracurricular_Activities',
    'Internet_Access_at_Home', 'Parent_Education_Level',
    'Family_Income_Level'
]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    encoders[c] = le

# ------------------------
# Train model
# ------------------------
X = df.drop('Total_Score', axis=1)
y = df['Total_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------
# Title + Logo
# ------------------------
col1, col2 = st.columns([1,5])
with col1:
    st.image("GradeScope logo 1.png", width=80)
with col2:
    st.markdown("<h1 style='color: white; text-align: left;'>GradeScope</h1>", unsafe_allow_html=True)

st.markdown("### Student Performance Prediction")

# ------------------------
# Input Form
# ------------------------
with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study_hours = st.slider("Study Hours/Week", 0, 60, 15)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        sleep = st.slider("Sleep Hours/Night", 0, 12, 7)
        midterm = st.number_input("Midterm Score", 0, 100, 50)
        final = st.number_input("Final Score", 0, 100, 60)

    with right:
        assignments = st.number_input("Assignments Avg", 0, 100, 70)
        projects = st.number_input("Projects Score", 0, 100, 65)
        quizzes = st.number_input("Quizzes Avg", 0, 100, 55)
        gender = st.selectbox("Gender", encoders['Gender'].classes_)
        age = st.number_input("Age", 10, 30, 18)
        dept = st.selectbox("Department", encoders['Department'].classes_)
        activities = st.selectbox("Extracurricular Activities", encoders['Extracurricular_Activities'].classes_)
        internet = st.selectbox("Internet Access at Home", encoders['Internet_Access_at_Home'].classes_)
        parent_edu = st.selectbox("Parent Education Level", encoders['Parent_Education_Level'].classes_)
        income = st.selectbox("Family Income Level", encoders['Family_Income_Level'].classes_)

    submitted = st.form_submit_button("Predict")

# ------------------------
# Prediction
# ------------------------
if submitted:
    input_data = {
        'Gender': encoders['Gender'].transform([gender])[0],
        'Age': age,
        'Department': encoders['Department'].transform([dept])[0],
        'Attendance (%)': attendance,
        'Midterm_Score': midterm,
        'Final_Score': final,
        'Assignments_Avg': assignments,
        'Projects_Score': projects,
        'Study_Hours_per_Week': study_hours,
        'Extracurricular_Activities': encoders['Extracurricular_Activities'].transform([activities])[0],
        'Quizzes_Avg': quizzes,
        'Internet_Access_at_Home': encoders['Internet_Access_at_Home'].transform([internet])[0],
        'Parent_Education_Level': encoders['Parent_Education_Level'].transform([parent_edu])[0],
        'Family_Income_Level': encoders['Family_Income_Level'].transform([income])[0],
        'Stress_Level (1-10)': stress,
        'Sleep_Hours_per_Night': sleep
    }

    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Total Score: {pred:.2f}")
