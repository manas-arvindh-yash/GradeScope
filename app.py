import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import base64

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="GradeScope", layout="centered")

# ------------------------
# Load Data
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cstperformance01.csv")
    return df

df = load_data()

# ------------------------
# Encode Columns
# ------------------------
required_columns = [
    'Gender', 'Age', 'Department', 'Attendance (%)',
    'Midterm_Score', 'Final_Score', 'Assignments_Avg',
    'Projects_Score', 'Study_Hours_per_Week',
    'Extracurricular_Activities','Quizzes_Avg',
    'Internet_Access_at_Home', 'Parent_Education_Level',
    'Family_Income_Level', 'Stress_Level (1-10)',
    'Sleep_Hours_per_Night', 'Total_Score'
]

df = df[required_columns]

cat_cols = [
    'Gender', 'Department', 'Extracurricular_Activities',
    'Internet_Access_at_Home', 'Parent_Education_Level',
    'Family_Income_Level'
]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ------------------------
# Train Model
# ------------------------
X = df.drop('Total_Score', axis=1)
y = df['Total_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------
# Background Styling (more blur on background only)
# ------------------------
def add_bg(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        /* Blurred background layer */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: inherit;
            filter: blur(36px);  /* ⬅️ Increased blur from 6px to 12px */
            z-index: -2;
        }}
        /* Dark overlay layer */
        .stApp::after {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.35);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("A celebratory backgr.png")


# ------------------------
# Logo + Title (without emoji)
# ------------------------
col_logo, col_title = st.columns([1,5])
with col_logo:
    st.image("GradeScope logo 1.png", width=90)
with col_title:
    st.markdown("<h1 style='text-align: left; color: white;'>GradeScope</h1>", unsafe_allow_html=True)


# ------------------------
# Prediction Form
# ------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study_hours = st.slider("Study Hours/Week", 0, 60, 15)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        sleep = st.slider("Sleep Hours/Night", 0, 12, 7)
        midterm = st.number_input("Midterm Score", min_value=0, max_value=100, value=50)
        final = st.number_input("Final Score", min_value=0, max_value=100, value=60)

    with col2:
        gender = st.selectbox("Gender", encoders['Gender'].classes_)
        age = st.number_input("Age", min_value=10, max_value=30, value=18)
        dept = st.selectbox("Department", encoders['Department'].classes_)
        assignments = st.number_input("Assignments Avg", min_value=0, max_value=100, value=70)
        projects = st.number_input("Projects Score", min_value=0, max_value=100, value=65)
        activities = st.selectbox("Extracurricular Activities", encoders['Extracurricular_Activities'].classes_)
        quizzes = st.number_input("Quizzes Avg", min_value=0, max_value=100, value=55)
        internet = st.selectbox("Internet Access", encoders['Internet_Access_at_Home'].classes_)
        parent_edu = st.selectbox("Parent Education Level", encoders['Parent_Education_Level'].classes_)
        income = st.selectbox("Family Income Level", encoders['Family_Income_Level'].classes_)

    submitted = st.form_submit_button("Predict")
# ------------------------
# Style: Bold + Italic form labels
# ------------------------
st.markdown(
    """
    <style>
    label, .stSlider label, .stNumberInput label, .stSelectbox label {
        font-weight: bold !important;
        font-style: italic !important;
        color: white; /* optional, keeps text visible on dark bg */
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    prediction = model.predict(input_df)[0]

    st.success(f"🎯 Predicted Total Score: **{prediction:.2f}**")
