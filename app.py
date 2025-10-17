import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="GradeScope", layout="wide")

# ------------------------
# Background (no blur)
# ------------------------
def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        /* Background image */
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* Global text color */
        h1, h2, h3, label, p, span, div {{
            color: #ffffff !important;
        }}

        /* Title + logo layout */
        .center-title {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-top: 25px;
            margin-bottom: 10px;
        }}

        .center-title img {{
            width: 70px;
            height: auto;
            border-radius: 10px;
        }}

        .center-title h1 {{
            font-size: 2.8rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
            padding-top: 10px;  /* lowers the title */
        }}

        /* Prediction text */
        .prediction-text {{
            font-size: 1.5rem;
            font-weight: 800;
            color: #00ffff;
            font-family: 'Trebuchet MS', sans-serif;
            text-transform: uppercase;
            margin-left: 15px;
            display: inline;
        }}

        /* Predict button */
        .stButton>button {{
            background-color: #0072ff;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            display: inline;
            margin-right: 10px;
        }}

        .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 95%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("bckgrnd.png")

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
    'Extracurricular_Activities', 'Quizzes_Avg',
    'Internet_Access_at_Home', 'Parent_Education_Level',
    'Family_Income_Level', 'Stress_Level (1-10)',
    'Sleep_Hours_per_Night', 'Total_Score'
]
if not all(col in df.columns for col in required_columns):
    st.error("Dataset missing required columns.")
    st.stop()

df = df[required_columns]

# Encode categorical columns
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

# Train model
X = df.drop('Total_Score', axis=1)
y = df['Total_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------
# Header (Logo + Title)
# ------------------------
st.markdown(
    """
    <div class='center-title'>
        <img src='GradeScope logo 1.png'>
        <h1>GradeScope</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------
# Input Form
# ------------------------
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study_hours = st.slider("Study Hours/Week", 0, 60, 15)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        sleep = st.slider("Sleep Hours/Night", 0, 12, 7)
        midterm = st.number_input("Midterm Score", 0, 100, 50)

    with c2:
        final = st.number_input("Final Score", 0, 100, 60)
        assignments = st.number_input("Assignments Avg", 0, 100, 70)
        projects = st.number_input("Projects Score", 0, 100, 65)
        quizzes = st.number_input("Quizzes Avg", 0, 100, 55)
        gender = st.selectbox("Gender", encoders['Gender'].classes_)

    with c3:
        age = st.number_input("Age", 10, 30, 18)
        dept = st.selectbox("Department", encoders['Department'].classes_)
        activities = st.selectbox("Extracurricular Activities", encoders['Extracurricular_Activities'].classes_)
        internet = st.selectbox("Internet Access at Home", encoders['Internet_Access_at_Home'].classes_)
        parent_edu = st.selectbox("Parent Education Level", encoders['Parent_Education_Level'].classes_)
        income = st.selectbox("Family Income Level", encoders['Family_Income_Level'].classes_)

    # Predict button and result side by side (no gap)
    c_btn, c_output = st.columns([1, 4])
    with c_btn:
        submitted = st.form_submit_button("Predict")
    with c_output:
        result_placeholder = st.empty()

# ------------------------
# Prediction Output
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

    result_placeholder.markdown(
        f"<p class='prediction-text'>YOUR PREDICTED SCORE IS: {pred:.2f}</p>",
        unsafe_allow_html=True
    )
