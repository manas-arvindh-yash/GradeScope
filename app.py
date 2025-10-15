# app.py
import base64
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="GradeScope", layout="centered")

# ------------------------
# Load dataset
# ------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cstperformance01.csv")

df = load_data()

# ------------------------
# Required columns check
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
if not all(col in df.columns for col in required_columns):
    st.error("Dataset is missing required columns.")
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
# Train simple model
# ------------------------
X = df.drop('Total_Score', axis=1)
y = df['Total_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------
# Background + CSS (blurred background only; keep UI on top)
# ------------------------
def set_background_blur(image_path, blur_px=14, overlay_opacity=0.30):
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    css = f"""
    <style>
    /* Put a blurred background image behind everything (fixed) */
    body::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("data:image/png;base64,{b64}");
        background-size: cover;
        background-position: center;
        filter: blur({blur_px}px);
        transform: scale(1.03); /* reduce edge artifacts */
        z-index: -3;
        pointer-events: none;
    }}

    /* Dark overlay to improve contrast */
    body::after {{
        content: "";
        position: fixed;
        inset: 0;
        background-color: rgba(0,0,0,{overlay_opacity});
        z-index: -2;
        pointer-events: none;
    }}

    /* Ensure Streamlit main content is above the background layers */
    .stApp, .main, .block-container, .stSidebar, header {{
        position: relative;
        z-index: 1;
    }}

    /* Style form labels bold + italic and visible */
    .stLabel, label, .css-1kyxreq.e16nr0p33, .stMarkdown {{
        font-weight: 700 !important;
        font-style: italic !important;
        color: #ffffff !important;
    }}

    /* Make the form container slightly brighter so it stands out */
    .block-container {{
        background: rgba(255,255,255,0.02);
        padding: 1rem 1.2rem;
        border-radius: 8px;
    }}

    /* Tweak button style */
    .stButton>button {{
        font-weight: 700;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call this with your background file name (adjust blur_px if needed)
set_background_blur("A celebratory backgr.png", blur_px=14, overlay_opacity=0.30)

# ------------------------
# Header: logo + title (black text)
# ------------------------
logo_col, title_col = st.columns([0.9, 6])
with logo_col:
    st.image("GradeScope logo 1.png", width=84)
with title_col:
    # black text as requested (it will still be readable due to overlay)
    st.markdown("<h1 style='text-align: left; color: black; margin: 0;'>GradeScope</h1>", unsafe_allow_html=True)

# ------------------------
# Compact two-column form (left: sliders + numeric; right: other inputs)
# ------------------------
with st.form("prediction_form"):
    left, right = st.columns([1,1])

    with left:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study_hours = st.slider("Study Hours/Week", 0, 60, 15)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        sleep = st.slider("Sleep Hours/Night", 0, 12, 7)

        # additional numeric inputs placed under sliders for compactness
        midterm = st.number_input("Midterm Score", min_value=0, max_value=100, value=50)
        final = st.number_input("Final Score", min_value=0, max_value=100, value=60)
        assignments = st.number_input("Assignments Avg", min_value=0, max_value=100, value=70)
        projects = st.number_input("Projects Score", min_value=0, max_value=100, value=65)

    with right:
        gender = st.selectbox("Gender", encoders['Gender'].classes_)
        age = st.number_input("Age", min_value=10, max_value=30, value=18)
        dept = st.selectbox("Department", encoders['Department'].classes_)
        quizzes = st.number_input("Quizzes Avg", min_value=0, max_value=100, value=55)
        activities = st.selectbox("Extracurricular Activities", encoders['Extracurricular_Activities'].classes_)
        internet = st.selectbox("Internet Access at Home", encoders['Internet_Access_at_Home'].classes_)
        parent_edu = st.selectbox("Parent Education Level", encoders['Parent_Education_Level'].classes_)
        income = st.selectbox("Family Income Level", encoders['Family_Income_Level'].classes_)

    submitted = st.form_submit_button("Predict")

# ------------------------
# Prediction handling
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
    st.success(f"Predicted Total Score: {pred:.2f}")

# Optional: small footer or instructions (kept minimal)
st.caption("Enter details and click Predict. Design tuned for compact view.")
