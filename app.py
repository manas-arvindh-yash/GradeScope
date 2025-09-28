import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="GradeScope", layout="wide")

# ------------------------
# Load Data
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cstperformance01.csv")
    return df

df = load_data()

# ------------------------
# Validate Columns
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
    st.error("❌ Dataset is missing required columns.")
    st.stop()

df = df[required_columns]

# ------------------------
# Encode Categorical Columns
# ------------------------
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
# Train-Test Split
# ------------------------
X = df.drop('Total_Score', axis=1)
y = df['Total_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Train Model
# ------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------
# Metrics
# ------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ------------------------
# Streamlit UI
# ------------------------
st.title("📊 GradeScope: Student Performance Prediction")
st.markdown("AI-powered prediction of student performance using academic, personal, and socio-economic features.")

# --- Model performance
st.header("Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE", f"{rmse:.2f}")
with col2:
    st.metric("R² Score", f"{r2:.2f}")

# --- Visualization
st.header("Actual vs Predicted")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6, color="blue")
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted Total Score")
st.pyplot(fig)

# ------------------------
# Live Prediction Form
# ------------------------
st.header("Try Live Prediction")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", encoders['Gender'].classes_)
    age = st.number_input("Age", min_value=10, max_value=30, value=18)
    dept = st.selectbox("Department", encoders['Department'].classes_)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    midterm = st.number_input("Midterm Score", min_value=0, max_value=100, value=50)
    final = st.number_input("Final Score", min_value=0, max_value=100, value=60)
    assignments = st.number_input("Assignments Avg", min_value=0, max_value=100, value=70)
    projects = st.number_input("Projects Score", min_value=0, max_value=100, value=65)
    study_hours = st.slider("Study Hours per Week", 0, 60, 15)
    activities = st.selectbox("Extracurricular Activities", encoders['Extracurricular_Activities'].classes_)
    quizzes = st.number_input("Quizzes Avg", min_value=0, max_value=100, value=55)
    internet = st.selectbox("Internet Access at Home", encoders['Internet_Access_at_Home'].classes_)
    parent_edu = st.selectbox("Parent Education Level", encoders['Parent_Education_Level'].classes_)
    income = st.selectbox("Family Income Level", encoders['Family_Income_Level'].classes_)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    sleep = st.slider("Sleep Hours per Night", 0, 12, 7)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode categorical inputs
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
 
