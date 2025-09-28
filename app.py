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

st.header("1. Dataset Preview")
st.dataframe(df.head())

st.header("2. Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE", f"{rmse:.2f}")
with col2:
    st.metric("R² Score", f"{r2:.2f}")

st.header("3. Actual vs Predicted")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6, color="blue")
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted Total Score")
st.pyplot(fig)

st.success("✅ Model training and evaluation complete. You can now explore predictions and analysis.")
