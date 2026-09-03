import sys
sys.path.insert(0, '.')

import streamlit as st
from src.predict import predict_depression_risk, DEGREE_OPTIONS

st.set_page_config(page_title="Student Depression Risk Predictor", layout="centered")
st.title("Student Depression Risk Predictor")
st.write(
    "Enter student information below to get a model-estimated depression risk. "
    "This is a predictive tool based on associations in survey data — it does "
    "not diagnose depression or represent clinical advice."
)

st.header("Student Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 59, 22)
    academic_pressure = st.slider("Academic Pressure (1-5)", 1.0, 5.0, 3.0, step=0.5)
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, step=0.1)
    study_satisfaction = st.slider("Study Satisfaction (1-5)", 1.0, 5.0, 3.0, step=0.5)
    sleep_duration = st.selectbox(
        "Sleep Duration",
        ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],
    )

with col2:
    dietary_habits = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"])
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
    work_study_hours = st.slider("Work/Study Hours per day", 0.0, 12.0, 6.0, step=0.5)
    financial_stress = st.slider("Financial Stress (1-5)", 1.0, 5.0, 3.0, step=0.5)
    family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])
    degree = st.selectbox("Degree", sorted(DEGREE_OPTIONS))

st.header("Prediction")

if st.button("Predict Depression Risk"):
    student = {
        "Gender": gender,
        "Age": age,
        "Academic Pressure": academic_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history,
        "Degree": degree,
    }

    result = predict_depression_risk(student)

    st.subheader("Result")
    if result["predicted_class"] == 1:
        st.error(f"Model-estimated risk: HIGH (probability: {result['probability']:.1%})")
    else:
        st.success(f"Model-estimated risk: LOW (probability: {result['probability']:.1%})")

    st.caption(
        "This estimate reflects patterns in survey data and is not a "
        "clinical diagnosis. If you or someone you know is struggling, "
        "please consult a mental health professional."
    )