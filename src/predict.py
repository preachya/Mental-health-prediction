"""Prediction interface for the trained Logistic Regression depression model.

Takes raw student characteristics (the same categories that appear in the
original CSV) and applies the same encoding used in 02_preprocessing.ipynb,
then returns a predicted class and probability using the saved model.
"""

import joblib
import pandas as pd

MODEL_PATH = "models/logistic_regression.joblib"
SCALER_PATH = "models/scaler.joblib"

GENDER_MAP = {"Male": 1, "Female": 0}
YES_NO_MAP = {"Yes": 1, "No": 0}
DIETARY_MAP = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2, "Others": 0}
SLEEP_MAP = {
    "Less than 5 hours": 0,
    "5-6 hours": 1,
    "7-8 hours": 2,
    "More than 8 hours": 3,
    "Others": 0,
}

# Exact column order the model was trained on (from data/processed/train.csv)
FEATURE_COLUMNS = [
    "Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction",
    "Sleep Duration", "Dietary Habits", "Have you ever had suicidal thoughts ?",
    "Work/Study Hours", "Financial Stress", "Family History of Mental Illness",
    "Degree_B.Arch", "Degree_B.Com", "Degree_B.Ed", "Degree_B.Pharm",
    "Degree_B.Tech", "Degree_BA", "Degree_BBA", "Degree_BCA", "Degree_BE",
    "Degree_BHM", "Degree_BSc", "Degree_Class 12", "Degree_LLB", "Degree_LLM",
    "Degree_M.Com", "Degree_M.Ed", "Degree_M.Pharm", "Degree_M.Tech",
    "Degree_MA", "Degree_MBA", "Degree_MBBS", "Degree_MCA", "Degree_MD",
    "Degree_ME", "Degree_MHM", "Degree_MSc", "Degree_Others", "Degree_PhD",
]

DEGREE_OPTIONS = [c.replace("Degree_", "") for c in FEATURE_COLUMNS if c.startswith("Degree_")]


def _encode(student: dict) -> pd.DataFrame:
    """Turn a raw student dict into a single-row, model-ready DataFrame."""
    row = {
        "Gender": GENDER_MAP[student["Gender"]],
        "Age": student["Age"],
        "Academic Pressure": student["Academic Pressure"],
        "CGPA": student["CGPA"],
        "Study Satisfaction": student["Study Satisfaction"],
        "Sleep Duration": SLEEP_MAP[student["Sleep Duration"]],
        "Dietary Habits": DIETARY_MAP[student["Dietary Habits"]],
        "Have you ever had suicidal thoughts ?": YES_NO_MAP[student["Have you ever had suicidal thoughts ?"]],
        "Work/Study Hours": student["Work/Study Hours"],
        "Financial Stress": student["Financial Stress"],
        "Family History of Mental Illness": YES_NO_MAP[student["Family History of Mental Illness"]],
    }

    degree = student["Degree"]
    for col in FEATURE_COLUMNS:
        if col.startswith("Degree_"):
            row[col] = 1 if col == f"Degree_{degree}" else 0

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_depression_risk(student: dict) -> dict:
    """Predict depression risk for one student.

    `student` must have keys: Gender, Age, Academic Pressure, CGPA,
    Study Satisfaction, Sleep Duration, Dietary Habits,
    'Have you ever had suicidal thoughts ?', Work/Study Hours,
    Financial Stress, Family History of Mental Illness, Degree.

    Returns a dict with predicted_class (0/1) and probability (0-1).
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X = _encode(student)
    X_scaled = scaler.transform(X)

    predicted_class = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0, 1])

    return {
        "predicted_class": predicted_class,
        "probability": round(probability, 4),
    }
