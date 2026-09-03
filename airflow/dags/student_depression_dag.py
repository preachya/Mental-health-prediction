"""Airflow DAG: rebuild the Student Depression model end-to-end.

Reruns the same cleaning/encoding decisions from 01_eda.ipynb and
02_preprocessing.ipynb, then retrains the chosen final model (Logistic
Regression) from 03_modeling.ipynb. Reads the raw CSV from and writes
processed data / the trained model back into the project folder on the
Windows filesystem (mounted at /mnt/c inside WSL).
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_DIR = "/mnt/c/Users/Acer/Documents/Mental-health-prediction"
RAW_CSV = f"{PROJECT_DIR}/data/raw/Student Depression Dataset.csv"
TRAIN_CSV = f"{PROJECT_DIR}/data/processed/train.csv"
TEST_CSV = f"{PROJECT_DIR}/data/processed/test.csv"
MODEL_PATH = f"{PROJECT_DIR}/models/logistic_regression.joblib"
SCALER_PATH = f"{PROJECT_DIR}/models/scaler.joblib"

SUICIDAL_COL = "Have you ever had suicidal thoughts ?"


def preprocess_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(RAW_CSV)

    # Same decisions as 01_eda.ipynb / 02_preprocessing.ipynb
    df = df.dropna(subset=["Financial Stress"]).reset_index(drop=True)
    df = df.drop(columns=["id", "City", "Profession", "Work Pressure", "Job Satisfaction"])

    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["Family History of Mental Illness"] = (
        df["Family History of Mental Illness"].map({"Yes": 1, "No": 0})
    )
    df[SUICIDAL_COL] = df[SUICIDAL_COL].map({"Yes": 1, "No": 0})

    df["Dietary Habits"] = df["Dietary Habits"].replace("Others", "Unhealthy")
    df["Dietary Habits"] = df["Dietary Habits"].map({"Unhealthy": 0, "Moderate": 1, "Healthy": 2})

    df["Sleep Duration"] = df["Sleep Duration"].replace("Others", "Less than 5 hours")
    df["Sleep Duration"] = df["Sleep Duration"].map({
        "Less than 5 hours": 0,
        "5-6 hours": 1,
        "7-8 hours": 2,
        "More than 8 hours": 3,
    })

    df = pd.get_dummies(df, columns=["Degree"], prefix="Degree")

    X = df.drop(columns=["Depression"])
    y = df["Depression"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train = X_train.copy()
    train["Depression"] = y_train
    test = X_test.copy()
    test["Depression"] = y_test

    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"Saved train {train.shape} and test {test.shape} to data/processed/")


def train_model():
    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    X_train = train.drop(columns=["Depression"])
    y_train = train["Depression"]
    X_test = test.drop(columns=["Depression"])
    y_test = test["Depression"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"F1:       {f1_score(y_test, pred):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")


with DAG(
    "student_depression_ml_pipeline",
    description="Rebuild the Student Depression Logistic Regression model",
    start_date=datetime(2026, 1, 1),
    schedule=None,  # trigger manually; change to e.g. '@weekly' to automate
    catchup=False,
    tags=["student-depression"],
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    preprocess_task >> train_task
