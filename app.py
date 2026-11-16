
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="centered"
)

# ==================================================
# CONFIGURATION
# ==================================================

DATA_FILE = "HR-Employee-Attrition.csv"

MODEL_FILE = "attrition_model_12_features.pkl"

# Exactly 12 selected features

NUMERIC_FEATURES = [
    "Age",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsWithCurrManager",
    "PerformanceRating",
    "JobSatisfaction"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "EducationField",
    "Department",
    "JobRole",
    "OverTime"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ==================================================
# FILE LOCATION
# ==================================================

def find_file(filename):

    app_dir = os.path.dirname(os.path.abspath(__file__))

    paths = [
        os.path.join(app_dir, filename),
        os.path.join(os.getcwd(), filename)
    ]

    return next(
        (path for path in paths if os.path.isfile(path)),
        None
    )


data_path = find_file(DATA_FILE)


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data(path):

    return pd.read_csv(path)


if not data_path:

    st.error(
        "HR-Employee-Attrition.csv not found. "
        "Place it in the application folder."
    )

    st.stop()


try:

    data = load_data(data_path)

except Exception as exc:

    st.error(f"Unable to load dataset: {exc}")
    st.stop()


# ==================================================
# VALIDATE DATA
# ==================================================

required_columns = FEATURES + ["Attrition"]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:

    st.error(
        "Missing columns in dataset: "
        + ", ".join(missing_columns)
    )

    st.stop()


# Clean target variable

data["Attrition"] = (
    data["Attrition"]
    .astype(str)
    .str.strip()
    .str.lower()
)

data["Attrition"] = data["Attrition"].map({
    "yes": 1,
    "no": 0
})

# Convert numeric fields

for col in NUMERIC_FEATURES:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

# Remove incomplete records

data = data.dropna(
    subset=FEATURES + ["Attrition"]
).copy()

if data.empty:

    st.error("No valid training records found.")
    st.stop()

data["Attrition"] = data["Attrition"].astype(int)

if data["Attrition"].nunique() != 2:

    st.error(
        "Training requires both Attrition Yes and No records."
    )

    st.stop()


# ==================================================
# TRAIN MODEL
# ==================================================

@st.cache_resource
def train_model(data):

    X = data[FEATURES].copy()
    y = data["Attrition"].copy()

    # Preprocessing

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                "passthrough",
                NUMERIC_FEATURES
            )
        ]
    )

    # Random Forest Classifier

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=42,
        class_weight=None,
        n_jobs=-1
    )

    # Complete pipeline

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier)
    ])

    pipeline.fit(X, y)

    return pipeline


with st.spinner("Preparing attrition prediction model..."):

    try:

        model = train_model(data)

    except Exception as exc:

        st.error(f"Model training failed: {exc}")
        st.stop()


# ==================================================
# APPLICATION HEADER
# ==================================================

st.title("Employee Attrition Predictor")

st.caption(
    "AI-powered employee attrition risk assessment"
)

st.divider()

st.markdown(
    "### Employee Profile"
)

st.write(
    "Enter employee information below to estimate "
    "the likelihood of attrition."
)


# ==================================================
# INPUT FORM
# ==================================================

employee_input = {}

with st.form("employee_prediction_form"):

    # ----------------------------------------------
    # PERSONAL INFORMATION
    # ----------------------------------------------

    st.markdown("#### 1. Personal Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            min_value=int(data["Age"].min()),
            max_value=int(data["Age"].max()),
            value=int(data["Age"].median())
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            sorted(
                data["Gender"].dropna().astype(str).unique()
            )
        )

    col1, col2 = st.columns(2)

    with col1:

        education_field = st.selectbox(
            "Education Field",
            sorted(
                data["EducationField"]
                .dropna()
                .astype(str)
                .unique()
            )
        )

    with col2:

        department = st.selectbox(
            "Department",
            sorted(
                data["Department"]
                .dropna()
                .astype(str)
                .unique()
            )
        )

    # ----------------------------------------------
    # JOB INFORMATION
    # ----------------------------------------------

    st.divider()

    st.markdown("#### 2. Job Information")

    col1, col2 = st.columns(2)

    with col1:

        job_role = st.selectbox(
            "Job Role",
            sorted(
                data["JobRole"]
                .dropna()
                .astype(str)
                .unique()
            )
        )

    with col2:

        overtime = st.selectbox(
            "Overtime",
            sorted(
                data["OverTime"]
                .dropna()
                .astype(str)
                .unique()
            )
        )

    # ----------------------------------------------
    # COMPENSATION AND EXPERIENCE
    # ----------------------------------------------

    st.divider()

    st.markdown("#### 3. Compensation & Experience")

    monthly_income = st.slider(
        "Monthly Income",
        min_value=int(data["MonthlyIncome"].min()),
        max_value=int(data["MonthlyIncome"].max()),
        value=int(data["MonthlyIncome"].median()),
        step=100
    )

    col1, col2 = st.columns(2)

    with col1:

        total_working_years = st.slider(
            "Total Working Years",
            min_value=int(data["TotalWorkingYears"].min()),
            max_value=int(data["TotalWorkingYears"].max()),
            value=int(data["TotalWorkingYears"].median())
        )

    with col2:

        years_at_company = st.slider(
            "Years at Company",
            min_value=int(data["YearsAtCompany"].min()),
            max_value=int(data["YearsAtCompany"].max()),
            value=int(data["YearsAtCompany"].median())
        )

    years_with_manager = st.slider(
        "Years with Current Manager",
        min_value=int(data["YearsWithCurrManager"].min()),
        max_value=int(data["YearsWithCurrManager"].max()),
        value=int(data["YearsWithCurrManager"].median())
    )

    # ----------------------------------------------
    # PERFORMANCE AND SATISFACTION
    # ----------------------------------------------

    st.divider()

    st.markdown("#### 4. Performance & Satisfaction")

    col1, col2 = st.columns(2)

    with col1:

        performance_rating = st.slider(
            "Performance Rating",
            min_value=int(data["PerformanceRating"].min()),
            max_value=int(data["PerformanceRating"].max()),
            value=int(data["PerformanceRating"].median())
        )

    with col2:

        job_satisfaction = st.slider(
            "Job Satisfaction",
            min_value=int(data["JobSatisfaction"].min()),
            max_value=int(data["JobSatisfaction"].max()),
            value=int(data["JobSatisfaction"].median())
        )

    # ----------------------------------------------
    # SUBMIT
    # ----------------------------------------------

    st.divider()

    submitted = st.form_submit_button(
        "Predict Attrition Risk",
        type="primary",
        use_container_width=True
    )


# ==================================================
# PREDICTION
# ==================================================

if submitted:

    try:

        # Prepare employee input

        employee_input = {

            "Age": age,
            "Gender": gender,
            "EducationField": education_field,
            "Department": department,
            "JobRole": job_role,
            "OverTime": overtime,
            "MonthlyIncome": monthly_income,
            "TotalWorkingYears": total_working_years,
            "YearsAtCompany": years_at_company,
            "YearsWithCurrManager": years_with_manager,
            "PerformanceRating": performance_rating,
            "JobSatisfaction": job_satisfaction

        }

        # Create dataframe with exact feature order

        input_df = pd.DataFrame(
            [employee_input],
            columns=FEATURES
        )

        # Generate prediction

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]

        # Extract attrition probability

        positive_index = list(model.classes_).index(1)

        attrition_probability = (
            probabilities[positive_index] * 100
        )

        # Risk classification
        # Initial presentation thresholds only

        if attrition_probability >= 60:

            risk = "High"
            color = "red"

        elif attrition_probability >= 30:

            risk = "Moderate"
            color = "orange"

        else:

            risk = "Low"
            color = "green"

        # ------------------------------------------
        # DISPLAY RESULT
        # ------------------------------------------

        st.divider()

        st.markdown("## Prediction Result")

        st.metric(
            "Employee Attrition Probability",
            f"{attrition_probability:.1f}%"
        )

        st.progress(
            float(attrition_probability / 100)
        )

        st.markdown(
            f"### Risk Classification: :{color}[{risk}]"
        )

        if prediction == 1:

            st.warning(
                "The model predicts potential employee attrition."
            )

        else:

            st.success(
                "The model predicts employee retention."
            )

        st.caption(
            "This is a model-estimated probability, "
            "not a certainty of employee departure."
        )

        # ------------------------------------------
        # EMPLOYEE PROFILE
        # ------------------------------------------

        with st.expander("View Employee Parameters"):

            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Employee Attrition Prediction | HR Analytics"
)

st.caption(
    "For workforce planning and exploratory analysis. "
    "Predictions should not be used as the sole basis "
    "for employment decisions."
)