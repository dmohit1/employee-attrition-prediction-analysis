# Employee Attrition Prediction

A machine learning-powered HR analytics application that predicts employee attrition probability using 12 key employee attributes.

## Project Overview
This project uses a Random Forest Classifier to estimate employee attrition risk based on demographic, job-related, compensation, experience, and satisfaction parameters.

## Key Features
- Interactive Streamlit interface with dropdowns and sliders.
- 12 selected employee attributes for prediction.
- Automated categorical encoding using One-Hot Encoding.
- Individual attrition probability (%) prediction.
- Low, Moderate, and High risk classification.

## Input Parameters
Age, Gender, Education Field, Department, Job Role, Overtime, Monthly Income, Total Working Years, Years at Company, Years with Current Manager, Performance Rating, and Job Satisfaction.

## Technology Stack
- Python
- Pandas & NumPy
- Scikit-learn
- Random Forest Classifier
- Streamlit

## Dataset
IBM HR Employee Attrition dataset (1,470 employee records).

## Run Locally

```bash
pip install streamlit pandas numpy scikit-learn joblib
streamlit run app.py
```

## Objective
Demonstrate how machine learning can support proactive workforce analytics by identifying employee attrition risk patterns and enabling data-informed HR planning.

*Note: Predictions are exploratory estimates and should not be used as the sole basis for employment decisions.*
