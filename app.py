import streamlit as st
import joblib
import pandas as pd
import pickle

# Load model
data = joblib.load("salary_model.pkl")
model = data["model"]
features = data["features"]

# Load encoders
experience_encoder = pickle.load(open("experience_level_encoder.pkl", "rb"))
employment_encoder = pickle.load(open("employment_type_encoder.pkl", "rb"))
job_encoder = pickle.load(open("job_title_encoder.pkl", "rb"))
currency_encoder = pickle.load(open("salary_currency_encoder.pkl", "rb"))
residence_encoder = pickle.load(open("employee_residence_encoder.pkl", "rb"))
location_encoder = pickle.load(open("company_location_encoder.pkl", "rb"))
size_encoder = pickle.load(open("company_size_encoder.pkl", "rb"))

st.title("Salary Prediction App")
st.write("Enter details to predict salary")

# Numeric inputs
work_year = st.number_input("Work Year", 2020, 2030, 2024)
remote_ratio = st.slider("Remote Ratio", 0, 100, 0)
salary = st.number_input("Salary", min_value=0, value=50000)

# Categorical inputs (SELECTBOX)
experience_level = st.selectbox(
    "Experience Level", experience_encoder.classes_
)

employment_type = st.selectbox(
    "Employment Type", employment_encoder.classes_
)

job_title = st.selectbox(
    "Job Title", job_encoder.classes_
)

salary_currency = st.selectbox(
    "Salary Currency", currency_encoder.classes_
)

employee_residence = st.selectbox(
    "Employee Residence", residence_encoder.classes_
)

company_location = st.selectbox(
    "Company Location", location_encoder.classes_
)

company_size = st.selectbox(
    "Company Size", size_encoder.classes_
)

# 🔁 Convert to encoded values
input_data = pd.DataFrame([{
    "work_year": work_year,
    "salary": salary,
    "remote_ratio": remote_ratio,
    "experience_level": experience_encoder.transform([experience_level])[0],
    "employment_type": employment_encoder.transform([employment_type])[0],
    "job_title": job_encoder.transform([job_title])[0],
    "salary_currency": currency_encoder.transform([salary_currency])[0],
    "employee_residence": residence_encoder.transform([employee_residence])[0],
    "company_location": location_encoder.transform([company_location])[0],
    "company_size": size_encoder.transform([company_size])[0]
}])

# Ensure correct order
input_data = input_data[features]

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Salary in USD: ${prediction[0]:,.2f}")