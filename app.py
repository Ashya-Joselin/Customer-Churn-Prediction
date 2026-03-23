import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/churn_model.pkl")

st.title("Customer Churn Prediction Dashboard")
st.write("Enter customer details to predict churn probability.")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service,
}

input_df = pd.DataFrame([input_data])
input_df["Contract"] = input_df["Contract"].astype(str)
input_df["InternetService"] = input_df["InternetService"].astype(str)
input_df = pd.get_dummies(input_df)

if not hasattr(model, "feature_names_in_"):
    st.error("Model does not expose feature names. Re-train with a named DataFrame.")
    st.stop()

input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("Predict"):
    probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.write("Churn Probability:", round(probability, 3))

    if prediction == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")