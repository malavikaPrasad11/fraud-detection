import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load("models/xgboost_model.pkl")

# Define feature names
FEATURES = [
    'TX_AMOUNT',
    'IS_HIGH_AMOUNT',
    'CUSTOMER_TX_COUNT_7D',
    'TERMINAL_FRAUD_COUNT_28D',
    'CUSTOMER_AVG_AMOUNT_7D',
    'AMOUNT_TO_AVG_RATIO'
]

st.title("💳 Fraud Transaction Detector")
st.write("Enter transaction details to check if it's fraudulent.")

# User input fields
tx_amount = st.slider("Transaction Amount", 0, 1000, 100)
is_high_amount = int(tx_amount > 220)

customer_tx_7d = st.slider("Customer's Past 7-Day TX Count", 0, 50, 5)
terminal_fraud_28d = st.slider("Terminal Fraud Count (28D)", 0, 20, 2)
customer_avg_7d = st.slider("Customer Avg Amount (7D)", 0, 500, 60)

# Auto-calculated ratio
amount_to_avg_ratio = round(tx_amount / (customer_avg_7d + 1e-5), 2) if customer_avg_7d else 0

#  Show calculated field before prediction
st.markdown(f"**💡 Amount to Avg Ratio:** `{amount_to_avg_ratio}`")

# Prepare input for model
input_dict = {
    "TX_AMOUNT": tx_amount,
    "IS_HIGH_AMOUNT": is_high_amount,
    "CUSTOMER_TX_COUNT_7D": customer_tx_7d,
    "TERMINAL_FRAUD_COUNT_28D": terminal_fraud_28d,
    "CUSTOMER_AVG_AMOUNT_7D": customer_avg_7d,
    "AMOUNT_TO_AVG_RATIO": amount_to_avg_ratio
}

df = pd.DataFrame([input_dict])

# Prediction
if st.button("🚨 Predict Fraud"):
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    label = "Fraud" if pred == 1 else "Legit"
    color = "red" if pred == 1 else "green"

    st.markdown(
        f"<h3 style='color:{color};'>🧾 Result: {label}</h3>", 
        unsafe_allow_html=True
    )
    st.markdown(f"**Confidence:** `{round(prob * 100, 2)}%`")
    st.write("### 🔍 Transaction Details")
    st.dataframe(df)
