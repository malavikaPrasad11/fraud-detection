import joblib
import pandas as pd
import numpy as np

FEATURES = [
    'TX_AMOUNT',
    'IS_HIGH_AMOUNT',
    'CUSTOMER_TX_COUNT_7D',
    'TERMINAL_FRAUD_COUNT_28D',
    'CUSTOMER_AVG_AMOUNT_7D',
    'AMOUNT_TO_AVG_RATIO'
]

# Load the trained model
model = joblib.load("models/xgboost_model.pkl")

def predict_transaction(transaction_dict):
    """
    Predicts if a transaction is fraudulent.
    Input: dict with transaction features
    Output: prediction label (0/1) and fraud probability
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([transaction_dict])
    
    # Ensure all features are present and in correct order
    for col in FEATURES:
        if col not in df:
            df[col] = 0.0  # Default if missing

    df = df[FEATURES]

    # Predict
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    
    label = "Fraud" if pred == 1 else "Legit"
    
    return {
        "prediction": int(pred),
        "label": label,
        "probability": round(proba, 4)
    }

