# main.py

from src.data_loader import load_all_transaction_data
from src.preprocess import preprocess_transactions
from src.train import train_model
from src.predict import predict_transaction
import pandas as pd
import os

def main():
    print("📥 Step 1: Loading transaction data...")
    df = load_all_transaction_data("dataset/data")

    print("🧹 Step 2: Preprocessing & feature engineering...")
    df = preprocess_transactions(df)

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/engineered_transactions.csv", index=False)
    print("✅ Saved cleaned data to outputs/engineered_transactions.csv")

    print("🎯 Step 3: Training model...")
    best_model_name = train_model(df)
    print(f"✅ Best model trained and saved: {best_model_name}")

    print("🔍 Step 4: Running prediction on sample transaction...")

    sample_tx = {
        "TX_AMOUNT": 280,
        "IS_HIGH_AMOUNT": 1,
        "CUSTOMER_TX_COUNT_7D": 7,
        "TERMINAL_FRAUD_COUNT_28D": 3,
        "CUSTOMER_AVG_AMOUNT_7D": 40,
        "AMOUNT_TO_AVG_RATIO": 7.0
    }

    result = predict_transaction(sample_tx)
    print("🔮 Sample Transaction Prediction:", result)

if __name__ == "__main__":
    main()
