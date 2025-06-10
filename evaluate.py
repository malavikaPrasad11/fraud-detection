from src.predict import predict_transaction

# Sample input transaction
sample_tx = {
    "TX_AMOUNT": 300,
    "IS_HIGH_AMOUNT": 1,
    "CUSTOMER_TX_COUNT_7D": 5,
    "TERMINAL_FRAUD_COUNT_28D": 2,
    "CUSTOMER_AVG_AMOUNT_7D": 50,
    "AMOUNT_TO_AVG_RATIO": 6
}

result = predict_transaction(sample_tx)
print(result)
