import pandas as pd
from tqdm import tqdm

def preprocess_transactions(df):
    df['IS_HIGH_AMOUNT'] = df['TX_AMOUNT'] > 220

    df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
    df['CUSTOMER_TX_COUNT_7D'] = 0
    for customer_id in tqdm(df['CUSTOMER_ID'].unique(), desc="Customer TX count"):
        cdf = df[df['CUSTOMER_ID'] == customer_id]
        tx_times = cdf['TX_DATETIME']
        counts = [
            tx_times[(tx_times > tx - pd.Timedelta(days=7)) & (tx_times <= tx)].count()
            for tx in tx_times
        ]
        df.loc[df['CUSTOMER_ID'] == customer_id, 'CUSTOMER_TX_COUNT_7D'] = counts

    df = df.sort_values(['TERMINAL_ID', 'TX_DATETIME'])
    df['TERMINAL_FRAUD_COUNT_28D'] = 0
    for term_id in tqdm(df['TERMINAL_ID'].unique(), desc="Terminal fraud count"):
        tdf = df[df['TERMINAL_ID'] == term_id]
        tx_times = tdf['TX_DATETIME']
        fraud_flags = tdf['TX_FRAUD']
        frauds = [
            fraud_flags[(tx_times > tx - pd.Timedelta(days=28)) & (tx_times <= tx)].sum()
            for tx in tx_times
        ]
        df.loc[df['TERMINAL_ID'] == term_id, 'TERMINAL_FRAUD_COUNT_28D'] = frauds

    df['CUSTOMER_AVG_AMOUNT_7D'] = 0.0
    for customer_id in tqdm(df['CUSTOMER_ID'].unique(), desc="Customer avg amount"):
        cdf = df[df['CUSTOMER_ID'] == customer_id]
        tx_times = cdf['TX_DATETIME']
        tx_amounts = cdf['TX_AMOUNT']
        avgs = [
            tx_amounts[(tx_times > tx - pd.Timedelta(days=7)) & (tx_times <= tx)].mean()
            if not tx_amounts[(tx_times > tx - pd.Timedelta(days=7)) & (tx_times <= tx)].empty else 0
            for tx in tx_times
        ]
        df.loc[df['CUSTOMER_ID'] == customer_id, 'CUSTOMER_AVG_AMOUNT_7D'] = avgs

    df['AMOUNT_TO_AVG_RATIO'] = df['TX_AMOUNT'] / (df['CUSTOMER_AVG_AMOUNT_7D'] + 1e-5)
    return df
