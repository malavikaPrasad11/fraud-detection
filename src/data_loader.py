import os
import pandas as pd

def load_all_transaction_data(data_folder="dataset/data"):
    pkl_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.pkl')])
    df_list = [pd.read_pickle(os.path.join(data_folder, f)) for f in pkl_files]
    df = pd.concat(df_list, ignore_index=True)
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    df = df.sort_values('TX_DATETIME').reset_index(drop=True)
    return df
