from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import pandas as pd
import joblib
import os

def train_model(df):
    df.fillna(0, inplace=True)
    
    features = [
        'TX_AMOUNT',
        'IS_HIGH_AMOUNT',
        'CUSTOMER_TX_COUNT_7D',
        'TERMINAL_FRAUD_COUNT_28D',
        'CUSTOMER_AVG_AMOUNT_7D',
        'AMOUNT_TO_AVG_RATIO'
    ]
    X = df[features]
    y = df['TX_FRAUD']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "LightGBM": LGBMClassifier(class_weight='balanced')
    }

    best_auc = 0
    best_model_name = None
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"{name} AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name

    os.makedirs("models", exist_ok=True)
    path = f"models/{best_model_name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(best_model, path)
    return best_model_name
