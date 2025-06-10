# 💳 Fraud Transaction Detection System

A machine learning system that detects whether a financial transaction is fraudulent or legitimate using behavioral features and time-series analysis.



## 📝 Description

This project builds an end-to-end pipeline to detect fraudulent transactions in real-time.  
It combines behavioral feature engineering, machine learning, and a Streamlit dashboard for live predictions.

Use case: early fraud detection in banking/fintech systems.




## ✨ Features

- Load and combine daily `.pkl` transaction data
- Feature engineering with behavioral indicators
- Train multiple models (XGBoost, Random Forest, etc.)
- Automatically pick and save the best model
- Real-time fraud prediction via web app
- Visual analysis: class balance, feature distribution, correlations



## ⚙️ Installation

```bash
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt
```

##🚀 Usage
#📦 Run full ML pipeline:
 ```bash
 python main.py
 ```

##🧠 Launch Streamlit prediction app:
```bash
streamlit run app.py
```

##🧰 Tech Stack
-Python

-pandas, scikit-learn, xgboost, lightgbm

-matplotlib, seaborn

-Streamlit for dashboard

-joblib for model serialization
