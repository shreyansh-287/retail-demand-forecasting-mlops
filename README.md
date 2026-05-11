# 📊 Retail Demand Forecasting MLOps System

An end-to-end machine learning system for forecasting retail sales using time-series modeling, deployed via FastAPI with built-in logging and data drift monitoring.

---

## 🚀 Project Overview

This project predicts weekly retail sales at a Store × Time level using historical demand patterns and external economic indicators. The system is designed not just for modeling, but for real-world usage with deployment, monitoring, and reproducibility.

---

## 🎯 Objectives

- Forecast weekly sales using historical and contextual data  
- Capture temporal patterns using lag and rolling features  
- Compare multiple ML models and select the best-performing one  
- Deploy the model as a REST API for real-time predictions  
- Implement logging and monitoring for production reliability  

---

## 🧠 Modeling Approach

### 🔹 Feature Engineering

- Lag features: `lag_1`, `lag_4`, `lag_12`, `lag_52`  
- Rolling averages: `rolling_mean_4`, `rolling_mean_12`  
- Time features: Year, Month, Week  
- External features: CPI, Temperature, Fuel Price, Unemployment  

---

### 🔹 Model Comparison

The following models were evaluated:

- Baseline (Naive lag-based)  
- Ridge Regression  
- Random Forest  
- XGBoost (final model)  

XGBoost was selected after hyperparameter tuning using **TimeSeriesSplit + RandomizedSearchCV**, ensuring robust performance on future unseen data.

---

### 🔹 Evaluation Strategy

- Time-based train-test split (to avoid data leakage)  
- Metrics used:
  - MAE (Mean Absolute Error)  
  - RMSE (Root Mean Squared Error)  
  - R² Score  

---

## ⚙️ System Architecture

```
Data → Feature Engineering → Model Training → API (FastAPI)
                                         ↓
                                      Logging
                                         ↓
                                  Drift Monitoring
```

---

## 🔌 API Deployment

The trained model is deployed using **FastAPI**.

### 🔹 Run API

```
uvicorn app:app --reload
```

---

### 🔹 Endpoint

`POST /predict`

---

### 🔹 Sample Input

```
{
  "Store": 1,
  "Holiday_Flag": 0,
  "Temperature": 45,
  "Fuel_Price": 2.5,
  "CPI": 210,
  "Unemployment": 8.1,
  "Year": 2012,
  "Month": 1,
  "Week": 5,
  "lag_1": 1500000,
  "lag_4": 1400000,
  "lag_12": 1300000,
  "lag_52": 1600000,
  "rolling_mean_4": 1450000,
  "rolling_mean_12": 1350000
}
```

---

### 🔹 Sample Output

```
{
  "predicted_sales": 1520000,
  "drift_detected": false,
  "drift_features": []
}
```

---

## 📈 MLOps Features

### ✅ Logging

- Stores input features, predictions, and timestamps  
- Helps in debugging and tracking model behavior  

---

### ✅ Data Drift Monitoring

- Compares incoming inputs with training distribution  
- Flags anomalies using statistical thresholds (z-score)  

---

### ✅ Reproducibility

- Model saved using `joblib`  
- Training statistics stored for monitoring  

---

## 📁 Project Structure

```
retail-demand-forecasting-mlops/
│
├── app.py
├── training_stats.json
├── requirements.txt
├── README.md
├── notebooks/
│   └── model_training.ipynb
```

---

## 🛠️ Tech Stack

- Python  
- XGBoost  
- Scikit-learn  
- FastAPI  
- Pandas / NumPy  

---

## 💡 Key Learnings

- Handling time-series data using lag and rolling features  
- Avoiding data leakage with time-based validation  
- Hyperparameter tuning using TimeSeriesSplit  
- Deploying ML models via FastAPI  
- Implementing logging and monitoring for production systems  

---

## 📌 Future Improvements

- Automated retraining pipeline  
- Monitoring dashboard  
- Cloud deployment (AWS / GCP / Render)  

---

## 👤 Author

Shreyansh Pathak
