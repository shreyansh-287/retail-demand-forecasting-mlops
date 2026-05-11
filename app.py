from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

app = FastAPI()

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("model.pkl")

# -------------------------------
# Load Training Stats for Drift
# -------------------------------
with open("training_stats.json", "r") as f:
    training_stats = json.load(f)

# -------------------------------
# Log file
# -------------------------------
LOG_FILE = "predictions_log.csv"

# -------------------------------
# Drift Detection Function
# -------------------------------
def check_drift(input_data):
    drift_flags = []

    for key in input_data:
        if key in training_stats:
            mean = training_stats[key]['mean']
            std = training_stats[key]['std']

            if std == 0:
                continue

            z_score = abs((input_data[key] - mean) / std)

            if z_score > 3:
                drift_flags.append(key)

    return drift_flags


# -------------------------------
# Root Endpoint
# -------------------------------
@app.get("/")
def home():
    return {"message": "Retail Sales Prediction API is running"}


# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
def predict(data: dict):

    # Feature order MUST match training
    features = [
        data['Store'],
        data['Holiday_Flag'],
        data['Temperature'],
        data['Fuel_Price'],
        data['CPI'],
        data['Unemployment'],
        data['Year'],
        data['Month'],
        data['Week'],
        data['lag_1'],
        data['lag_4'],
        data['lag_12'],
        data['lag_52'],
        data['rolling_mean_4'],
        data['rolling_mean_12']
    ]

    features_array = np.array(features).reshape(1, -1)

    prediction = model.predict(features_array)[0]

    # -------------------------------
    # Drift Detection
    # -------------------------------
    drift_features = check_drift(data)
    drift_detected = len(drift_features) > 0

    # -------------------------------
    # Logging
    # -------------------------------
    log_data = data.copy()
    log_data['prediction'] = float(prediction)
    log_data['drift_detected'] = drift_detected
    log_data['drift_features'] = str(drift_features)
    log_data['timestamp'] = datetime.now()

    log_df = pd.DataFrame([log_data])

    if not os.path.exists(LOG_FILE):
        log_df.to_csv(LOG_FILE, index=False)
    else:
        log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)

    # -------------------------------
    # Response
    # -------------------------------
    return {
        "predicted_sales": float(prediction),
        "drift_detected": drift_detected,
        "drift_features": drift_features
    }