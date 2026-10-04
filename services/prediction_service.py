
import joblib
import numpy as np

model = joblib.load("model/kidney_model.pkl")
scaler = joblib.load("model/scaler.pkl")

def predict_data(data):
    # logic كامل هنا ✔️
    row = [float(v) for v in data.values()]
    X_scaled = scaler.transform([row])
    proba = model.predict_proba(X_scaled)[0]

    return {
        "prediction": int(proba[1] > 0.5),
        "confidence": float(max(proba)) * 100
    }