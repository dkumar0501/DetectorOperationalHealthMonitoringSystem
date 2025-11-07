"""
Trains a lightweight model to estimate a 0–1 Stability Index
from environment telemetry (higher = more stable).
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

DATA = "data/env_telemetry.csv"
MODEL = "data/stability_model.pkl"

if not os.path.exists(DATA):
    raise FileNotFoundError("Missing data/env_telemetry.csv. Run: python src/data_generator.py")

df = pd.read_csv(DATA)

# Synthetic stability law (domain-informed):
# Penalize high dose rate, high temp, high humidity, high vibration, low airflow, high particulate,
# large magnetic field, and PSU anomalies (voltage droop + current spikes).
stability = (
    1.0
    - 0.6 * (df["dose_rate_Gy_s"] / (0.05 + 1e-9))**0.5
    - 0.01 * (df["temperature_C"] - 24.0).clip(lower=0)
    - 0.005 * (df["humidity_%"] - 40).clip(lower=0)
    - 0.25 * df["vibration_level"]
    - 0.04 * (df["particulate_ppm"] / 25.0)
    - 0.05 * (abs(df["magnetic_field_T"] - 0.4) / 0.2)
    - 0.05 * (abs(df["ps_voltage_V"] - 12.0) / 0.5)
    - 0.03 * (abs(df["ps_current_A"] - 3.0) / 1.0)
    - 0.03 * ((2.2 - df["airflow_m_s"]).clip(lower=0))  # penalize low airflow
).clip(0, 1)

df["stability_index"] = stability

features = [
    "temperature_C", "humidity_%", "dose_rate_Gy_s", "magnetic_field_T",
    "vibration_level", "airflow_m_s", "particulate_ppm",
    "ps_voltage_V", "ps_current_A"
]
X = df[features]
y = df["stability_index"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_tr, y_tr)
r2 = model.score(X_te, y_te)
print(f"✅ Trained Stability model. R²: {r2:.3f}")

os.makedirs("data", exist_ok=True)
joblib.dump(model, MODEL)
print(f"💾 Saved model → {MODEL}")
