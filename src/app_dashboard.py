"""
Detector Environment Monitoring Dashboard
- Streams synthetic telemetry
- Predicts Stability Index (0–1)
- Live charts and alerts
"""

import os, time, sys, subprocess
import numpy as np
import pandas as pd
import joblib
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE, "..", "data", "env_telemetry.csv")
MODEL_FILE = os.path.join(BASE, "..", "data", "stability_model.pkl")

st.set_page_config(page_title="Detector Environment Monitoring", layout="wide")
st.title("🛰️ Detector Environment Monitoring Dashboard")
st.caption("Real-time environment telemetry • Stability prediction • Alerting")

# Ensure data + model exist
if not os.path.exists(DATA_FILE):
    st.warning("Generating telemetry…")
    subprocess.run([sys.executable, os.path.join(BASE, "data_generator.py")], check=True)
if not os.path.exists(MODEL_FILE):
    st.warning("Training stability model…")
    subprocess.run([sys.executable, os.path.join(BASE, "train_model.py")], check=True)

# Sidebar controls
st.sidebar.header("Controls")
refresh = st.sidebar.slider("Refresh interval (s)", 0.5, 5.0, 1.0, 0.5)
window = st.sidebar.slider("Rows per update", 1, 50, 10, 1)
show_table = st.sidebar.checkbox("Show raw data table", value=False)

# Load resources
df_all = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)

# Rolling simulation index
idx = 0
placeholder = st.empty()

# Feature list (must match training)
FEATURES = [
    "temperature_C", "humidity_%", "dose_rate_Gy_s", "magnetic_field_T",
    "vibration_level", "airflow_m_s", "particulate_ppm",
    "ps_voltage_V", "ps_current_A"
]

def status_color(si: float) -> str:
    if si >= 0.8: return "🟢 Stable"
    if si >= 0.6: return "🟡 Degrading"
    return "🔴 Critical"

while True:
    # Simulate live sliding window
    end = min(idx + window, len(df_all))
    if idx >= len(df_all):
        # loop over the dataset (or regenerate if you prefer)
        idx = 0
        end = window
    df = df_all.iloc[idx:end].copy()
    idx = end

    # Predict stability
    preds = model.predict(df[FEATURES])
    df["stability_index"] = preds
    avg_si = float(np.mean(preds))
    status = status_color(avg_si)

    with placeholder.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Stability Index", f"{avg_si:.3f}")
        c2.metric("Status", status)
        c3.metric("Dose Rate (median Gy/s)", f"{df['dose_rate_Gy_s'].median():.4f}")
        c4.metric("Temp (median °C)", f"{df['temperature_C'].median():.2f}")

        st.subheader("Live Trends")
        st.line_chart(df[["temperature_C", "humidity_%", "dose_rate_Gy_s"]])
        st.line_chart(df[["ps_voltage_V", "ps_current_A"]])

        colA, colB = st.columns(2)
        with colA:
            st.bar_chart(df[["vibration_level", "airflow_m_s", "particulate_ppm"]])
        with colB:
            st.area_chart(df[["magnetic_field_T", "stability_index"]])

        if show_table:
            st.subheader("Raw Telemetry (current window)")
            st.dataframe(df.reset_index(drop=True))

    time.sleep(float(refresh))
