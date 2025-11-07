"""
Generates synthetic detector-hall environment telemetry.
Columns are stable and reused by the model and dashboard.
"""

import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)

def generate(samples: int = 5000, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    time_s = np.arange(samples)

    # Core environmental signals (base realistic ranges)
    temperature_C     = rng.normal(24.5, 2.0, samples)         # °C
    humidity_pct      = rng.uniform(30, 55, samples)           # %
    dose_rate_Gy_s    = rng.normal(0.02, 0.01, samples).clip(0, None)  # Gy/s
    magnetic_field_T  = rng.normal(0.4, 0.05, samples).clip(0, None)   # Tesla near magnets
    vibration_level   = rng.uniform(0.0, 1.0, samples)         # normalized 0–1
    airflow_m_s       = rng.normal(2.5, 0.5, samples).clip(0, None)    # m/s HVAC
    particulate_ppm   = rng.normal(12, 4, samples).clip(0, None)       # ppm
    ps_voltage_V      = rng.normal(12.0, 0.2, samples)         # power rail
    ps_current_A      = rng.normal(3.0, 0.3, samples).clip(0, None)

    df = pd.DataFrame({
        "time_s": time_s,
        "temperature_C": temperature_C,
        "humidity_%": humidity_pct,
        "dose_rate_Gy_s": dose_rate_Gy_s,
        "magnetic_field_T": magnetic_field_T,
        "vibration_level": vibration_level,
        "airflow_m_s": airflow_m_s,
        "particulate_ppm": particulate_ppm,
        "ps_voltage_V": ps_voltage_V,
        "ps_current_A": ps_current_A,
    })

    out = "data/env_telemetry.csv"
    df.to_csv(out, index=False)
    print(f"✅ Generated: {out}  (rows: {len(df)})")
    return out

if __name__ == "__main__":
    generate()
