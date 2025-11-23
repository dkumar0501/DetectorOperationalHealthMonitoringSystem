<h1 align="left">Detector Operational Health Monitoring System</h1> 
  
<p align="left">    
  <strong>Detector Environment • Python • Machine Learning • Real Time Monitoring • Operational Health</strong> 
</p>         
       
<!-- Badges -->             
<p align="left">        
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">  
  <img src="https://img.shields.io/badge/Machine%20Learning-Enabled-orange?logo=scikit-learn&logoColor=white" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-success?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Real%20Time-Environment%20Telemetry-9cf?logo=anaconda&logoColor=white" alt="Real Time">
</p>



## 📘 Overview

This project is a **Python based real time monitoring and analysis system** designed to assess the **operational health of particle detectors** through environment driven telemetry. It simulates multiple parameters such as **temperature, humidity, dose rate, magnetic field, vibration, airflow, and power stability**, and uses a **machine learning model** to predict a **Stability Index (0–1)** that reflects detector health. The system visualizes these signals through a **Streamlit based interactive dashboard**, enabling early detection of performance degradation during long-term radiation or beamline operations.



## 🚀 Features

- Real time monitoring of detector environment parameters  
- Machine learning–based **Stability Index** prediction  
- Interactive dashboard with continuous visual updates  
- Synthetic data generation pipeline for controlled experimentation  
- Multi-parameter analysis and dynamic health visualization  
- Automated model training and adaptive telemetry handling  



## 🧠 Technical Overview

| Component | Description |
|------------|-------------|
| **Programming Language** | Python (NumPy, Pandas, Scikit-learn, Streamlit) |
| **Simulation Engine** | Generates synthetic telemetry for temperature, radiation, vibration, magnetic field, and power health |
| **Modeling** | Random Forest regression to predict detector stability and health |
| **Visualization** | Real time trend charts, bar plots, and stability indicators |
| **Output** | Stability Index [0–1], health status alerts, and live plots |



## 📷 Project Live Working

https://github.com/user-attachments/assets/9c92a180-4b80-4d61-a83b-b99b181a6217

## 🧩 Future Enhancements

- Integration with **real detector data streams** (via MQTT / InfluxDB)  
- Addition of **automated anomaly alerting** (email / Slack / Webhooks)  
- Deployment on **Docker or AWS EC2** for remote monitoring setups  
- Inclusion of **dose-rate compensation** and **temperature drift correction**  
- Integration with **sensor-level fault diagnostics**



## 👨‍💻 Author

**Developed by [D Kumar](https://github.com/dkumar0501)** — **[IIT Patna]**

---
<p align="center">
  <em>“Enabling intelligent, real time insight into detector operational health.”</em>
</p>
