import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os


# DEBUG - check working directory
st.write("Current directory:", os.getcwd())
st.write("Files here:", os.listdir())

# Define absolute file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# Check if files exist
if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    st.error("model.pkl OR scaler.pkl not found. Please upload them in the same folder.")
else:
    st.success("Files found successfully!")

# Load model and scaler safely
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)



st.write("Current directory:", os.getcwd())
st.write("Files here:", os.listdir())

# -----------------------------
# Load Model & Scaler
# -----------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except:
    st.error("❌ model.pkl OR scaler.pkl not found. Please upload them in the same folder.")
    st.stop()

# -----------------------------
# Health Impact Mapping
# -----------------------------
health_impact_class_map = {
    0: "⭐ Very Low Impact (Health Impact Score ≥ 80)",
    1: "🟢 Low Impact (60 ≤ Score < 80)",
    2: "🟡 Moderate Impact (40 ≤ Score < 60)",
    3: "🟠 High Impact (20 ≤ Score < 40)",
    4: "🔴 Very High Impact (Score < 20)"
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌍 Air Quality & Health Impact Prediction App")
st.write("Enter air quality parameters below to predict the health risk category.")

st.header("📌 Enter Air Quality Parameters")

# Layout in columns for cleaner UI
col1, col2 = st.columns(2)

with col1:
    aqi = st.number_input("AQI", min_value=0.0)
    pm10 = st.number_input("PM10", min_value=0.0)
    pm25 = st.number_input("PM2.5", min_value=0.0)
    no2 = st.number_input("NO2", min_value=0.0)
    so2 = st.number_input("SO2", min_value=0.0)
    o3 = st.number_input("Ozone (O3)", min_value=0.0)

with col2:
    temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0)
    resp_cases = st.number_input("Respiratory Cases", min_value=0.0)
    cardio_cases = st.number_input("Cardiovascular Cases", min_value=0.0)
    hospital_admissions = st.number_input("Hospital Admissions", min_value=0.0)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Health Impact"):
    user_input = np.array([[aqi, pm10, pm25, no2, so2, o3, temp, humidity,
                            wind_speed, resp_cases, cardio_cases, hospital_admissions]])

    # Scale data
    scaled_input = scaler.transform(user_input)

    # Predict
    prediction_probabilities = model.predict(scaled_input)
    predicted_class = int(np.argmax(prediction_probabilities))

    # Result text
    result_text = health_impact_class_map.get(predicted_class, "Unknown Class")

    st.success("### 🧾 Prediction Result:")
    st.write(f"**Predicted Class Index:** {predicted_class}")
    st.write(f"**Health Impact Category:** {result_text}")

    # Show probabilities
    st.subheader("📊 Class Probabilities")
    st.write(prediction_probabilities)


