# 🌍 SenseSmog

### Air Quality & Health Impact Prediction System

SenseSmog is a machine-learning based application that predicts the potential health impact of air pollution using air-quality, weather, and health-related parameters. The project combines a trained neural-network model with a simple Streamlit interface so users can enter environmental data and receive an estimated health-impact category.

## ✨ Features

* Predicts health-impact category from air-quality and environmental conditions.
* Uses AQI and pollutant measurements including PM10, PM2.5, NO2, SO2, and O3.
* Considers temperature, humidity, and wind speed.
* Includes respiratory cases, cardiovascular cases, and hospital admissions as model inputs.
* Displays the predicted health-impact category and class probabilities.
* Provides an interactive Streamlit web interface.

## 🧠 Health Impact Categories

| Class | Category            | Score Range     |
| ----- | ------------------- | --------------- |
| 0     | ⭐ Very Low Impact   | Score ≥ 80      |
| 1     | 🟢 Low Impact       | 60 ≤ Score < 80 |
| 2     | 🟡 Moderate Impact  | 40 ≤ Score < 60 |
| 3     | 🟠 High Impact      | 20 ≤ Score < 40 |
| 4     | 🔴 Very High Impact | Score < 20      |

## 🏗️ Project Structure

```text
SenseSmog/
├── app.py
├── model.pkl
├── scaler.pkl
├── air_quality_health_impact_data.csv
├── AQI (5).ipynb
├── AQI (6).ipynb
├── requirements.txt
└── README.md
```

## ⚙️ How It Works

1. The user enters air-quality, weather, and health-related parameters.
2. The application combines the inputs into the feature vector expected by the trained model.
3. The saved scaler transforms the input data.
4. The trained model predicts probabilities for the five health-impact classes.
5. The class with the highest predicted probability is selected.
6. SenseSmog displays the predicted health-impact category and model probabilities.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/agamya1709/SenseSmog.git
cd SenseSmog
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project uses Streamlit, NumPy, Pandas, Scikit-learn, TensorFlow, and Keras.

### 4. Run the application

```bash
streamlit run app.py
```

The application loads `model.pkl` and `scaler.pkl` from the project directory before making predictions.

## 📊 Input Parameters

The application accepts:

* AQI
* PM10
* PM2.5
* NO2
* SO2
* Ozone (O3)
* Temperature (°C)
* Humidity (%)
* Wind Speed (m/s)
* Respiratory Cases
* Cardiovascular Cases
* Hospital Admissions

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy**
* **Scikit-learn**
* **TensorFlow / Keras**
* **Streamlit**
* **Jupyter Notebook**

## 📁 Dataset

The repository contains `air_quality_health_impact_data.csv`, which is used for data analysis and machine-learning development.

## ⚠️ Disclaimer

SenseSmog is an educational and predictive machine-learning project. Its predictions are estimates and should not be treated as medical advice, clinical diagnosis, or a replacement for professional healthcare guidance.

## 🔮 Future Improvements

* Integrate real-time AQI and weather data.
* Add location-based air-quality monitoring.
* Improve model performance with additional datasets.
* Add historical trend visualizations.
* Provide personalized health recommendations.
* Deploy the application for public access.

## 👥 Project

SenseSmog is a machine-learning project focused on understanding the relationship between air pollution and potential health impacts.
