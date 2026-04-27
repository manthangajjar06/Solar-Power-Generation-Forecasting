# Solar Power Generation Forecasting

> A machine learning web application that predicts solar DC power output based on real-time environmental conditions — irradiation, module temperature, and ambient temperature — using a Random Forest Regressor trained on Kaggle's Solar Plant 1 dataset.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technical Architecture](#technical-architecture)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)

---

## Overview

Solar energy generation is highly dependent on weather conditions that vary throughout the day and across seasons. Accurate short-term power forecasting is critical for grid operators, energy traders, and solar plant managers to balance supply with demand, reduce curtailment, and optimize storage usage.

This project builds a solar DC power output forecasting system using real sensor data from a commercial solar plant in India. Two models are developed and compared — a Linear Regression baseline and a tuned Random Forest Regressor. The final model is served through a Streamlit web interface where users can input environmental readings and receive an instant DC power prediction.

---

## Live Demo

> 🚀 [Launch App on Streamlit Cloud](https://solar-power-generation-forecasting-jikb5jn7otdlxzbzu75kg9.streamlit.app/)

---

## Features

- **Real-time DC power prediction** — Input irradiation, module temperature, and ambient temperature to get an instant DC power forecast in kilowatts
- **Time-aware prediction** — Users provide a date and time; the app extracts hour, day, month, and day-of-week features automatically to capture diurnal and seasonal patterns
- **Dual model development** — Notebook compares Linear Regression (baseline) vs Random Forest (final model) with full evaluation metrics
- **Clean Streamlit UI** — Minimal, focused interface built for operational use

---

## Project Structure

```
Solar-Power-Generation-Forecasting/
│
├── app.py                                           # Streamlit application — main entry point
├── Solar Power Generation Forecast.ipynb            # End-to-end ML pipeline notebook
│
├── data/
│   ├── Plant_1_Generation_Data.csv                  # Raw inverter-level generation data
│   ├── Plant_1_Weather_Sensor_Data.csv              # Raw weather sensor readings
│   └── Plant1_Merged_Dataset.csv                    # Merged & aggregated dataset (generated)
│
├── model/
│   ├── solar_Power_eneration_Forecasting_model.pkl  # Final Random Forest model (serialized)
│   └── scaler.pkl                                   # Fitted StandardScaler (serialized)
│
├── requirements.txt                                 # Python dependencies
└── README.md
```

> **Note:** Files are stored flat at the repository root. The folder structure above reflects logical grouping for clarity.

---

## Technical Architecture

```
Raw Data
(Generation CSV + Weather CSV)
          │
          ▼
┌──────────────────────────────────────────────┐
│             Data Merging                     │
│  • Aggregate generation data by DATE_TIME    │
│  • Aggregate weather data by DATE_TIME       │
│  • Inner join both on DATE_TIME              │
│  • Save as Plant1_Merged_Dataset.csv         │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│           Data Preprocessing                 │
│  • DateTime parsing & indexing               │
│  • Outlier removal via IQR capping           │
│  • Filter daytime records (IRRADIATION > 0)  │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│          Feature Engineering                 │
│  • Extract: HOUR, DAY, MONTH, DAY_OF_WEEK    │
│  • Feature scaling: StandardScaler           │
│  • Target: DC_POWER                          │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│        Model Training & Comparison           │
│  • Baseline: Linear Regression               │
│  • Final:    Random Forest Regressor         │
│  • Metrics:  MSE, R² Score                   │
│  • Split:    80/20 random split              │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│         Serialization                        │
│  • joblib.dump → .pkl model + scaler         │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
         Streamlit Web Application
    (loads model + scaler, transforms input,
         returns DC power prediction)
```

---

## Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) |
| Plant | Solar Plant 1, India |
| Files | `Plant_1_Generation_Data.csv` + `Plant_1_Weather_Sensor_Data.csv` |
| Generation records | Inverter-level DC/AC power readings at 15-minute intervals |
| Weather records | Ambient temperature, module temperature, irradiation at sensor level |
| Merge key | `DATE_TIME` — both datasets aggregated then joined |
| Target variable | `DC_POWER` — aggregated DC output in kW |

**Key preprocessing decisions:**

- Generation and weather data were independently aggregated by `DATE_TIME` (sum for power, mean for temperatures and irradiation) before merging to avoid row duplication from multiple inverters and sensors
- Outliers were capped using IQR bounds rather than dropped to preserve time continuity
- Only daytime records (`IRRADIATION > 0`) were used for model training since nighttime zero-power records would create a trivial learned pattern that does not generalize to real forecasting scenarios
- Correlation analysis confirmed `IRRADIATION` as the strongest predictor of DC power output, followed by `MODULE_TEMPERATURE`

---

## Model Details

### Model Comparison

| Model | MSE | R² Score | Notes |
|---|---|---|---|
| Linear Regression | Higher | Lower | Baseline — cannot capture non-linear irradiation-power relationship |
| Random Forest Regressor | Lower | Higher | Final model — handles non-linearity and feature interactions |

Random Forest was selected as the final model because the relationship between irradiation and DC power output is non-linear — power generation saturates at high irradiation levels and is modulated by temperature effects that interact non-additively.

### Feature Set

| Feature | Type | Description |
|---|---|---|
| `IRRADIATION` | Continuous | Solar irradiation at sensor level (W/m²) |
| `MODULE_TEMPERATURE` | Continuous | Solar panel surface temperature (°C) |
| `AMBIENT_TEMPERATURE` | Continuous | Air temperature at plant level (°C) |
| `HOUR` | Cyclic (int) | Hour of day extracted from DATE_TIME |
| `DAY` | Integer | Day of month |
| `MONTH` | Integer | Month of year |
| `DAY_OF_WEEK` | Integer | Weekday index (0=Monday, 6=Sunday) |

### Preprocessing Pipeline

All features are standardized using `StandardScaler` fitted on the training set. The fitted scaler is serialized alongside the model (`scaler.pkl`) so that inference inputs are transformed identically to how training data was prepared — preventing train/serve skew.

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Solar-Power-Generation-Forecasting.git
cd Solar-Power-Generation-Forecasting

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### Dependencies

```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
```

---

## Usage

1. Open the app in your browser
2. Enter a **Date and Time** in the format `YYYY-MM-DD HH:MM` (e.g. `2020-06-15 14:00`)
3. Enter the current **Irradiation** reading from the sensor (W/m²)
4. Enter the **Module Temperature** — solar panel surface temperature in °C
5. Enter the **Ambient Temperature** — surrounding air temperature in °C
6. Click **"Predict DC Power"**
7. The app displays the predicted DC power output in kilowatts

**Example input:**

| Field | Value |
|---|---|
| Date and Time | 2020-06-15 14:00 |
| Irradiation | 0.85 |
| Module Temperature | 48.5 |
| Ambient Temperature | 32.0 |

---

## Results

The Random Forest model significantly outperforms the Linear Regression baseline on the held-out 20% test set, confirming that non-linear feature interactions — particularly between irradiation, temperature, and time of day — are important for accurate DC power forecasting.

The model performs best during peak daytime hours (10:00–16:00) where irradiation patterns are stable and more training data exists. Prediction accuracy is lower during dawn and dusk transition periods where small changes in irradiation produce larger relative variance in power output.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web framework | Streamlit |
| ML library | scikit-learn |
| Data processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model serialization | joblib |
| Notebook environment | Jupyter |

---

## Future Improvements

- **Multi-step forecasting** — Extend from single-point prediction to a 24-hour ahead forecast using a rolling window approach
- **AC power prediction** — Add a second output to simultaneously forecast AC power alongside DC, since AC output depends on inverter efficiency which varies with load
- **Plant 2 generalization** — Train and evaluate on Plant 2 data to test model generalizability across different solar installations
- **Real-time sensor integration** — Connect to live sensor API feeds to automatically populate inputs rather than requiring manual entry
- **Confidence intervals** — Add prediction intervals using quantile regression forests to communicate forecast uncertainty to operators
- **Streamlit dashboard upgrade** — Add historical trend visualization, daily generation summary, and a model performance metrics panel
- **Automated retraining pipeline** — Schedule retraining with new sensor data using GitHub Actions as more plant data accumulates

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Dataset sourced from [Kaggle — Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
- Built with [Streamlit](https://streamlit.io/) — open-source ML app framework
- Inspired by real-world solar plant operations in India
