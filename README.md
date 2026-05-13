# TrafficADS v3 — Traffic Anomaly Detection System

A machine-learning pipeline for urban traffic congestion prediction, pothole detection, landmark recognition, and route optimization.

## Features

- **Congestion Prediction** — Trains Random Forest, Gradient Boosting, and Logistic Regression models on 50K rows of traffic data; saves the best model by AUC
- **Dataset Generation** — Synthetic traffic, weather, POI, pothole, incident, and road-network datasets
- **Interactive Dashboard** — Streamlit/Flask dashboard (`dashboard.py`) for visualizing results
- **End-to-End Pipeline** — Single `run_all.py` script to generate data → train → detect → optimize

## Project Structure

```
TrafficADS_v3/
├── run_all.py              # Full pipeline runner
├── generate_dataset.py     # Synthetic data generation
├── traffic_model.py        # Congestion prediction model (v2.0)
├── dashboard.py            # Visualization dashboard
├── requirements.txt
├── data/                   # Generated at runtime (gitignored)
│   ├── traffic_data.csv
│   ├── weather_data.csv
│   ├── poi_data.csv
│   ├── pothole_data.csv
│   ├── incidents_data.csv
│   ├── road_network.csv
│   └── road_images/
├── models/                 # Saved model artifacts (gitignored)
│   └── congestion_model.pkl
└── outputs/                # Generated plots (gitignored)
```

## Setup

```bash
git clone https://github.com/<your-username>/TrafficADS_v3.git
cd TrafficADS_v3
pip install -r requirements.txt
```

## Usage

**Run the full pipeline:**
```bash
python run_all.py
```

**Run individual steps:**
```bash
python generate_dataset.py   # Generate synthetic data
python traffic_model.py      # Train congestion model
python dashboard.py          # Launch dashboard
```

## Model Performance

The congestion model compares three classifiers and saves the best one (by ROC-AUC):

| Model | Notes |
|---|---|
| Random Forest | 150 estimators, parallel fit |
| Gradient Boosting | 100 estimators |
| Logistic Regression | Scaled features, 500 iterations |

Trained model is saved to `models/congestion_model.pkl`.

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
scipy
flask
opencv-python
Pillow
```

## License

MIT
