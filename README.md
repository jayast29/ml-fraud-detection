# Banking Fraud Detection System

An end-to-end MLOps pipeline for detecting fraudulent financial transactions using the PaySim synthetic dataset.

## Project Overview

This project builds a fraud detection system using machine learning and MLOps best practices including experiment tracking, data versioning, model serving via REST API, and real-time monitoring.

**Dataset:** PaySim — 6.3M synthetic mobile money transactions with 0.13% fraud rate.

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML | XGBoost, Scikit-learn |
| Experiment Tracking | MLflow, DagsHub |
| Data Versioning | DVC, AWS S3 |
| Model Serving | Flask |
| Monitoring | Prometheus, Grafana |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Cloud | AWS S3, ECR, EKS |

---

## Project Structure

```
ml-fraud-detection/
├── data/
│   ├── raw_data.csv                  # Original PaySim dataset
│   ├── data.csv                      # Preprocessed (DVC tracked)
│   └── sample.csv                    # Sample for development
├── notebooks/
│   ├── 00_eda_feature_engineering.ipynb
│   ├── 01_logistic_regression.ipynb
│   ├── 02_random_forest.ipynb
│   └── 03_xgboost.ipynb
├── src/
│   ├── config.py                     # AWS credentials loader
│   ├── versioning.py                 # Data preprocessing + DVC
│   ├── train.py                      # XGBoost training script
│   └── app.py                        # Flask app + Prometheus metrics
├── tests/
│   ├── test_app.py                   # Flask app tests
│   └── test_model.py                 # Model tests
├── models/
│   └── model.pkl                     # Trained XGBoost artifact
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI pipeline
├── dvc.yaml                          # DVC pipeline definition
├── dvc.lock                          # Pipeline state lock
├── Dockerfile                        # Docker image definition
├── docker-compose.yml                # Flask + Prometheus + Grafana
├── prometheus.yml                    # Prometheus scrape config
└── .env                              # AWS + DagsHub credentials (not committed)
```

---

## ML Pipeline

```
raw_data.csv → versioning.py → data.csv → train.py → model.pkl → app.py → API
```

Run full pipeline:
```bash
dvc repro
```

---

## Model Comparison

| Model | ROC-AUC | Recall | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.977 | 0.88 | 0.54 |
| Random Forest | 0.998 | 0.78 | 0.83 |
| **XGBoost (tuned)** | **0.998** | **0.96** | **0.86** |

XGBoost selected as final model with optimal decision threshold of 0.9084 for best precision-recall balance.

---

## Fraud Detection Pattern

Fraud in this dataset follows a specific pattern:
- Transaction type: TRANSFER or CASH_OUT
- Origin account balance completely wiped out
- Destination balance unchanged after transaction

---

## Setup

```bash
# Clone repo
git clone https://github.com/jayast29/ml-fraud-detection
cd ml-fraud-detection

# Create environment
conda create -n mlops python=3.10
conda activate mlops

# Install dependencies
pip install -r requirements.txt

# Add credentials
cp .env.example .env
# Fill in AWS and DagsHub credentials

# Pull data from S3
dvc pull

# Run pipeline
dvc repro

# Start app
python src/app.py
```

---


## Monitoring

Prometheus metrics available at `http://localhost:5000/metrics`:
- `fraud_request_total` — total predictions
- `fraud_detected_total` — total fraud detected
- `legit_detected_total` — total legitimate transactions
- `high_risk_total` — transactions with probability > 0.8
- `fraud_probability_avg` — average fraud probability
- `fraud_request_latency_seconds` — response time

---

## CI/CD

GitHub Actions runs on every push to `main`:
1. Install dependencies
2. Run tests (`pytest tests/ -v`)

---

## Experiment Tracking

All experiments tracked on DagsHub MLflow:
[https://dagshub.com/jayast29/ml-fraud-detection](https://dagshub.com/jayast29/ml-fraud-detection)