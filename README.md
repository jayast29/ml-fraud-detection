# Fraud Detection System

An end-to-end MLOps pipeline for detecting fraudulent financial transactions using the PaySim synthetic dataset.

## Project Overview

This project builds a fraud detection system using machine learning and MLOps best practices including experiment tracking, data versioning, and model serving via REST API.

**Dataset:** PaySim — 6.3M synthetic mobile money transactions with 0.13% fraud rate.

---

## Project Structure

```
ml-fraud-detection/
├── data/
│   ├── raw_data.csv          # Original PaySim dataset
│   ├── data.csv              # Preprocessed (DVC tracked)
│   └── sample.csv            # Sample for development
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_logistic_regression.ipynb
│   ├── 02_random_forest.ipynb
│   └── 03_xgboost.ipynb
├── src/
│   ├── config.py             # AWS credentials loader
│   ├── versioning.py         # Data preprocessing + DVC
│   ├── train.py              # XGBoost training script
│   └── predict.py            # FastAPI prediction endpoint
├── models/
│   └── model.pkl             # Trained XGBoost model
├── dvc.yaml                  # DVC pipeline definition
├── dvc.lock                  # Pipeline state lock
└── .env                      # AWS credentials (not committed)
```

---

## ML Pipeline

```
raw_data.csv → versioning.py → data.csv → train.py → model.pkl → predict.py → API
```

---

## Model Comparison

| Model | ROC-AUC | Recall | Avg Precision |
|---|---|---|---|
| Logistic Regression | 0.976 | 0.88 | 0.54 |
| Random Forest | 0.962 | 0.78 | 0.87 |
| **XGBoost** | **0.998** | **0.96** | **0.85** |

XGBoost selected as final model — best recall and ROC-AUC for fraud detection.

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
```

---

## API Usage

Start the server:
```bash
python src/predict.py
```

Make a prediction:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"step": 1, "type": 2, "amount": 9839.64, "balance_diff_orig": 170136.0, "balance_diff_dest": 0.0}'
```

Response:
```json
{"fraud": true, "probability": 1.0}
```

---

## Experiment Tracking

All experiments tracked on DagsHub MLflow:
[https://dagshub.com/jayast29/ml-fraud-detection](https://dagshub.com/jayast29/ml-fraud-detection)