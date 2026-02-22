import os
import logging
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(path: str):
    logger.info("Loading data...")
    df = pd.read_csv(path)
    X = df.drop(columns=['isFraud', 'isFlaggedFraud'])
    y = df['isFraud'].values
    return X, y

def split_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = RobustScaler()
    num_cols = X_train.select_dtypes(include=['number']).columns
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, y_train, y_test

def get_best_threshold(model, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_threshold = thresholds[f1_scores.argmax()]
    logger.info(f"Best threshold: {best_threshold:.4f}")
    return best_threshold

def train(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Training complete")
    return model

def evaluate(model, X_test, y_test, threshold):
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_scores):.4f}")
    logger.info(f"PR-AUC: {average_precision_score(y_test, y_scores):.4f}")

def save_model(artifact, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(artifact, path)
    logger.info(f"Model saved to {path}")

if __name__ == "__main__":
    X, y = load_data("data/data.csv")
    X_train, X_test, y_train, y_test = split_scale(X, y)
    model = train(X_train, y_train)
    threshold = get_best_threshold(model, X_test, y_test)
    evaluate(model, X_test, y_test, threshold)
    
    artifact = {
        "model": model,
        "threshold": threshold,
        "features": X_train.columns.tolist()
    }
    save_model(artifact, "models/model.pkl")