import joblib
import numpy as np
import logging
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
model = joblib.load("models/model.pkl")

class Transaction(BaseModel):
    step: float
    type: float
    amount: float
    balance_diff_orig: float
    balance_diff_dest: float

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array([[
        transaction.step,
        transaction.type,
        transaction.amount,
        transaction.balance_diff_orig,
        transaction.balance_diff_dest
    ]])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    logger.info(f"Prediction: {prediction} | Probability: {probability:.4f}")
    return {
        "fraud": bool(prediction),
        "probability": round(float(probability), 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)