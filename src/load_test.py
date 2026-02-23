import requests
import random
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

APP_URL = "http://18.191.177.88:5000/predict"

def generate_transaction(fraud=False):
    if fraud:
        return {
            "step": random.randint(1, 100),
            "type": 4,
            "amount": random.uniform(100000, 500000),
            "balance_diff_orig": random.uniform(100000, 500000),
            "balance_diff_dest": 0
        }
    else:
        return {
            "step": random.randint(1, 100),
            "type": random.randint(0, 3),
            "amount": random.uniform(10, 5000),
            "balance_diff_orig": random.uniform(10, 5000),
            "balance_diff_dest": random.uniform(10, 5000)
        }

def run_load_test(total=1000, fraud_count=3):
    fraud_indices = random.sample(range(total), fraud_count)
    logger.info(f"Starting load test — {total} transactions, {fraud_count} fraud")
    
    for i in range(total):
        is_fraud = i in fraud_indices
        data = generate_transaction(fraud=is_fraud)
        try:
            response = requests.post(APP_URL, data=data)
            result = response.json()
            if result.get("fraud"):
                logger.info(f"Transaction {i+1} — FRAUD DETECTED | Probability: {result['probability']}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
        time.sleep(0.06)
    
    logger.info("Load test complete")

if __name__ == "__main__":
    run_load_test(total=1000, fraud_count=3)