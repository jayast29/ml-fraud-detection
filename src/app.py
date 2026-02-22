from flask import Flask, render_template, request, Response
import joblib
import numpy as np
import logging
import os
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
model = joblib.load(os.path.join(os.path.dirname(__file__), '../models/model.pkl'))

# Prometheus metrics
REQUEST_COUNT = Counter('fraud_request_total', 'Total prediction requests')
FRAUD_COUNT = Counter('fraud_detected_total', 'Total fraud predictions')
LEGIT_COUNT = Counter('legit_detected_total', 'Total legitimate predictions')
REQUEST_LATENCY = Histogram('fraud_request_latency_seconds', 'Request latency')
HIGH_RISK_COUNT = Counter('high_risk_total', 'Transactions with probability > 0.8')
PREDICTION_ERRORS = Counter('prediction_errors_total', 'Failed predictions')
AVG_FRAUD_PROB = Gauge('fraud_probability_avg', 'Average fraud probability')

prob_sum = 0
pred_count = 0

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    global prob_sum, pred_count
    start = time.time()
    REQUEST_COUNT.inc()

    try:
        data = np.array([[
            float(request.form["step"]),
            float(request.form["type"]),
            float(request.form["amount"]),
            float(request.form["balance_diff_orig"]),
            float(request.form["balance_diff_dest"])
        ]])

        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        result = {
            "fraud": bool(prediction),
            "probability": round(float(probability), 4)
        }

        if result["fraud"]:
            FRAUD_COUNT.inc()
        else:
            LEGIT_COUNT.inc()

        if probability > 0.8:
            HIGH_RISK_COUNT.inc()

        prob_sum += probability
        pred_count += 1
        AVG_FRAUD_PROB.set(round(prob_sum / pred_count, 4))

        REQUEST_LATENCY.observe(time.time() - start)
        logger.info(f"Prediction: {result}")

    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error(f"Prediction error: {e}")
        result = {"error": str(e)}

    return render_template("index.html", result=result)

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    app.run(debug=True, port=5000)