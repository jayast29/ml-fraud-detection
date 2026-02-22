from flask import Flask, render_template, request
import joblib
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
model = joblib.load(os.path.join(os.path.dirname(__file__), '../models/model.pkl'))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
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
        logger.info(f"Prediction: {result}")
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)