import pytest
import joblib
import numpy as np

model = joblib.load("models/model.pkl")

def test_model_loads():
    assert model is not None

def test_model_predicts():
    data = np.array([[1, 4, 500000, 500000, 0]])
    prediction = model.predict(data)
    assert prediction[0] in [0, 1]

def test_model_probability():
    data = np.array([[1, 4, 500000, 500000, 0]])
    prob = model.predict_proba(data)[0][1]
    assert 0 <= prob <= 1

def test_legitimate_transaction():
    data = np.array([[1, 3, 100, 100, 100]])
    prediction = model.predict(data)
    assert prediction[0] == 0