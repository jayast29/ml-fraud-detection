import pytest
import sys
sys.path.insert(0, 'src')
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200

def test_prediction_post(client):
    response = client.post("/predict", data={
        "step": 1,
        "type": 4,
        "amount": 500000,
        "balance_diff_orig": 500000,
        "balance_diff_dest": 0
    })
    assert response.status_code == 200