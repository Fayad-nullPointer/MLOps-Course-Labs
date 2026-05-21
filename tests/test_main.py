"""
Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""


import pytest
from litestar.testing import TestClient
from app.model_utils import predict_churn
from main import app

# ---------------------------------------------------------------------------
# Function Tests
# ---------------------------------------------------------------------------

def test_predict_churn_direct():
    sample = [22, 1000, 3, 50000, 2, 1, 1, 50000, 0, 0, 1]
    result = predict_churn(sample)
    assert result in (0, 1)

def test_predict_churn_edge_cases():
    # TODO 2 (bonus): Write another function test with edge-case inputs
    sample_zeros = [0, 0, 0, 0.0, 0, 0, 0, 0.0, 0, 0, 0]
    result = predict_churn(sample_zeros)
    assert result in (0, 1)


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------

def test_predict_endpoint_valid():
    valid_payload = {
        "credit_score": 619,
        "age": 42,
        "tenure": 2,
        "balance": 0.0,
        "num_of_products": 1,
        "has_cr_card": 1,
        "is_active_member": 1,
        "estimated_salary": 101348.88,
        "geography": "France",
        "gender": "Female"
    }
    with TestClient(app=app) as client:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 201
        assert "prediction" in response.json()
        assert response.json()["prediction"] in (0, 1)

def test_health_endpoint():
    with TestClient(app=app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_index():
    with TestClient(app=app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == "Welcome There!"

def test_predict_endpoint_invalid_input():
    invalid_payload = {"age": "not an int"}
    with TestClient(app=app) as client:
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 400
