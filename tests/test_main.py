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
    sample_zeros = [0, 0, 0, 0.0, 0, 0, 0, 0.0, 0, 0, 0]
    result = predict_churn(sample_zeros)
    assert result in (0, 1)
from pydantic import ValidationError
from main import ChurnRequest

def test_logical_constraints_edge_cases():
    """Test that invalid logical inputs are caught by Pydantic validation."""
    
    # We expect this code block to raise a ValidationError
    with pytest.raises(ValidationError) as exc_info:
        # Create a request with logically impossible or constrained values
        ChurnRequest(
            credit_score=0,
            age=0,                # Fails because age must be > 18
            tenure=-1,            # Logically impossible
            balance=-500.0,       # Bank balance shouldn't be negative in this context
            num_of_products=0, 
            has_cr_card=5,        # Should be strictly 0 or 1
            is_active_member=3,   # Should be strictly 0 or 1
            estimated_salary=-10, # Logically impossible
            geography="Italy",    # Fails because only France, Spain, Germany are allowed
            gender="Alien"        # Fails because only Male, Female are allowed
        )
    
    # Assert that the error specifically caught the age constraint
    assert "age" in str(exc_info.value)


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
