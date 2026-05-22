"""
Churn Prediction API

Run with:
    litestar --app main:app run --reload
Then open:
    http://localhost:8000/schema/swagger
"""

from typing import List, Literal
from litestar import Litestar, post, get
from pydantic import BaseModel, Field
from app.model_utils import predict_churn
from app.logger_setup import setup_logging

logger = setup_logging()


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
# Age > 18 
class ChurnRequest(BaseModel):
    credit_score: int = Field(ge=0)
    age: int = Field(gt=18)
    tenure: int = Field(ge=0)
    balance: float = Field(ge=0.0)
    num_of_products: int = Field(ge=1)
    has_cr_card: Literal[0, 1] 
    is_active_member: Literal[0, 1]
    estimated_salary: float = Field(ge=0.0)
    geography: Literal["Germany", "Spain", "France"]
    gender: Literal["Male", "Female"]
    


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@get("/")
async def index() -> str:
    return "Welcome There!"

# TODO 3: Create a GET endpoint at "/health" that returns {"status": "healthy"}
@get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@post("/predict")
async def predict(data: ChurnRequest) -> dict[str, int]:
    features = [
        data.credit_score,
        data.age,
        data.tenure,
        data.balance,
        data.num_of_products,
        data.has_cr_card,
        data.is_active_member,
        data.estimated_salary,
        1 if data.geography == "Germany" else 0,
        1 if data.geography == "Spain" else 0,
        1 if data.gender == "Male" else 0
    ]
    prediction = predict_churn(features)
    logger.info(f"Input: {features}, Prediction: {prediction}")
    return {"prediction": prediction}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
# TODO 5: Register your endpoint functions in the list below
app = Litestar(
    route_handlers=[index, health, predict],
)