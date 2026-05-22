"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

import joblib

# TODO 1: Load your serialized churn model from data/model.joblib
from pathlib import Path

# Build the correct path dynamically based on where this file is located
# __file__ is app/model_utils.py
# .parent is app/
# .parent.parent is MLOps-Course-Labs/
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "model.pkl"

model = joblib.load(MODEL_PATH)  # preprocessor = joblib.load("data/preprocessor.pkl")


def predict_churn(features: list[float]) -> int:
    """
    Takes a list of feature values and returns a churn prediction (0 or 1).
    """
    prediction = model.predict([features])
    return int(prediction[0])


if __name__ == "__main__":
    # TODO 3: Replace with sample features that match your model
    #     Feature Names Expected by the Model:
    # ['standardscaler__CreditScore' 'standardscaler__Age'
    #  'standardscaler__Tenure' 'standardscaler__Balance'
    #  'standardscaler__NumOfProducts' 'standardscaler__HasCrCard'
    #  'standardscaler__IsActiveMember' 'standardscaler__EstimatedSalary'
    #  'onehotencoder__Geography_Germany' 'onehotencoder__Geography_Spain'
    #  'onehotencoder__Gender_Male']

    sample = [22, 1000, 3, 50000, 2, 1, 1, 50000, 0, 0, 1]
    print(f"Input:      {sample}")
    print(f"Prediction: {predict_churn(sample)}")
