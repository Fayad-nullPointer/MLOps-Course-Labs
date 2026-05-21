"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""
import joblib
# TODO 1: Load your serialized churn model from data/model.joblib
model = joblib.load("/media/ahmed-fayad/3b40def2-87b7-41ce-8913-2981f887941c/home/ITI Cont.../MLOPs/MLOps-Course-Labs/data/model.pkl")
# preprocessor = joblib.load("data/preprocessor.pkl")

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
