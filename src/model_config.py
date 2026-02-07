"""
Model configurations for churn prediction experiments.
Each model has a name, class, and hyperparameters.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Model configurations dictionary
# Each entry: "model_name": {"model": ModelClass, "params": {hyperparameters}}
MODELS_CONFIG = {
    "logistic_regression": {
        "model": LogisticRegression,
        "params": {
            "max_iter": 1000,
            "random_state": 1912,
        },
    },
    "decision_tree": {
        "model": DecisionTreeClassifier,
        "params": {
            "max_depth": 5,
            "random_state": 1912,
        },
    },
    "random_forest": {
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 1912,
        },
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier,
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 1912,
        },
    },
    "adaboost": {
        "model": AdaBoostClassifier,
        "params": {
            "n_estimators": 50,
            "learning_rate": 1.0,
            "random_state": 1912,
        },
    },
    "svm": {
        "model": SVC,
        "params": {
            "kernel": "rbf",
            "C": 1.0,
            "random_state": 1912,
        },
    },
    "knn": {
        "model": KNeighborsClassifier,
        "params": {
            "n_neighbors": 5,
        },
    },
    "naive_bayes": {
        "model": GaussianNB,
        "params": {},
    },
}


def get_model(model_name: str):
    """
    Get a model instance by name.
    
    Args:
        model_name: Name of the model from MODELS_CONFIG
        
    Returns:
        Instantiated model with configured parameters
    """
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODELS_CONFIG.keys())}")
    
    config = MODELS_CONFIG[model_name]
    return config["model"](**config["params"])


def get_model_params(model_name: str) -> dict:
    """
    Get the parameters for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of model parameters
    """
    return MODELS_CONFIG[model_name]["params"]


def list_available_models() -> list:
    """Return list of available model names."""
    return list(MODELS_CONFIG.keys())
