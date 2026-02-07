"""
This module contains functions to preprocess and train multiple models
for bank consumer churn prediction with MLflow tracking.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import mlflow

from model_config import get_model, get_model_params, list_available_models

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())
    import joblib

    # mlflow.sklearn.log_model(col_transf, "preprocessor")

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())
    joblib.dump(col_transf, "preprocessor.pkl")  # Save the preprocessor to a file
    mlflow.log_artifact("preprocessor.pkl")  # Log the preprocessor as an artifact
    # Log the transformer as an artifact
    # mlflow.sklearn.log_model(col_transf, "preprocessor")

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train, model_name: str):
    """
    Train a model specified by name.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target
        model_name (str): Name of the model to train

    Returns:
        trained model instance
    """
    model = get_model(model_name)
    model_params = get_model_params(model_name)
    
    model.fit(X_train, y_train)

    # Log the model with the input and output schema
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)

    # Log model parameters
    for param_name, param_value in model_params.items():
        mlflow.log_param(param_name, param_value)

    # Log data info
    mlflow.log_param("num_rows", X_train.shape[0])
    mlflow.log_param("num_features", X_train.shape[1])

    return model


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    ### Set the experiment name
    mlflow.set_experiment("churn_prediction")

    # Load data once
    df = pd.read_csv("/media/ahmed-fayad/data/MLops/MLOps-Course-Labs/src/Churn_Modelling.xls")

    # Get all available models
    models_to_train = list_available_models()
    print(f"Training {len(models_to_train)} models: {models_to_train}")

    results = []

    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print('='*50)

        ### Start a new run for each model
        with mlflow.start_run(run_name=model_name):
            col_transf, X_train, X_test, y_train, y_test = preprocess(df)

            model = train(X_train, y_train, model_name)

            y_pred = model.predict(X_test)

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            ### Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            ### Log tag
            mlflow.set_tag("model_type", model_name)

            # Log confusion matrix
            conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
            conf_mat_disp = ConfusionMatrixDisplay(
                confusion_matrix=conf_mat, display_labels=model.classes_
            )
            conf_mat_disp.plot()
            
            plt.title(f"Confusion Matrix - {model_name}")
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
            plt.close()

            # Store results
            results.append({
                "model": model_name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
            })

            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1 Score:  {f1:.4f}")

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print('='*60)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1_score", ascending=False)
    print(results_df.to_string(index=False))
    
    best_model = results_df.iloc[0]
    print(f"\nBest Model: {best_model['model']} (F1: {best_model['f1_score']:.4f})")



if __name__ == "__main__":
    
    main()
