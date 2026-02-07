# Bank Customer Churn Prediction

A modular MLOps project for predicting bank customer churn using multiple machine learning models with MLflow tracking.

## Project Structure

```
MLOps-Course-Labs/
├── README.md
├── requirements.txt
├── dataset/
│   └── note.txt
└── src/
    ├── train.py           # Main training script
    ├── model_config.py    # Model configurations
    └── Churn_Modelling.xls # Dataset
```

## Features

- **Multi-Model Training**: Train and compare 8 different ML models automatically
- **MLflow Integration**: Full experiment tracking with metrics, parameters, and artifacts
- **Modular Design**: Easy to add new models via configuration
- **Automated Comparison**: Generates summary with best model recommendation

## Available Models

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear classifier with regularization |
| Decision Tree | Tree-based classifier |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential ensemble method |
| AdaBoost | Adaptive boosting classifier |
| SVM | Support Vector Machine with RBF kernel |
| KNN | K-Nearest Neighbors |
| Naive Bayes | Gaussian Naive Bayes |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Start MLflow Server

```bash
mlflow ui --port 5000
```

### 2. Run Training

```bash
cd src
python train.py
```

### 3. View Results

Open http://localhost:5000 to view experiment results in MLflow UI.

## Adding New Models

Edit `src/model_config.py` to add new models:

```python
MODELS_CONFIG = {
    "your_new_model": {
        "model": YourModelClass,
        "params": {
            "param1": value1,
            "param2": value2,
        },
    },
    # ... existing models
}
```

## Metrics Tracked

- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall

## Artifacts

Each run logs:
- Trained model (MLflow sklearn format)
- Preprocessor (`preprocessor.pkl`)
- Confusion matrix plot

## Configuration

Models and hyperparameters are configured in `src/model_config.py`. Modify the `MODELS_CONFIG` dictionary to:

- Change hyperparameters
- Enable/disable specific models
- Add new model types

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- matplotlib
- mlflow
- joblib

## License

MIT