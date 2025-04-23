import boto3
import pickle
import os
import json
import tempfile
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# S3 details
s3 = boto3.client("s3")
MODEL_BUCKET = "sagemaker-us-east-1-436606633363"
MODEL_PREFIX = "take/models/"

# List of models in S3
model_files = [
    "trained_model_ca_ach.pkl",
    "trained_model_ca_corporate_card.pkl",
    "trained_model_ca_corporate_card_auth.pk1",
    "trained_model_ca_ebp.pkl",
    "trained_model_ca_fraud_protection.pkl",
    "trained_model_ca_image_cash_letter.pk1"
]

# Global cache
loaded_models = {}

# Function to load model from S3
def load_model_from_s3(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name]

    key = MODEL_PREFIX + model_name
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        s3.download_fileobj(MODEL_BUCKET, key, tmp_file)
        tmp_path = tmp_file.name

    with open(tmp_path, "rb") as f:
        model = pickle.load(f)

    loaded_models[model_name] = model
    return model

# Generate synthetic data
def create_synthetic_data(n_samples=100, n_features=20):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    return X, y

# Lambda entry point
def lambda_handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))
        n_samples = body.get('n_samples', 100)

        predictions = {}

        for model_name in model_files:
            try:
                model = load_model_from_s3(model_name)

                # Get feature count dynamically
                if isinstance(model, LogisticRegression):
                    n_features = model.coef_.shape[1]
                elif isinstance(model, lgb.LGBMClassifier):
                    n_features = model.booster_.num_feature()
                else:
                    predictions[model_name] = "Unsupported model type"
                    continue

                X, _ = create_synthetic_data(n_samples=n_samples, n_features=n_features)
                preds = model.predict(X).tolist()
                predictions[model_name] = preds[:20]

            except Exception as model_error:
                predictions[model_name] = f"Error: {str(model_error)}"

        return {
            'statusCode': 200,
            'body': json.dumps(predictions)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
