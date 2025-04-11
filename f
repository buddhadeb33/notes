import os
import pickle
import pandas as pd
import io

# Load the model from model_dir/model.pkl
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Convert incoming request to a DataFrame
def input_fn(request_body, request_content_type):
    if request_content_type == "application/x-parquet":
        return pd.read_parquet(io.BytesIO(request_body))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# Make prediction
def predict_fn(input_data, model):
    return model.predict(input_data)

# Format output
def output_fn(prediction, content_type):
    return str(prediction.tolist())
