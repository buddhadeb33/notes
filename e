# inference.py

import os
import pickle
import pandas as pd
import numpy as np

def model_fn(model_dir):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/x-parquet":
        import io
        df = pd.read_parquet(io.BytesIO(request_body))
        return df
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return str(prediction.tolist())


#####


import boto3
import sagemaker
from sagemaker import Model
from sagemaker.model_package import ModelPackage
from sagemaker.session import ModelPackageGroup
from sagemaker.model_metrics import MetricsSource, ModelMetrics

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

# Define model registry group name
model_package_group_name = "ca-model-registry"

# Create a model package group if it doesn't exist
client = boto3.client("sagemaker")
try:
    client.create_model_package_group(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageGroupDescription="Model registry for CA model"
    )
except client.exceptions.ResourceInUse:
    print("ModelPackageGroup already exists")

# Define model
model = Model(
    model_data="s3://bucket/model/model_ca.tar.gz",
    image_uri=sagemaker.image_uris.retrieve("sklearn", region=region, version="1.2-1"),
    role=role,
    entry_point="inference.py",
    source_dir="code"  # only used if model is created in code; not used here since already tarred
)

# Create ModelPackage
model_package = model.register(
    content_types=["application/x-parquet"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=model_package_group_name,
    approval_status="PendingManualApproval",
    description="Scikit-learn CA model using custom inference script"
)

print(f"Model Package ARN: {model_package.model_package_arn}")
#############

# test_inference_local.py

import pandas as pd
import inference
import os

# Simulate SageMaker model directory
model_dir = os.getcwd()

# Load model using model_fn
model = inference.model_fn(model_dir)

# Load input parquet
with open("input.parquet", "rb") as f:
    raw_data = f.read()

# Simulate input_fn
data = inference.input_fn(raw_data, "application/x-parquet")

# Predict using predict_fn
prediction = inference.predict_fn(data, model)

# Format response using output_fn
response = inference.output_fn(prediction, "application/json")

print("Prediction Output:")
print(response)
