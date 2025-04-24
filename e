import boto3

client = boto3.client('sagemaker')

client.create_model_package_group(
    ModelPackageGroupName='TAKE',
    ModelPackageGroupDescription='Models under TAKE group'
)

client.create_model_package_group(
    ModelPackageGroupName='HAS',
    ModelPackageGroupDescription='Models under HAS group'
)

client.create_model_package(
    ModelPackageGroupName='TAKE',
    ModelPackageDescription='TAKE group - US model',
    InferenceSpecification={...},  # your inference containers
    ModelApprovalStatus='Approved',
    Tags=[
        {'Key': 'region', 'Value': 'US'},
        {'Key': 'model_type', 'Value': 'HAS'}
    ]
)

client.create_model_package(
    ModelPackageGroupName='HAS',
    ModelPackageDescription='HAS group - CA model',
    InferenceSpecification={...},
    ModelApprovalStatus='Approved',
    Tags=[
        {'Key': 'region', 'Value': 'CA'},
        {'Key': 'model_type', 'Value': 'TAKE'}
    ]
)
---
import boto3

sagemaker = boto3.client("sagemaker")

response = sagemaker.create_model_package(
    ModelPackageGroupName="TAKE",  # Model Group Name
    ModelPackageDescription="TAKE group - US model v1",
    InferenceSpecification={
        "Containers": [
            {
                "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.1-cpu-py38-ubuntu20.04",
                "ModelDataUrl": "s3://your-bucket/path/to/model.tar.gz",
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "s3://your-bucket/path/to/code.tar.gz"
                }
            }
        ],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
    ModelApprovalStatus="PendingManualApproval",  # or 'Approved'
    Tags=[
        {"Key": "region", "Value": "US"},
        {"Key": "model_type", "Value": "HAS"},
    ]
)

print("Model Package ARN:", response["ModelPackageArn"])


--


import boto3

sagemaker = boto3.client("sagemaker")

response = sagemaker.create_model_package(
    ModelPackageGroupName='TAKE',
    ModelPackageDescription='TAKE group - US model',
    InferenceSpecification={
        "Containers": [
            {
                "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.1-cpu-py38-ubuntu20.04",
                "ModelDataUrl": "s3://your-bucket/take/us/model-v1/model.tar.gz",
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "s3://your-bucket/take/us/code.tar.gz"
                }
            }
        ],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"]
    },
    ModelApprovalStatus='Approved',
    Tags=[
        {'Key': 'region', 'Value': 'US'},
        {'Key': 'model_type', 'Value': 'HAS'}
    ]
)
print("Model registered. ARN:", response["ModelPackageArn"])
--
from sagemaker import image_uris

# Example for PyTorch
image_uri = image_uris.retrieve(
    framework='pytorch',
    region='us-west-2',
    version='1.12.1',
    py_version='py38',
    instance_type='ml.m5.large',  # affects CPU vs GPU
    image_scope='inference'       # for deployment, use 'training' for training jobs
)

print("Container Image URI:", image_uri)

--
import boto3

# Initialize SageMaker client for us-east-1
sagemaker = boto3.client("sagemaker", region_name="us-east-1")

# Define model registration parameters
model_group = "TAKE"
model_data_url = "s3://your-bucket/path/to/your-model.tar.gz"  # <-- update this
container_image = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3"

# First, create the model package group (if not already created)
response_group = sagemaker.create_model_package_group(
    ModelPackageGroupName=model_group,
    ModelPackageGroupDescription="Model group for TAKE models",
    Tags=[
        {"Key": "region", "Value": "US"},
        {"Key": "model_type", "Value": "HAS"}
    ]
)
print("Model Package Group created. ARN:", response_group["ModelPackageGroupArn"])

# Now, register a model under this group
response = sagemaker.create_model_package(
    ModelPackageGroupName=model_group,
    ModelPackageDescription="TAKE group - US sklearn model",
    InferenceSpecification={
        "Containers": [
            {
                "Image": container_image,
                "ModelDataUrl": model_data_url
            }
        ],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"]
    },
    ModelApprovalStatus="Approved",  # or 'PendingManualApproval'
)

print("Model registered successfully.")
print("Model Package ARN:", response["ModelPackageArn"])

