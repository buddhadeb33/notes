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
