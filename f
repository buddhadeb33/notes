import boto3

client = boto3.client("sagemaker")

client.create_model_package_group(
    ModelPackageGroupName="MyModelPackageGroup",
    ModelPackageGroupDescription="Group for my ML models"
)
---
from sagemaker.model import Model
from sagemaker import Session
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.execution_variables import ExecutionVariables

# Assuming you're using a pipeline session
sagemaker_session = PipelineSession()
role = "arn:aws:iam::<your-account-id>:role/service-role/AmazonSageMaker-ExecutionRole"

# Define the model artifact and container
model = Model(
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
    model_data="s3://my-bucket/path/to/model.tar.gz",
    role=role,
    sagemaker_session=sagemaker_session
)

# Create the RegisterModel step
register_model_step = RegisterModel(
    name="MyModelRegistrationStep",
    model=model,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large", "ml.m5.xlarge"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="MyModelPackageGroup",
    approval_status="PendingManualApproval",  # or "Approved"
    description="My model registration",
)

# If inside a pipeline, you’d include this in your steps list
