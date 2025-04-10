# Create model registry group if not exists
model_package_group_name = "my-pickle-model-group"

client = boto3.client("sagemaker")

try:
    client.create_model_package_group(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageGroupDescription="Model group for pickle-based models"
    )
except client.exceptions.ResourceInUse:
    print("Model group already exists.")


model = SKLearnModel(
    model_data=model_data,
    role=role,
    framework_version="1.0-1",  # Match your scikit-learn version
    sagemaker_session=sagemaker_session
)

model_package = model.register(
    content_types=["application/json"],
    response_types=["application/json"],
    model_package_group_name=model_package_group_name,
    approval_status="Approved",  # You can also set "PendingManualApproval"
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"]
)

print("Model registered in registry!")
