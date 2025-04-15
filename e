Here is the full list of all 39 models in a dataframe format:

Let me know if you'd like this in CSV, Excel, or any other format.
1.  ca_ach  
2.  ca_corporate_card  
3.  ca_corporate_card_auth  
4.  ca_ebp  
5.  ca_fraud_protection  
6.  ca_image_cash_letter  
7.  ca_int_high_val_wire  
8.  ca_letters_of_credit  
9.  ca_moneris  
10. ca_online_banking  
11. ca_op_check_acct  
12. ca_real_time_payments  
13. ca_remote_deposit_capture  
14. ca_saving_interest_acct  
15. ca_treasury_deposits  
16. ca_virtual_card  
17. ca_zba  
18. us_ach  
19. us_automated_credit_sweep  
20. us_corporate_card  
21. us_corporate_card_auth  
22. us_edi  
23. us_elavon  
24. us_fraud_protection  
25. us_image_cash_letter  
26. us_int_high_val_wire  
27. us_letters_of_credit  
28. us_online_banking  
29. us_op_check_acct  
30. us_real_time_payments  
31. us_remote_deposit_capture  
32. us_saving_interest_acct  
33. us_term_rev_loan  
34. us_tf_equipment_finance  
35. us_treasury_deposits  
36. us_virtual_card  
37. us_zba  
38. us_rev_loan  
39. us_data_sweep
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
