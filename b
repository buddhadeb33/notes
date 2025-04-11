# inference.py
import joblib
import os
import numpy as np

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    return model

def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        return np.array(request_body['instances'])
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data).tolist()

def output_fn(prediction, content_type='application/json'):
    return {"predictions": prediction}
----

tar -czf model.tar.gz model.pkl inference.py

----
aws s3 cp model.tar.gz s3://your-bucket-name/model/model.tar.gz
--
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import Session

role = "arn:aws:iam::<your-account>:role/service-role/SageMakerExecutionRole"

model = SKLearnModel(
    model_data="s3://your-bucket-name/model/model.tar.gz",
    role=role,
    entry_point="inference.py",
    framework_version="0.23-1",  # use version matching your sklearn
    py_version='py3',
)

predictor = model.deploy(instance_type="ml.m5.large", initial_instance_count=1)

--
response = predictor.predict({
    "instances": [[0.1, 0.0, 1.0, 0.3, 0.2, 0.0, 1.0, 0.4, 0.2, 0.3, 0.0, 1.0, 0.5, 0.1, 0.3, 0.0]]
})

{
  "instances": [
    [0.23, 0.11, 0.45, 0.67, 0.39, 0.78, 0.54, 0.31, 0.25, 0.68, 0.90, 0.12, 0.34, 0.88, 0.76],
    [0.45, 0.22, 0.66, 0.19, 0.58, 0.74, 0.93, 0.12, 0.43, 0.57, 0.61, 0.38, 0.49, 0.27, 0.80]
  ]
}

print("Prediction:", response)
--
import json

# Send a single row of input (16 features if including intercept)
input_data = {
    "instances": [buddha_df.iloc[0].tolist()]
}

response = predictor.predict(input_data)
print(response)
