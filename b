import boto3
import json

# Initialize the SageMaker runtime client
runtime = boto3.client('sagemaker-runtime')

# Define your endpoint name (replace with your actual one)
endpoint_name = 'sagemaker-scikit-learn-2025-04-11-12-26-51-318'

# Input: A single sample (15 features)
input_data = {
    "instances": [
        [0.23, 0.11, 0.45, 0.67, 0.39, 0.78, 0.54, 0.31, 0.23, 0.15, 0.68, 0.12, 0.34, 0.88, 0.76]
    ]
}

# Convert to JSON
payload = json.dumps(input_data)

# Invoke the endpoint
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=payload
)

# Decode the response
result = json.loads(response['Body'].read())
print("Prediction:", result)
