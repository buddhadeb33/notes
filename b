import boto3
import json
import numpy as np
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Your SageMaker endpoint name (replace this with your actual name)
endpoint_name = "your-endpoint-name"

# Create a predictor object
predictor = Predictor(
    endpoint_name=endpoint_name,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

# Example input (match your model's expected feature size!)
input_data = [[0.5, 1.2, 3.3, 4.0, 2.1, 0.7]]  # One row, 6 features

# Make prediction
response = predictor.predict({"instances": input_data})

# Print the result
print("Prediction:", response)
