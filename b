response = sm_client.list_inference_components(EndpointName=endpoint_name)
for comp in response['InferenceComponents']:
    print(comp['InferenceComponentName'])

---
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

endpoint_name = "ARM-1744363227697-Endpoint-20250411-092031"
inference_component_name = "YourInferenceComponentName"  # Replace with actual name

predictor = Predictor(
    endpoint_name=endpoint_name,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

# Provide InferenceComponentName in the request
response = predictor.predict(
    {"instances": [[0.5, 1.2, 3.3, 4.0, 2.1, 0.7]]},
    inference_component_name=inference_component_name
)

print("Prediction:", response)
