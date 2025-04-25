# SageMaker Inference Pipeline with Multiple Logistic Regression Models

This documentation outlines the process of registering multiple pre-trained logistic regression models in the Amazon SageMaker Model Registry and deploying them as a pipeline endpoint for batch or real-time inference. It is designed for scenarios where the models already exist as pickle files (`model1.pkl`, `model2.pkl`).

---

## **Overview**

**Steps Covered:**
1. Writing the inference entry script (`inference.py`)
2. Initializing SageMaker session and creating a Model Package Group
3. Packaging and uploading the models
4. Registering models in the SageMaker Model Registry
5. Generating synthetic test data
6. Creating a pipeline model and deploying as an endpoint
7. Invoking the endpoint to get predictions

---

## **1. Entry Point Script (`inference.py`)**
This script defines how SageMaker loads the model, handles input/output, and performs predictions.

```python
import os
import pickle
import json
import pandas as pd
import io

def model_fn(model_dir):
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        return pickle.load(f)

def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        return pd.read_csv(io.StringIO(request_body), header=None)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    return model.predict_proba(input_data)

def output_fn(predictions, content_type):
    if content_type == 'application/json':
        return json.dumps(predictions.tolist()), content_type
    raise ValueError(f"Unsupported content type: {content_type}")
```

---

## **2. Initialize SageMaker and Create Model Package Group**

```python
import boto3
import sagemaker

session = sagemaker.Session()
role = sagemaker.get_execution_role()
sm = boto3.client('sagemaker')

model_package_group_name = 'logistic-reg-model-group'
try:
    sm.create_model_package_group(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageGroupDescription='Logistic regression models'
    )
except sm.exceptions.ResourceInUse:
    pass
```

---

## **3. Package and Register Models**
Each pickle file is archived, uploaded to S3, and registered as a model package.

```python
import tarfile

bucket = session.default_bucket()
prefix = 'sagemaker/inference-pipeline'
model_files = ['model1.pkl', 'model2.pkl']
model_package_arns = []

for idx, mf in enumerate(model_files, start=1):
    artifact = f'model{idx}.tar.gz'
    with tarfile.open(artifact, 'w:gz') as tar:
        tar.add(mf, arcname='model.pkl')

    s3_uri = session.upload_data(path=artifact, bucket=bucket, key_prefix=f'{prefix}/model{idx}')

    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription=f'Logistic regression #{idx}',
        InferenceSpecification={
            'Containers': [{
                'Image': sagemaker.image_uris.retrieve('sklearn', session.boto_region_name, version='0.23-1'),
                'ModelDataUrl': s3_uri
            }],
            'SupportedContentTypes': ['text/csv'],
            'SupportedResponseMIMETypes': ['application/json']
        },
        ModelApprovalStatus='Approved'
    )
    model_package_arns.append(response['ModelPackageArn'])
```

---

## **4. Generate Synthetic Input Data**

```python
from sklearn.datasets import make_classification
import pandas as pd

X_test, _ = make_classification(n_samples=5, n_features=10, random_state=123)
pd.DataFrame(X_test).to_csv('synthetic_data.csv', index=False, header=False)
```

---

## **5. Deploy as a Pipeline Model Endpoint**

```python
from sagemaker.model import ModelPackage
from sagemaker.pipeline import PipelineModel
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

model_pkgs = [ModelPackage(role, arn) for arn in model_package_arns]

pipeline = PipelineModel(
    name='logreg-inference-pipeline',
    role=role,
    models=model_pkgs,
    sagemaker_session=session
)

endpoint_name = 'logreg-registry-pipeline-endpoint'
predictor = pipeline.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=endpoint_name
)
predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()
```

---

## **6. Invoke Endpoint**

```python
with open('synthetic_data.csv', 'r') as f:
    payload = f.read()

predictions = predictor.predict(payload)
columns = [f'model{m}_class{i}' for m in range(1, 3) for i in range(2)]

import pandas as pd
pd.DataFrame(predictions, columns=columns)
```

---

## **Optional: Clean Up**

```python
# predictor.delete_endpoint()
# sm.delete_model_package_group(ModelPackageGroupName=model_package_group_name)
```

---

## **Conclusion**
This approach allows you to version, track, and invoke multiple pre-registered models through a single SageMaker pipeline endpoint. Ideal for ensembles, comparative evaluation, or routing predictions through a chain of models.

