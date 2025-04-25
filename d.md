Absolutely! Below is the complete **SageMaker Inference Pipeline Notebook** code, including:

- Creating and packaging two `LogisticRegression` models
- Writing `inference.py` inline
- Registering both models in a single **Model Package Group**
- Deploying them in a `PipelineModel`
- Performing inference on synthetic data

---

## ✅ Full Notebook Code: SageMaker Inference Pipeline

```python
# %% [markdown]
# # SageMaker Inference Pipeline Notebook

# %% [markdown]
## Step 1: Create and Save Two Logistic Regression Models

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pickle

# Generate data
X, y = make_classification(n_samples=200, n_features=10, random_state=42)

# Train two different LogisticRegression models
model1 = LogisticRegression(max_iter=1000).fit(X, y)
model2 = LogisticRegression(C=0.5, max_iter=1000).fit(X, y)

# Save them as pickle files
with open('model1.pkl', 'wb') as f:
    pickle.dump(model1, f)

with open('model2.pkl', 'wb') as f:
    pickle.dump(model2, f)
```

---

## Step 2: Write `inference.py` inside the notebook

```python
%%writefile inference.py
import os
import pickle
import json
import pandas as pd
import io

def model_fn(model_dir):
    """Load model from local directory"""
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        return pickle.load(f)

def input_fn(request_body, request_content_type):
    """Parse CSV payload"""
    if request_content_type == 'text/csv':
        return pd.read_csv(io.StringIO(request_body), header=None)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Return class probabilities"""
    return model.predict_proba(input_data)

def output_fn(predictions, content_type):
    """Convert to JSON"""
    if content_type == 'application/json':
        return json.dumps(predictions.tolist()), content_type
    raise ValueError(f"Unsupported content type: {content_type}")
```

---

## Step 3: Setup SageMaker Clients and Model Package Group

```python
import boto3
import sagemaker

session = sagemaker.Session()
role = sagemaker.get_execution_role()
sm = boto3.client('sagemaker')

model_package_group_name = 'logistic-reg-model-group'

# Create the model group if it doesn't exist
try:
    sm.create_model_package_group(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageGroupDescription='Logistic regression models'
    )
except sm.exceptions.ResourceInUse:
    pass

print(f"✅ Using Model Package Group: {model_package_group_name}")
```

---

## Step 4: Package and Register Each Model in Registry

```python
import os, tarfile

bucket = session.default_bucket()
prefix = 'sagemaker/inference-pipeline'
model_files = ['model1.pkl', 'model2.pkl']
model_package_arns = []

for idx, mf in enumerate(model_files, start=1):
    # Rename to model.pkl in tar
    artifact = f'model{idx}.tar.gz'
    with tarfile.open(artifact, 'w:gz') as tar:
        tar.add(mf, arcname='model.pkl')

    # Upload to S3
    s3_uri = session.upload_data(
        path=artifact,
        bucket=bucket,
        key_prefix=f'{prefix}/model{idx}'
    )

    # Register to Model Package Group
    resp = sm.create_model_package(
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

    model_package_arns.append(resp['ModelPackageArn'])
    print(f"✅ Registered model {idx}: {resp['ModelPackageArn']}")
```

---

## Step 5: Inspect Registered Models

```python
import pandas as pd

tracking = []
for arn in model_package_arns:
    desc = sm.describe_model_package(ModelPackageName=arn)
    tracking.append({
        'Model ARN': arn,
        'Created': desc['CreationTime'],
        'Status': desc['ModelPackageStatus']
    })

pd.DataFrame(tracking)
```

---

## Step 6: Generate Test Data

```python
from sklearn.datasets import make_classification
import pandas as pd

X_test, _ = make_classification(n_samples=5, n_features=10, random_state=123)
input_csv = 'synthetic_data.csv'
pd.DataFrame(X_test).to_csv(input_csv, index=False, header=False)
```

---

## Step 7: Deploy Inference Pipeline Endpoint

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

print(f"✅ Deployed pipeline endpoint: {endpoint_name}")
```

---

## Step 8: Invoke Endpoint

```python
with open(input_csv, 'r') as f:
    payload = f.read()

predictions = predictor.predict(payload)

# Each model gives [p0, p1] for each sample → total columns = 2 models × 2 classes
columns = [f'model{m}_class{i}' for m in range(1, 3) for i in range(2)]
pd.DataFrame(predictions, columns=columns)
```

---

## (Optional) Step 9: Cleanup

```python
# predictor.delete_endpoint()
# sm.delete_model_package_group(ModelPackageGroupName=model_package_group_name)
```

---

Let me know if you'd like this as a downloadable `.ipynb` or want to add automated model version tagging!
