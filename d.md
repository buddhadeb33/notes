# %% [markdown]
# SageMaker Inference Pipeline Notebook

This notebook demonstrates how to:

1. Register two logistic regression models in the SageMaker Model Registry
2. Build an inference pipeline from registry model packages
3. Generate synthetic test data
4. Deploy and invoke the pipeline endpoint
5. Track and inspect SageMaker models and packages

---

# %% [markdown]
## Step 1: Create the inference script (`inference.py`)

We implement four required functions:

- `model_fn`: load the pickle
- `input_fn`: parse CSV input
- `predict_fn`: run `predict_proba`
- `output_fn`: serialize to JSON

# %%
%%bash
cat << 'EOF' > inference.py
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
EOF

---

# %% [markdown]
## Step 2: Setup AWS Clients and Registry Group

Initialize Boto3 & SageMaker sessions, and create (or reuse) a Model Package Group.

# %%
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
print(f"Using Model Package Group: {model_package_group_name}")

---

# %% [markdown]
## Step 3: Generate Synthetic Data

Use `sklearn.datasets.make_classification` to create features + labels, then write CSV without headers.

# %%
from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(
    n_samples=200, n_features=10,
    n_informative=5, n_redundant=2,
    random_state=42
)
df = pd.DataFrame(X)
df['target'] = y
input_csv = 'synthetic_data.csv'
df.drop('target', axis=1).to_csv(input_csv, index=False, header=False)
print(f"Saved synthetic data to: {input_csv}")

---

# %% [markdown]
## Step 4: Package, Upload & Register Models

- Tar each `model{i}.pkl`
- Upload to S3
- Call `create_model_package` in the registry

# %%
import os, tarfile

bucket = session.default_bucket()
prefix = 'sagemaker/inference-pipeline'
model_files = ['model1.pkl', 'model2.pkl']
model_package_arns = []

for idx, mf in enumerate(model_files, start=1):
    artifact = f'model{idx}.tar.gz'
    with tarfile.open(artifact, 'w:gz') as tar:
        tar.add(mf, arcname='model.pkl')

    s3_uri = session.upload_data(
        path=artifact,
        bucket=bucket,
        key_prefix=f'{prefix}/model{idx}'
    )

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
    arn = resp['ModelPackageArn']
    model_package_arns.append(arn)
    print(f"Registered in registry: {arn}")

---

# %% [markdown]
## Step 5: Inspect Registered Model Packages

List ARNs, creation times, and statuses for tracking.

# %%
import pandas as pd
packages = [sm.describe_model_package(ModelPackageName=arn) for arn in model_package_arns]
tracking = [{
    'Arn': pkg['ModelPackageArn'],
    'Created': pkg['CreationTime'],
    'Status': pkg['ModelPackageStatus']
} for pkg in packages]
print(pd.DataFrame(tracking))

---

# %% [markdown]
## Step 6: Build & Deploy Inference Pipeline

Use `PipelineModel` with `ModelPackage` objects to deploy an endpoint.

# %%
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
predictor.serializer = CSVSerializer(separator=',')
predictor.deserializer = JSONDeserializer()
print(f"Deployed endpoint: {endpoint_name}")

---

# %% [markdown]
## Step 7: Invoke Endpoint & View Results

Read CSV payload, call `.predict()`, and convert to DataFrame of probabilities.

# %%
with open(input_csv, 'r') as f:
    payload = f.read()
preds = predictor.predict(payload)
cols = [f'model{n}_class{i}' for n in range(1,3) for i in (0,1)]
print(pd.DataFrame(preds, columns=cols).head())

---

# %% [markdown]
## (Optional) Step 8: Cleanup Resources

Remove endpoint and registry group if desired:
```python
# predictor.delete_endpoint()
# sm.delete_model_package_group(ModelPackageGroupName=model_package_group_name)
```
