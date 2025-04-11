import sagemaker

sagemaker_session = sagemaker.Session()
default_bucket = sagemaker_session.default_bucket()
print("Default bucket:", default_bucket)
