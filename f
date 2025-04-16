from sagemaker import image_uris

image_uri = image_uris.retrieve(
    framework='sklearn',
    region='us-east-1',
    version='1.2-1',  # or any version compatible with your model
    image_scope='inference'
)

print(image_uri)
