import boto3

sm_client = boto3.client('sagemaker')

response = sm_client.list_endpoints(
    SortBy='CreationTime',
    SortOrder='Descending'
)

for ep in response['Endpoints']:
    print(ep['EndpointName'], "-", ep['CreationTime'])
