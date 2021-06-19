"""
A driver script to execute the endpoint and validate if the inference pipeline is working fine
"""
import boto3


# Input in raw format
request_body = "organization  university washington lines     " \
               "nntp posting host  carson u washington edu fair number brave souls" \
               " upgraded si clock oscillator shared experiences poll  please send " \
               "brief message detailing experiences procedure  top speed attained " \
               " cpu rated speed  add cards adapters  heat sinks  hour usage per day" \
               "  floppy disk functionality         floppies especially requested  " \
               "summarizing next two days  please add network knowledge base done clock " \
               "upgrade answered poll  " \
               "thanks  guy kuo  guykuo u washington edu  "


# Create sagemaker client using boto3
client = boto3.client('sagemaker-runtime')

# Specify endpoint (After running the train script this endpoint will be created) and content_type
endpoint_name = "sagemaker-scikit-learn-2021-06-19-10-17-35-659"
content_type = "text/csv"

# Make call to endpoint
response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Body=request_body
    )

# Print out expected and returned labels
print("Expected : comp.sys.mac.hardware")
print("Returned:")
print(response['Body'].read().decode())
