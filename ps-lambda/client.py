import boto3
import json

# load data to S3



client = boto3.client('lambda')


# lambda invoke
def lambda_call(client, data):

    payload = json.dumps({"A": A, "B": B})
    r = client.invoke(FunctionName='matrix_dot', Payload=payload)['Payload']

    result = json.loads(r.read())
    return result


# train
def train(batch_size, lr, epochs, period):
    assert period >= batch_size and period % batch_size == 0
    pass
