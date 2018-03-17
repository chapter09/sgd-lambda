import boto3
import json
import mxnet as mx
from mxnet import nd, gluon
from threading import Thread
import argparse

from ps import push, pull

# load data to S3

client = boto3.client('lambda')


def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2


def gen_data(num_examples, num_inputs):
    X = nd.random_normal(shape=(num_examples, num_inputs))
    noise = .1 * nd.random_normal(shape=(num_examples,))
    y = real_fn(X) + noise
    return X, y


# lambda invoke
def lambda_call(client, payload):

    r = client.invoke(FunctionName='worker', Payload=payload)['Payload']

    result = json.loads(r.read())
    return result


# upload data
def upload_input_data(data, s3_url):
    push(data, s3_url)
    return s3_url


# train
def train(batch_size, num_lambda, lr, epochs, s3_url, kv_url):

    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    # square_loss = gluon.loss.L2Loss()

    weight, bias = mx.init.Normal(sigma=1)
    push(weight, bias)

    lambda_client = boto3.client('lambda')

    for epoch in range(1, epochs + 1):
        for rank in range(0, num_lambda):
            payload = json.dumps({
                "batch-size": batch_size,
                "learning-rate": lr,
                "s3-url": s3_url,
                "kv-url": kv_url,
                "rank": rank
            })

            lambda_call(lambda_client, payload)
            break
        break

    # total_loss = [np.mean(square_loss(net(X), y).asnumpy())]

    pass


def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', dest='is_data_ready', default=True,
                        action='store_false', help='is data ready in S3')
    args = parser.parse_args()

    epochs = 10
    learning_rate = .0001
    batch_size = 100

    num_inputs = 2
    num_outputs = 1
    num_examples = 10000

    kv_url = "s3://ps-lambda-mxnet/w-b-%d" % num_examples
    s3_url = "s3://ps-lambda-mxnet/X-y-%d" % num_examples

    num_lambda = num_examples / batch_size

    X, y = gen_data(num_examples, num_inputs)

    if not args.is_data_ready:
        upload_input_data([X, y], s3_url)

    # init params
    w = nd.random_normal(shape=(num_inputs, num_outputs))
    b = nd.random_normal(shape=num_outputs)

    # push params to kvstore
    push([w, b], kv_url)

    train(batch_size, num_lambda, learning_rate, epochs, s3_url, kv_url)


if __name__ == '__main__':
    main()
