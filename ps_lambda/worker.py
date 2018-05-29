import mxnet as mx
from mxnet import autograd
from mxnet import nd, gluon
import logging
import numpy as np

from ps import push, pull
from net import general_net


def load_data(s3_url, batch_size, rank):
    X, y = pull(s3_url)
    offset = rank * batch_size
    if (offset + batch_size) < len(X):
        return X[offset:offset+batch_size], y[offset:offset+batch_size]
    else:
        return X[offset:], y[offset:]


def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)


def SGD(params, lr, kv_url):
    ps_params = pull(kv_url)

    for i in range(0, len(params)):
        params[i][:] = ps_params[i] - lr * params[i].grad

    # print(params[0], params[1])
    push(params, kv_url)
    return params


def train(kv_url, s3_url, batch_size, lambda_size, rank, lr):
    data_ctx = mx.cpu()
    model_ctx = mx.cpu()

    # load data
    X, y = load_data(s3_url, lambda_size, rank)
    num_batches = y.shape[0] / batch_size

    params = pull(kv_url)
    # initialize with parameters from KV
    cumulative_loss = 0

    net = general_net.Net()
    # Initialize on CPU. Replace with `mx.gpu(0)`, or `[mx.gpu(0), mx.gpu(1)]`,
    # etc to use one or more GPUs.

    for i in range(0, 1):

        net.load_params('./params', ctx=mx.cpu())

        with autograd.record():
            output = net(X)
            L = gluon.loss.SoftmaxCrossEntropyLoss()
            loss = L(output, y)
        loss.backward()

        print('loss:', loss)
        for p in net.collect_params().values():
            print(p.name, p.data())
        # print('grad:', net.fc1.weight.grad())

        for p in net.collect_params().values():
            p.data()[:] -= lr / X.shape[0] * p.grad()

        net.save_params("./params")

    # push(params, kv_url)
    # print(cumulative_loss / num_batches)

    # result = (params[0], params[1], cumulative_loss / num_batches)
    return None


def lambda_handler(event, context):

    batch_size = int(event['batch-size'])
    learning_rate = float(event['learning-rate'])
    kv_url = event['kv-url'].split('/')[-1]
    s3_url = event['s3-url'].split('/')[-1]
    rank = int(event['rank'])
    lambda_size = int(event['lambda_size'])

    result = train(kv_url, s3_url, batch_size, lambda_size, rank, learning_rate)

    return {
        "headers": {
            "content-type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": '{"loss": %f}' % (result[2]),
        "statusCode": 200
    }


if __name__ == '__main__':

    ret = train("s3://ps-lambda-mxnet/w-b-10",
                "s3://ps-lambda-mxnet/X-y-10", 4, 1000, 0, 0.01)
    print(ret)

