import mxnet as mx
from mxnet import autograd
from mxnet import nd, gluon
import logging
import numpy

from ps import push, pull


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

    train_data = gluon.data.DataLoader(
        gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=True)

    params = pull(kv_url)
    # initialize with parameters from KV
    cumulative_loss = 0

    for param in params:
        param.attach_grad()
    # total_loss = [np.mean(square_loss(net(X), y).asnumpy())]

    def net(X):
        return mx.nd.dot(X, params[0]) + params[1]

    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        params = SGD(params, lr, kv_url)
        cumulative_loss += loss.asscalar()

    push([params[0], params[1]], kv_url)
    # print(cumulative_loss / num_batches)

    result = (params[0], params[1], cumulative_loss / num_batches)
    return result


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

    ret = train("s3://ps-lambda-mxnet/w-b-10000",
                "s3://ps-lambda-mxnet/X-y-10000", 4, 1000, 0, 0.001)
    print(ret)

