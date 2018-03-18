import mxnet as mx
from mxnet import autograd
from mxnet import nd

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


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
    return params


def train(kv_url, s3_url, batch_size, rank, lr):
    data_ctx = mx.cpu()
    model_ctx = mx.cpu()

    # load data
    X, y = load_data(s3_url, batch_size, rank)
    # sq_loss = gluon.loss.L2Loss()

    # initialize with parameters from KV
    w, b = pull(kv_url)

    params = [w, b]
    for param in params:
        param.attach_grad()
    # total_loss = [np.mean(square_loss(net(X), y).asnumpy())]

    def net(X):
        return mx.nd.dot(X, w) + b

    with autograd.record():
        output = net(X.as_in_context(model_ctx))
        loss = square_loss(output, y.as_in_context(model_ctx))
    loss.backward()
    params = SGD(params, lr)
    print(params)
    print(loss.asscalar())

    push([params[0], params[1]], kv_url)

    return loss.asscalar()


def lambda_handler(event, context):

    batch_size = int(event['batch-size'])
    learning_rate = float(event['learning-rate'])
    kv_url = event['kv-url'].split('/')[-1]
    s3_url = event['s3-url'].split('/')[-1]
    rank = int(event['rank'])

    ret = train(kv_url, s3_url, batch_size, rank, learning_rate)
    return {
        "headers": {
            "content-type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": '{"loss": %s}' % ret,
        "statusCode": 200
    }




if __name__ == '__main__':

    train("s3://ps-lambda-mxnet/w-b-10000",
          "s3://ps-lambda-mxnet/X-y-10000",
          1000, 1, 0.0001)
