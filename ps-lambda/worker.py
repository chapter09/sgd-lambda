import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet import gluon
from mxnet import ndarray as nd
import numpy as np
import random
import boto3
import json


def grad():
    pass


def push(w, b):
    pass


def pull():
    w = None
    b = None
    return w, b


def update_w():
    pass


def load_data(batch_size):
    dataset = None
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)


def train(batch_size, data_iter, lr, period):

    weight, bias = pull()

    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    square_loss = gluon.loss.L2Loss()

    net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    # total_loss = [np.mean(square_loss(net(X), y).asnumpy())]

    for batch_i, (data, label) in enumerate(data_iter):
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)

        # if batch_i * batch_size % period == 0:
        #     total_loss.append(np.mean(square_loss(net(X), y).asnumpy()))
    weight_update = np.reshape(net[0].weight.data().asnumpy(), (1, -1))
    bias_update = net[0].bias.data().asnumpy()[0]

    push(weight_update, bias_update)


def lambda_handler(event, context):

    batch_size = int(event['batch-size'])

    try:
        # API Gateway GET method

        if event['httpMethod'] == 'GET':
            url = event['queryStringParameters']['url']
        #API Gateway POST method
        elif event['httpMethod'] == 'POST':
            data = json.loads(event['body'])
            if data.has_key('dataurl'):
                data_url = data['dataurl']
            else:
                url = data['url']

    except KeyError:
        # direct invocation
        url = event['url']

    out = {
            "headers": {
                "content-type": "application/json",
                "Access-Control-Allow-Origin": "*"
                },
            "body": '{"address": "%s", "latlng": "%s"}' % (),
            "statusCode": 200
          }
    return out
