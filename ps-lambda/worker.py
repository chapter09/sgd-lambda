import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import ndarray as nd
import numpy as np
import random
import boto3
import json
from mxnet.gluon import ParameterDict, Parameter

from .ps import *


def load_data(s3_url):
    X, y = nd.load(s3_url)
    return X, y


def net(w, b):
    def _net(X):
        return mx.nd.dot(X, w) + b
    return _net


def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
    return params


def train(kv_url, s3_url, lr):
    # load data
    data, label = load_data(s3_url)
    square_loss = gluon.loss.L2Loss()

    # initialize with parameters from KV
    w, b = pull(kv_url)
    net_update = net(w, b)

    # total_loss = [np.mean(square_loss(net(X), y).asnumpy())]

    with autograd.record():
        output = net_update(data)
        loss = square_loss(output, label)
    loss.backward()
    SGD(params, learning_rate)

    w_update =
    b_update =

    push(weight_update, bias_update, kv_url)

    return loss


def lambda_handler(event, context):

    try:
        batch_size = int(event['batch-size'])
        learnng_rate = int(event['learning-rate'])
        kv_url = event['kv-url']
        s3_url = event['s3-url']
        rank = int(event['rank'])

        ret = train(kv_url, s3_url, learnng_rate)

    except KeyError:
        ret = ""
        exit(1)

    out = {
            "headers": {
                "content-type": "application/json",
                "Access-Control-Allow-Origin": "*"
                },
            "body": '{"ret": %s}' % ret,
            "statusCode": 200
          }
    return out
