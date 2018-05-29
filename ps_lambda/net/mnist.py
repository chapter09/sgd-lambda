import mxnet as mx
import mxnet.ndarray as F
import mxnet.gluon as gluon
from mxnet.gluon import nn


class MnistMlpNet(gluon.Block):
    def __init__(self, **kwargs):
        super(MnistMlpNet, self).__init__(**kwargs)

    def forward(self, data):

        # The first fully-connected layer
        # and the corresponding activation function
        fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
        act1 = mx.sym.Activation(data=fc1, act_type="relu")

        # The second fully-connected layer
        # and the corresponding activation function
        fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
        act2 = mx.sym.Activation(data=fc2, act_type="relu")

        # MNIST has 10 classes
        fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
        # Softmax with cross entropy loss
        mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

        return mlp
