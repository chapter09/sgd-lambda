import mxnet.ndarray as F
import mxnet.gluon as gluon
from mxnet.gluon import nn


class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(6, kernel_size=5)
            self.pool1 = nn.MaxPool2D(pool_size=(2,2))
            self.conv2 = nn.Conv2D(16, kernel_size=5)
            self.pool2 = nn.MaxPool2D(pool_size=(2,2))
            self.fc1 = nn.Dense(120)
            self.fc2 = nn.Dense(84)
            self.fc3 = nn.Dense(10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x