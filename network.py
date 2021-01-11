import torch.nn as nn
from torch.nn import Linear, ReLU, Flatten, Sequential, Conv1d, MaxPool1d, BatchNorm1d


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layer1 = Sequential(
            Conv1d(3, 32, kernel_size=21, stride=1, padding=10),
            # pytorch hat noch keine padding_mode = same implementation
            BatchNorm1d(32, momentum=0.99, eps=0.001),  # batch norm values taken from keras default
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2))
        self.cnn_layer2 = Sequential(
            Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
            BatchNorm1d(64, momentum=0.99, eps=0.001),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2))
        self.cnn_layer3 = Sequential(
            Conv1d(64, 128, kernel_size=11, stride=1, padding=5),
            BatchNorm1d(128, momentum=0.99, eps=0.001),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2))
        self.cnn_layer4 = Sequential(
            Conv1d(128, 256, kernel_size=9, stride=1, padding=4),
            BatchNorm1d(256, momentum=0.99, eps=0.001),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
            Flatten())

        self.linear_layers = Sequential(
            Linear(25 * 256, 200),  # the keras network uses 200 units, so...
            BatchNorm1d(200, momentum=0.99, eps=0.001),
            ReLU(),

            Linear(200, 2),
            BatchNorm1d(2, momentum=0.99, eps=0.001),
            ReLU()
        )

    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)
        x = self.linear_layers(x)
        return x

