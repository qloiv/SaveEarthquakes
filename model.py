import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, MaxPool2d, CrossEntropyLoss, Flatten, Sequential, Conv1d, MaxPool1d, Module, Softmax, \
    BatchNorm1d, Dropout
from torch.optim import Adam, SGD


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layer10 = Sequential(
            # pytorch hat noch keine padding_mode = same implementation
            Conv1d(3, 32, kernel_size=21, stride=1, padding=10))
        self.cnn_layer11 = Sequential(
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

            Linear(200, 200),
            BatchNorm1d(200, momentum=0.99, eps=0.001),
            ReLU(),

            Linear(200, 3),
            BatchNorm1d(3, momentum=0.99, eps=0.001),
            ReLU()
        )

    def forward(self, x):
        x = self.cnn_layer10(x)
        print(list(x.size()))
        x = self.cnn_layer11(x)
        print(list(x.size()))
        x = self.cnn_layer2(x)
        print(list(x.size()))
        x = self.cnn_layer3(x)
        print(list(x.size()))
        x = self.cnn_layer4(x)
        print(list(x.size()))
        x = self.linear_layers(x)
        print(list(x.size()))
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)
y = torch.randn(2, 3, 400)
yn = net(y)
