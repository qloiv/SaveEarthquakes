import torch.nn as nn
from torch.backends import cudnn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Flatten, Sequential, Conv1d, MaxPool1d, BatchNorm1d
from torch.optim import Adam
import numpy as np
from dataset import *


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


# y = torch.randn(2, 3, 400)
# yn = net(y)
# print(yn)
catalog_path = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
waveform_path = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

trainloader, evalloader = get_data_loaders(catalog_path, waveform_path,
                                           batch_size=64, num_workers=4, shuffle=True, test_run=True)
net = Net()
net.to(device)
optimizer = Adam(net.parameters())
# defining the loss function
criterion = CrossEntropyLoss()
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, inputs in enumerate(trainloader):
        waveform = inputs['waveform']
        label = inputs['label']
        waveform, label = waveform.to(device), label.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(waveform)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
