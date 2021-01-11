import torch.nn as nn
from torch.backends import cudnn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Flatten, Sequential, Conv1d, MaxPool1d, BatchNorm1d
from torch.optim import Adam
import numpy as np
from load import *
from datetime import datetime
from network import Net

# y = torch.randn(2, 3, 400)
# yn = net(y)
# print(yn)
catalog_path = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
waveform_path = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
model_path = "/home/viola/WS2021/Code/Models"
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

trainloader, evalloader = get_data_loaders(catalog_path, waveform_path,
                                           batch_size=64, num_workers=4, shuffle=True, test_run=False)
net = Net()
net.to(device)
optimizer = Adam(net.parameters())
# defining the loss function
criterion = CrossEntropyLoss()
print("Start Training")
for epoch in range(10):  # loop over the dataset multiple times

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
        if i % 200 == 199:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')
testloader = get_test_loader(catalog_path,waveform_path,test_run=False)
correct = 0
total = 0
with torch.no_grad():
    for inputs in testloader:
        waveform = inputs['waveform']
        label = inputs['label']
        waveform, label = waveform.to(device), label.to(device)
        outputs = net(waveform)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

now = datetime.now().strftime("%Y-%m-%d %H:%M")
path = 'GPD_net_' + str(now) + '.pth'
torch.save(net.state_dict(), os.path.join(model_path,path))