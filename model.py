import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, MaxPool2d, CrossEntropyLoss, Flatten, Sequential, Conv1d, MaxPool1d, Module, Softmax, \
    BatchNorm1d, Dropout
from torch.optim import Adam, SGD
import numpy as np

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

# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters())
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print(model)

pick_x = np.load('train_array.npy')[0:800]
noise_x = np.load('train_noise.npy')[0:800]
index = np.int64(np.concatenate((np.zeros(noise_x.shape[0]), np.ones(pick_x.shape[0]))))
train_examples = np.concatenate((noise_x, pick_x))

shuffler = np.random.permutation(len(index))

train_x = train_examples[shuffler]
train_y = index[shuffler]

pick_x_test = np.load('test_array.npy')[0:200]
noise_x_test = np.load('test_noise.npy')[0:200]
index_noise = np.int64(np.concatenate((np.zeros(noise_x_test.shape[0])
                                       , np.ones(pick_x_test.shape[0]))))
test_examples = np.concatenate((noise_x_test, pick_x_test))

shuffler = np.random.permutation(len(index_noise))

test_x = test_examples[shuffler]
test_y = index_noise[shuffler]
# defining the number of epochs
n_epochs = 5
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
for epoch in range(n_epochs):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = torch.from_numpy(train_x), torch.from_numpy(train_y)
    # getting the validation set
    x_test, y_test = torch.from_numpy(test_x), torch.from_numpy(test_y)
    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_test = x_test.cuda()
        y_test = y_test.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    # prediction for training and validation set
    output_train = model(x_train)
    output_test = model(x_test)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_test = criterion(output_test, y_test)
    train_losses.append(loss_train)
    val_losses.append(loss_test)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    print('Epoch : ', epoch + 1, '\t', 'loss :', loss_test, 'train_loss', loss_train)
