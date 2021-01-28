import pytorch_lightning as pl
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torch.nn import Linear, ReLU, Flatten, Sequential, Conv1d, MaxPool1d, BatchNorm1d


class LitNetwork(LightningModule):
    def __init__(self):
        super().__init__()
        self.cnn_layer1 = Sequential(
            Conv1d(3, 32, kernel_size=21, stride=1, padding=10),
            # pytorch hat noch keine padding_mode = same implementation
            BatchNorm1d(
                32, momentum=0.99, eps=0.001
            ),  # batch norm values taken from keras default
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn_layer2 = Sequential(
            Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
            BatchNorm1d(64, momentum=0.99, eps=0.001),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn_layer3 = Sequential(
            Conv1d(64, 128, kernel_size=11, stride=1, padding=5),
            BatchNorm1d(128, momentum=0.99, eps=0.001),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn_layer4 = Sequential(
            Conv1d(128, 256, kernel_size=9, stride=1, padding=4),
            BatchNorm1d(256, momentum=0.99, eps=0.001),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
            Flatten(),
        )

        self.linear_layers = Sequential(
            Linear(25 * 256, 200),  # the keras network uses 200 units, so...
            BatchNorm1d(200, momentum=0.99, eps=0.001),
            ReLU(),
            Linear(200, 2),
            BatchNorm1d(2, momentum=0.99, eps=0.001),
            ReLU(),
        )

        self.test_acc = pl.metrics.Accuracy()
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)
        x = self.linear_layers(x)
        return x

    def training_step(self, inputs, inputs_idx):
        waveform = inputs["waveform"]
        label = inputs["label"]
        outputs = self(waveform)
        criterion = CrossEntropyLoss()
        loss = criterion(outputs, label)
        _, predicted = torch.max(outputs.data, 1)
        self.train_acc(predicted, label)
        self.log("train_acc", self.train_acc)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, inputs, inputs_idx):
        waveform = inputs["waveform"]
        label = inputs["label"]
        outputs = self(waveform)
        criterion = CrossEntropyLoss()
        loss = criterion(outputs, label)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc)

    def test_step(self, inputs, inputs_idx):
        waveform = inputs["waveform"]
        label = inputs["label"]
        outputs = self(waveform)
        criterion = CrossEntropyLoss()
        loss = criterion(outputs, label)
        _, predicted = torch.max(outputs.data, 1)
        self.test_acc(predicted, label)
        self.log("test_acc", self.test_acc)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
