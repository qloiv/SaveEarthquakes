import pytorch_lightning as pl
import torch
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import (
    Linear,
    ReLU,
    Flatten,
    Sequential,
    Conv1d,
    MaxPool1d,
    BatchNorm1d,
    CrossEntropyLoss,
)

from load import SeismoDataset, DataLoader

catalog_path = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
waveform_path = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
model_path = "/home/viola/WS2021/Code/Models"
criterion = CrossEntropyLoss()


class Net(LightningModule):
    def __init__(self):
        super().__init__()
        self.test_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

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
        loss = criterion(outputs, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, inputs, inputs_idx):
        waveform = inputs["waveform"]
        label = inputs["label"]
        outputs = self(waveform)
        loss = criterion(outputs, label)
        self.log("val_loss", loss)

    def test_step(self, inputs, inputs_idx):
        waveform = inputs["waveform"]
        label = inputs["label"]
        outputs = self(waveform)
        loss = criterion(outputs, label)
        _, predicted = torch.max(outputs.data, 1)
        self.test_acc(predicted, label)
        self.log('test_acc', self.test_acc)

    #        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    # TODO maybe put something in prepare data def prepare_data(self):
    def train_dataloader(self):
        batch_size = 64
        num_workers = 4
        shuffle = True
        test_run = False

        if test_run:
            num_workers = 1

        training_data = SeismoDataset(
            catalog_path=catalog_path,
            waveform_path=waveform_path,
            split="TRAIN",
            test_run=test_run,
        )
        training_loader = DataLoader(
            training_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
        return training_loader

    def val_dataloader(self):
        batch_size = 64
        num_workers = 4
        test_run = False
        validation_data = SeismoDataset(
            catalog_path=catalog_path,
            waveform_path=waveform_path,
            split="DEV",
            test_run=test_run,
        )

        validation_loader = DataLoader(
            validation_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        return validation_loader

    def test_dataloader(self):
        batch_size = 64
        num_workers = 4
        test_run = False
        if test_run:
            num_workers = 1
        test_data = SeismoDataset(
            catalog_path=catalog_path,
            waveform_path=waveform_path,
            split="TEST",
            test_run=test_run,
        )

        test_loader = DataLoader(
            test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

        return test_loader
