import torch.nn
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import (
    Linear,
    ReLU,
    Flatten,
    Sequential,
    Conv1d,
    MaxPool1d,
    BatchNorm1d,
    Softplus,
    MSELoss,
)  # GaussianNLLLoss


class LitNetwork(LightningModule):
    def __init__(self):
        super().__init__()
        self.cnn_layer1 = Sequential(
            Conv1d(3, 32, kernel_size=21, stride=1, padding=10),
            # pytorch hat noch keine padding_mode = same implementation
            BatchNorm1d(32),  # batch norm values taken from keras default
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn_layer2 = Sequential(
            Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
            BatchNorm1d(64),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn_layer3 = Sequential(
            Conv1d(64, 128, kernel_size=11, stride=1, padding=5),
            BatchNorm1d(128),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn_layer4 = Sequential(
            Conv1d(128, 256, kernel_size=9, stride=1, padding=4),
            BatchNorm1d(256),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn_layer5 = Sequential(
            Conv1d(256, 256, kernel_size=7, stride=1, padding=3),
            BatchNorm1d(256),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn_layer6 = Sequential(
            Conv1d(256, 256, kernel_size=5, stride=1, padding=2),
            BatchNorm1d(256),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),
        )

        self.flatten_layer = Flatten()

        self.linear_layers1 = Sequential(
            Linear(256 * 31 + 1, 200),  # the keras network uses 200 units, so...
            # BatchNorm1d(200),
            ReLU(),
        )
        self.linear_layers2 = Sequential(
            Linear(200, 1),
            Softplus(),
        )

    def forward(self, x):
        max_str = x[1]
        x = x[0]
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)
        x = self.cnn_layer5(x)
        x = self.cnn_layer6(x)

        x = self.flatten_layer(x)
        x = torch.cat((x, max_str.view(x.shape[0], 1)), dim=1)
        x = self.linear_layers1(x)
        x = self.linear_layers2(x)
        return x

    def training_step(self, inputs, inputs_idx):
        waveform = inputs["waveform"]
        label = inputs["label"]
        outputs = self(waveform).squeeze()
        x = outputs
        criterion = MSELoss()
        loss = criterion(x, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, inputs, inputs_idx):
        waveform = inputs["waveform"]
        label = inputs["label"]
        outputs = self(waveform).squeeze()
        x = outputs
        criterion = MSELoss()
        loss = criterion(x, label)
        self.log("val_loss", loss)
        return loss

    def test_step(self, inputs, inputs_idx):
        waveform = inputs["waveform"]
        label = inputs["label"]
        outputs = self(waveform).squeeze()
        x = outputs
        criterion = MSELoss()
        loss = criterion(x, label)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)    # 0.001 is default lr
        return optimizer
