import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from litdatamodule import LitDataModule
from litnetwork import LitNetwork

catalog_path = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
waveform_path = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
model_path = "/home/viola/WS2021/Code/Models"

network = LitNetwork()
dm = LitDataModule(catalog_path=catalog_path, waveform_path=waveform_path)
logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = pl.Trainer(
    gpus=1,
    logger=logger,
)
trainer.fit(network, datamodule=dm)

trainer.test()

now = datetime.now().strftime("%Y-%m-%d %H:%M")
path = "GPD_net_" + str(now) + ".pth"
torch.save(network.state_dict(), os.path.join(model_path, path))
