import pytorch_lightning as pl

from litdatamodule import LitDataModule
from litnetwork import LitNetwork

catalog_path = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
waveform_path = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
model_path = "/home/viola/WS2021/Code/Models"

model = LitNetwork.load_from_checkpoint(
    checkpoint_path="/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_8/checkpoints/epoch=33-step=3093.ckpt",
    hparams_file="/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_8/hparams.yaml",
    map_location=None,
)
dm = LitDataModule(catalog_path, waveform_path)
# init trainer with whatever options
trainer = pl.Trainer(gpus=1)

# test (pass in the model)
trainer.test(model, datamodule=dm)
