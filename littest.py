import pytorch_lightning as pl

from litnetwork import LitNetwork

model = LitNetwork.load_from_checkpoint(
    checkpoint_path="/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_2/checkpoints/epoch=20-step=1910.ckpt",
    hparams_file="/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_2/hparams.yaml",
    map_location=None,
)

# init trainer with whatever options
trainer = pl.Trainer(gpus=1)

# test (pass in the model)
trainer.test(model)
