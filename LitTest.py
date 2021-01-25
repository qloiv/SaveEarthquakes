import pytorch_lightning as pl

from LitNetwork import Net

model = Net.load_from_checkpoint(
    checkpoint_path='/home/viola/WS2021/Code/SaveEarthquakes/lightning_logs/version_4/checkpoints/epoch=61-step=5641.ckpt',
    hparams_file='/home/viola/WS2021/Code/SaveEarthquakes/lightning_logs/version_4/hparams.yaml',
    map_location=None
)

# init trainer with whatever options
trainer = pl.Trainer(gpus=1)

# test (pass in the model)
trainer.test(model)
