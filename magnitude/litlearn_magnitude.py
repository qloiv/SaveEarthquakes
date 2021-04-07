from __future__ import print_function, division

import argparse
from random import randrange

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import signal
from tqdm import tqdm

from litdatamodule_magnitude import LitDataModule
from litnetwork_magnitude import LitNetwork
from utils import normalize_stream, obspy_detrend

cp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog.csv"
wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset.h5"
mp = "/home/viola/WS2021/Code/Models"
chp = "/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/magnitude/version_33/checkpoints/epoch=15-step=511.ckpt"


# checkpoint_path = "/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_8/checkpoints/epoch=33-step=3093.ckpt",
# hparams_file = "/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_8/hparams.yaml",
# map_location = None,

class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)

def learn(catalog_path, hdf5_path, model_path):
    network = LitNetwork()
    dm = LitDataModule(catalog_path=catalog_path, hdf5_path=hdf5_path)
    logger = TensorBoardLogger("../tb_logs", name="magnitude")
    checkpoint_callback = ModelCheckpoint()
    ch2 = ModelCheckpointAtEpochEnd()
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=[0],
        logger=logger,
    )
    trainer.fit(network, datamodule=dm)

    trainer.test()

    # now = datetime.now().strftime("%Y-%m-%d %H:%M")
    # path = "GPD_net_" + str(now) + ".pth"
    # torch.save(network.state_dict(), os.path.join(model_path, path))


def test(catalog_path, hdf5_path, checkpoint_path, hparams_file):
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams_file=hparams_file,
        map_location=None,
    )
    dm = LitDataModule(catalog_path, hdf5_path)
    # init trainer with whatever options
    trainer = pl.Trainer()

    # test (pass in the model)
    trainer.test(model, datamodule=dm)


def predict(catalog_path, hdf5_path, checkpoint_path):
    sequence_length = 20
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)

    catalog = pd.read_csv(catalog_path)

    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,

    )
    model.freeze()

    test = catalog[catalog["SPLIT"] == "TEST"]
    idx = randrange(0, len(test))
    event, station, ma, ml = test.iloc[idx][["EVENT", "STATION", "MA", "ML"]]
    print("MA",ma,"ML",ml)
    waveform = np.array(h5data.get(event + "/" + station))
    filt = signal.butter(
        2, 2, btype="highpass", fs=100, output="sos"
    )

    output = np.zeros(6000 - 2000)
    labels = np.zeros(6000 - 2000)
    labels.fill(ml)
    for i in tqdm(range(0, 6000 - 20 * 100)):
        station_stream = waveform[:, i: i + 20 * 100]
        d0 = obspy_detrend(station_stream[0])
        d1 = obspy_detrend(station_stream[1])
        d2 = obspy_detrend(station_stream[2])
        f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
        f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
        f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)
        station_stream = np.stack((f0, f1, f2))
        station_stream, _ = normalize_stream(station_stream)
        station_stream = torch.from_numpy(station_stream[None])
        out = model(station_stream)
        _, predicted = torch.max(out.data, 1)
        output[i] = predicted

    t = np.linspace(1, 4000, num=4000)
    fig, axs = plt.subplots(5)
    fig.suptitle("Predict Plot")
    axs[0].plot(t, labels, 'r')
    axs[0].plot(t, output, 'g')
    axs[0].axvline(2001, color="blue")

    axs[1].axvline(3001, color="blue")

    axs[2].plot(waveform[0], 'r')
    axs[3].plot(waveform[1], 'b')
    axs[1].plot(waveform[2], 'g')

    # plt.plot(t,mean_squared_error(s_output,s_labels),":")
    fig.savefig("predict plot")


# learn(cp,hp,mp)
#predict(cp, hp, chp)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, required=True)
    parser.add_argument('--catalog_path', type=str)
    parser.add_argument('--hdf5_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--hparams_file', type=str)
    args = parser.parse_args()
    action = args.action

    if action == 'learn':
        learn(catalog_path=args.catalog_path,
              hdf5_path=args.hdf5_path,
              model_path=args.model_path,
              )
    if action == 'test':
        test(catalog_path=args.catalog_path,
              hdf5_path=args.hdf5_path,
             checkpoint_path=args.checkpoint_path,
             hparams_file=args.hparams_file,
             )
    if action == 'predict':
        predict(catalog_path=args.catalog_path,
                hdf5_path=args.hdf5_path,
                checkpoint_path=args.checkpoint_path,

                )
