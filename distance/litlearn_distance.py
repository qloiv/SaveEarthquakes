from __future__ import print_function, division

import argparse
from random import randrange

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import signal
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from litdatamodule_distance import LitDataModule
from litnetwork_distance import LitNetwork

cp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog.csv"
wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset.h5"
mp = "/home/viola/WS2021/Code/Models"
chp = "/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/distance/version_5/checkpoints/epoch=20-step=671.ckpt"


# checkpoint_path = "/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_8/checkpoints/epoch=33-step=3093.ckpt",
# hparams_file = "/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_8/hparams.yaml",
# map_location = None,


def learn(catalog_path, hdf5_path, model_path):
    network = LitNetwork()
    dm = LitDataModule(catalog_path=catalog_path, hdf5_path=hdf5_path)
    logger = TensorBoardLogger("../tb_logs", name="distance")
    trainer = pl.Trainer(
        gpus=[0],
        logger=logger,
        gradient_clip_val=1,
        track_grad_norm=2,
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
    trainer = pl.Trainer(gpus=[0])

    # test (pass in the model)
    trainer.test(model, datamodule=dm)


def normalize_stream(stream, global_max=False):
    if global_max is True:
        ma = np.abs(stream).max()
        stream /= ma
    else:
        for tr in stream:
            ma_tr = np.abs(tr).max()
            tr /= ma_tr
    return stream


def predict(catalog_path, hdf5_path, checkpoint_path):  # TODO Change to 20sec Window
    sequence_length = 20
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)

    catalog = pd.read_csv(catalog_path)
    train = catalog[catalog["SPLIT"] == "TRAIN"]
    dist = np.append(np.array(train["DIST"]), [1, 600000])

    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaler.fit(dist.reshape(-1, 1))
    # train_scaled = scaler.transform(train.reshape(-1, 1))

    # print(scaler.data_max_)

    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,

    )
    model.freeze()

    test = catalog[catalog["SPLIT"] == "TEST"]
    idx = randrange(0, len(test))
    event, station, distance = test.iloc[idx][["EVENT", "STATION", 'DIST']]
    print(distance)
    print(scaler.transform(distance.reshape(1, -1)))

    waveform = np.array(h5data.get(event + "/" + station))
    filt = signal.butter(
        2, 2, btype="highpass", fs=100, output="sos"
    )

    real_output = np.zeros(6000 - 2000)
    real_labels = np.zeros(6000 - 2000)
    s_output = np.zeros(6000 - 2000)
    s_labels = np.zeros(6000 - 2000)
    for i in tqdm(range(0, 6000 - 20 * 100)):
        station_stream = waveform[:, i: i + 20 * 100]
        station_stream = signal.sosfilt(filt, station_stream, axis=-1).astype(
            np.float32
        )
        station_stream = signal.detrend(station_stream)
        station_stream = normalize_stream(station_stream)
        station_stream = torch.from_numpy(station_stream[None])

        out = model(station_stream).squeeze()
        out = np.array(out)
        s_output[i] = out
        s_labels[i] = scaler.transform(distance.reshape(1, -1))
        real_output[i] = scaler.inverse_transform(out.reshape(1, -1))
        real_labels[i] = distance

    print(real_output)
    print(real_labels)
    print(s_output)
    print(s_labels)
    print(mean_squared_error(real_output, real_labels))
    print(mean_squared_error(s_output, s_labels))
    t = np.linspace(1, 4000, num=4000)
    fig, axs = plt.subplots(4)
    fig.suptitle("Predict Plot")
    axs[0].plot(t, real_labels, 'r')
    axs[0].plot(t, real_output, 'g')
    # plt.plot(t,mean_squared_error(real_output,real_labels),":")
    axs[0].axvline(2001, color="blue")

    axs[1].axvline(3001, color="blue")
    axs[1].plot(waveform[0], 'r')
    axs[2].plot(waveform[1], 'b')
    axs[3].plot(waveform[2], 'g')

    # plt.plot(t,mean_squared_error(s_output,s_labels),":")
    fig.savefig("predict plot")


predict(cp, hp, chp)
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
