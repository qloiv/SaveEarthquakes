from __future__ import print_function, division

import argparse
import os
from datetime import datetime
from random import randrange
from scipy import signal
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib.dates import date2num
from obspy import UTCDateTime
from pytorch_lightning.loggers import TensorBoardLogger
import h5py
from sklearn.preprocessing import MinMaxScaler
from litdatamodule_distance import LitDataModule
from litnetwork_distance import LitNetwork

cp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog.csv"
wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset.h5"
mp = "/home/viola/WS2021/Code/Models"


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
        #gradient_clip_val=10,
        track_grad_norm = 2,
    )
    trainer.fit(network, datamodule=dm)

    trainer.test()

    #now = datetime.now().strftime("%Y-%m-%d %H:%M")
    #path = "GPD_net_" + str(now) + ".pth"
    #torch.save(network.state_dict(), os.path.join(model_path, path))


def test(catalog_path, waveform_path, checkpoint_path, hparams_file):
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams_file=hparams_file,
        map_location=None,
    )
    dm = LitDataModule(catalog_path, waveform_path)
    # init trainer with whatever options
    trainer = pl.Trainer(gpus=[0])

    # test (pass in the model)
    trainer.test(model, datamodule=dm)


def predict(catalog_path, hdf5_path, checkpoint_path):  # TODO Change to 20sec Window
    sequence_length = 20
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)

    catalog = pd.read_csv(catalog_path)
    train = catalog[catalog["SPLIT"] == "TRAIN"]
    dist = np.append(np.array(train["DIST"]), [1, 600000])
    
    scaler = MinMaxScaler()
    scaler.fit(dist.reshape(-1, 1))
    #train_scaled = scaler.transform(train.reshape(-1, 1))
    
    print(scaler.data_max_)

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
    
    real_output = np.zeros(6000-2000)
    real_labels = np.zeros(6000-2000)
    s_output= np.zeros(6000-2000)
    s_labels= np.zeros(6000-2000)
    for i in tqdm(range(0, 6000 - 20*100)):
        
        
        station_stream = waveform[:, i: i+20*100]
        station_stream = signal.sosfilt(filt, station_stream, axis=-1).astype(
            np.float32
        )
        station_stream = signal.detrend(station_stream)
        station_stream = signal.normalize(station_stream, 1)[0]
        station_stream = torch.from_numpy(station_stream[None])

        out = model(station_stream).squeeze()
        out = np.array(out)
        s_output[i] = out
        s_labels[i] = scaler.transform(distance.reshape(1,-1))
        real_output[i] = scaler.inverse_transform(out.reshape(1,-1))
        real_labels[i] = distance
                                             
    print(real_output)
    print(real_labels)
    print(s_output)
    print(s_labels)
    print(mean_squared_error(real_output,real_labels))
    print(mean_squared_error(s_output,s_labels))
    
    plt.plot(real_output,real_labels,'r')
    plt.savefig("current_plot")
    plt.plot(s_output,s_labels,'b')
    plt.savefig("cr2")


#learn(cp, hp, mp)
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
             waveform_path=args.waveform_path,
             checkpoint_path=args.checkpoint_path,
             hparams_file=args.hparams_file,
             )
    if action == 'predict':
        predict(catalog_path=args.catalog_path,
             hdf5_path = args.hdf5_path,
             checkpoint_path=args.checkpoint_path,
             
             )
