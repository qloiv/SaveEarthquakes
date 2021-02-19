from __future__ import print_function, division

import argparse
import os
from datetime import datetime
from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib.dates import date2num
from obspy import UTCDateTime
from pytorch_lightning.loggers import TensorBoardLogger

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
        gpus=0,
        logger=logger,
    )
    trainer.fit(network, datamodule=dm)

    trainer.test()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    path = "GPD_net_" + str(now) + ".pth"
    torch.save(network.state_dict(), os.path.join(model_path, path))


def test(catalog_path, waveform_path, checkpoint_path, hparams_file):
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams_file=hparams_file,
        map_location=None,
    )
    dm = LitDataModule(catalog_path, waveform_path)
    # init trainer with whatever options
    trainer = pl.Trainer(gpus=1)

    # test (pass in the model)
    trainer.test(model, datamodule=dm)


def predict(catalog_path, waveform_path, checkpoint_path):  # TODO Change to 20sec Window
    sequence_length = 20
    data = pd.read_csv(catalog_path)
    print(len(data))
    data = data[data["STATION"] != "LVC"]
    events = sorted(data["EVENT"].unique())
    waves = []
    for file in os.listdir(waveform_path):
        if file.endswith(".mseed"):
            waves.append(file.strip(".mseed"))
    intersect = list(set(waves).intersection(set(events)))
    data = data[data["EVENT"].isin(intersect)]

    idx = randrange(0, len(data))
    event, station, p_pick, time = data.iloc[idx][
        ["EVENT", "STATION", "P_PICK", "TIME"]
    ]  # TODO time means start time of event?
    waveform = obspy.read(os.path.join(waveform_path, f"{event}.mseed"))
    station_stream = waveform.select(station=station, channel="HH*")  # high frequency
    while not station_stream:
        idx = randrange(0, len(data))
        event, station, p_pick, time = data.iloc[idx][
            ["EVENT", "STATION", "P_PICK", "TIME"]
        ]  # TODO time means start time of event?
        waveform = obspy.read(os.path.join(waveform_path, f"{event}.mseed"))
        station_stream = waveform.select(station=station, channel="HH*")  # high frequency

    station_stream.plot()
    # to prevent loading the entire catalog, choose a random event and check, whether it is in the waveform folder

    # print now maybe with highlight on the P Pick
    # maybe slice out with beginning on start time?
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,

    )
    model.freeze()

    out_times = np.arange(
        station_stream[0].stats.starttime, station_stream[0].stats.endtime + 1, 1
    )
    outs = np.zeros(len(out_times))
    for i in range(0, len(out_times) - 4):
        station_stream = waveform.select(station=station, channel="HH*")  # reload
        window = station_stream.slice(starttime=out_times[i], endtime=out_times[i] + 4)
        window.filter("highpass", freq=2, zerophase=True)
        window.detrend().normalize()

        # for windowed_st in station_stream.slide(window_length=sequence_length, step=1):
        #     window = windowed_st.copy()
        #     window.detrend().normalize()
        trace_z = np.float32(window[0])
        trace_n = np.float32(window[1])
        trace_e = np.float32(window[2])
        example = np.stack((trace_z, trace_n, trace_e))[:, 0:-1]
        example = torch.from_numpy(example[None])
        output = model(example)
        _, predicted = torch.max(output, 1)

        outs[i: i + 4] = outs[i: i + 4] + np.full(4, predicted[0])
        # print(outs)
    #    print(windowed_st)

    #    print("---")

    print(outs)
    out_times = date2num(out_times)

    tr1 = station_stream[0]
    tr2 = station_stream[1]
    tr3 = station_stream[2]

    fig, ax = plt.subplots(4, sharex=True)
    ax[0].plot(tr1.times("matplotlib"), tr1.data, color="black", linewidth=0.4)
    ax[0].xaxis_date()
    ax[0].axvline(date2num(UTCDateTime(p_pick)), color="blue")

    ax[1].plot(tr2.times("matplotlib"), tr2.data, color="black", linewidth=0.4)
    ax[1].xaxis_date()
    ax[1].axvline(date2num(UTCDateTime(p_pick)), color="blue")

    ax[2].plot(tr3.times("matplotlib"), tr3.data, color="black", linewidth=0.4)
    ax[2].xaxis_date()
    ax[2].axvline(date2num(UTCDateTime(p_pick)), color="blue")

    ax[3].plot(out_times, outs, "r-")
    ax[3].xaxis_date()
    fig.autofmt_xdate()
    plt.show()
    plt.savefig("current_plot")


learn(cp, hp, mp)
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
        test(catalog_path=args.catalog_path,
             waveform_path=args.waveform_path,
             checkpoint_path=args.checkpoint_path,
             hparams_file=args.hparams_file,
             )
