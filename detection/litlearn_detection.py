from __future__ import print_function, division

import argparse
from random import randrange

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import signal

from litdatamodule_detection import LitDataModule
from litnetwork_detection import LitNetwork
from utils import *

cp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog.csv"
wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset.h5"
mp = "/home/viola/WS2021/Code/Models"


def learn(catalog_path, hdf5_path, model_path):
    network = LitNetwork()
    dm = LitDataModule(catalog_path=catalog_path, hdf5_path=hdf5_path)
    logger = TensorBoardLogger("../tb_logs", name="detection")
    trainer = pl.Trainer(
        gpus=[0],
        logger=logger,
        # precision=16,
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


def test_one(catalog_path, checkpoint_path, hdf5_path):
    # load catalog with random test event
    catalog = pd.read_csv(catalog_path)
    test = catalog[catalog["SPLIT"] == "TEST"]
    idx = randrange(0, len(test))
    print(idx)
    event, station, p_pick = test.iloc[idx][["EVENT", "STATION", 'P_PICK']]

    # load network
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.freeze()

    raw_waveform = np.array(h5data.get(event + "/" + station))
    seq_len = 4 * 100  # *sampling rate
    p_pick_array = 3001
    random_point = np.random.randint(seq_len)
    waveform = raw_waveform[:, p_pick_array - random_point: p_pick_array + (seq_len - random_point)]
    fig, axs = plt.subplots(3)
    fig.suptitle("Input of Detection Network - full Trace")
    axs[0].plot(raw_waveform[0], 'r')
    axs[1].plot(raw_waveform[1], 'b')
    axs[2].plot(raw_waveform[2], 'g')
    fig.savefig("TestOne:Full Trace")

    fig, axs = plt.subplots(3)
    fig.suptitle("Cut Out Input")
    axs[0].plot(waveform[0], 'r')
    axs[1].plot(waveform[1], 'b')
    axs[2].plot(waveform[2], 'g')
    fig.savefig("TestOne: Input")
    d0 = obspy_detrend(waveform[0])
    d1 = obspy_detrend(waveform[1])
    d2 = obspy_detrend(waveform[2])
    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending")
    axs[0].plot(d0, 'r')
    axs[1].plot(d1, 'b')
    axs[2].plot(d2, 'g')
    fig.savefig("TestOne:Detrended")

    # set filter
    filt = signal.butter(
        2, 2, btype="highpass", fs=100, output="sos"
    )
    f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
    f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
    f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)
    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending, then Filtering")
    axs[0].plot(f0, 'r')
    axs[1].plot(f1, 'b')
    axs[2].plot(f2, 'g')
    fig.savefig("TestOne:Detrended and Filtered")
    waveform = np.stack((f0, f1, f2))
    waveform, _ = normalize_stream(waveform)
    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending->Filtering->Normalizing")
    axs[0].plot(waveform[0], 'r')
    axs[1].plot(waveform[1], 'b')
    axs[2].plot(waveform[2], 'g')
    fig.savefig("TestOne: Detrended, Filtered and Normalized")
    station_stream = torch.from_numpy(waveform[None])
    out = model(station_stream)
    _, predicted = torch.max(out, 1)
    print(predicted)
    fig, axs = plt.subplots(3)
    fig.suptitle("Modified data with P-Pick, was detected as P-Wave? " + str(bool(predicted)))
    axs[0].plot(waveform[0], 'r')
    axs[1].plot(waveform[1], 'b')
    axs[2].plot(waveform[2], 'g')
    axs[0].axvline(random_point, color="black")
    axs[1].axvline(random_point, color="black")
    axs[2].axvline(random_point, color="black")

    fig.savefig("TestOne: Results")


def predict(catalog_path, checkpoint_path, hdf5_path):
    # load catalog with random test event
    catalog = pd.read_csv(catalog_path)
    test = catalog[catalog["SPLIT"] == "TEST"]
    idx = randrange(0, len(test))
    print(idx)
    event, station, p_pick = test.iloc[idx][["EVENT", "STATION", 'P_PICK']]

    # load network
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.freeze()

    raw_waveform = np.array(h5data.get(event + "/" + station))
    filt = signal.butter(
        2, 2, btype="highpass", fs=100, output="sos"
    )
    outs = np.zeros(len(raw_waveform[0]))
    for i in range(0, len(raw_waveform[0]) - 400):
        raw_waveform = np.array(h5data.get(event + "/" + station))  # reload stream
        window = raw_waveform[:, i:i + 400]
        d0 = obspy_detrend(window[0])
        d1 = obspy_detrend(window[1])
        d2 = obspy_detrend(window[2])

        f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
        f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
        f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)
        station_stream = np.stack((f0, f1, f2))
        station_stream, _ = normalize_stream(station_stream)
        station_stream = torch.from_numpy(station_stream[None])
        out = model(station_stream)
        _, predicted = torch.max(out, 1)

        outs[i: i + 400] = outs[i: i + 400] + np.full(400, predicted[0])
    # print(outs)
    #    print(windowed_st)

    #    print("---")

    fig, axs = plt.subplots(4)
    fig.suptitle("Raw data with P-Pick, Detections added up ")
    # d0 = obsyp_detrend_simple(raw_waveform[0])
    # d1 = obsyp_detrend_simple(raw_waveform[1])
    # d2 = obsyp_detrend_simple(raw_waveform[2])
    # filt = signal.butter(
    #     2, 2, btype="highpass", fs=100, output="sos"
    # )
    # f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
    # f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
    # f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)
    # station_stream = np.stack((f0, f1, f2))
    # station_stream = normalize_stream(station_stream)
    station_stream = np.array(h5data.get(event + "/" + station))

    axs[0].plot(station_stream[0], 'r')
    axs[1].plot(station_stream[1], 'b')
    axs[2].plot(station_stream[2], 'g')
    axs[3].plot(outs)
    axs[0].axvline(3001, color="black")
    axs[1].axvline(3001, color="black")
    axs[2].axvline(3001, color="black")
    axs[3].axvline(3001, color="black")
    fig.savefig("Detection Predict Plot")
    #
    # sequence_length = 4
    # data = pd.read_csv(catalog_path)
    # print(len(data))
    # data = data[data["STATION"] != "LVC"]
    # events = sorted(data["EVENT"].unique())
    # waves = []
    # for file in os.listdir(waveform_path):
    #     if file.endswith(".mseed"):
    #         waves.append(file.strip(".mseed"))
    # intersect = list(set(waves).intersection(set(events)))
    # data = data[data["EVENT"].isin(intersect)]
    #
    # idx = randrange(0, len(data))
    # event, station, p_pick, time = data.iloc[idx][
    #     ["EVENT", "STATION", "P_PICK", "TIME"]
    # ]  # TODO time means start time of event?
    # waveform = obspy.read(os.path.join(waveform_path, f"{event}.mseed"))
    # station_stream = waveform.select(station=station, channel="HH*")  # high frequency
    # while not station_stream:
    #     idx = randrange(0, len(data))
    #     event, station, p_pick, time = data.iloc[idx][
    #         ["EVENT", "STATION", "P_PICK", "TIME"]
    #     ]  # TODO time means start time of event?
    #     waveform = obspy.read(os.path.join(waveform_path, f"{event}.mseed"))
    #     station_stream = waveform.select(station=station, channel="HH*")  # high frequency
    #
    # station_stream.plot()
    # # to prevent loading the entire catalog, choose a random event and check, whether it is in the waveform folder
    #
    # # print now maybe with highlight on the P Pick
    # # maybe slice out with beginning on start time?
    # model = LitNetwork.load_from_checkpoint(
    #     checkpoint_path=checkpoint_path,
    #
    # )
    # model.freeze()
    #
    # out_times = np.arange(
    #     station_stream[0].stats.starttime, station_stream[0].stats.endtime + 1, 1
    # )
    # outs = np.zeros(len(out_times))
    # for i in range(0, len(out_times) - 4):
    #     station_stream = waveform.select(station=station, channel="HH*")  # reload
    #     window = station_stream.slice(starttime=out_times[i], endtime=out_times[i] + 4)
    #     window.filter("highpass", freq=2, zerophase=True)
    #     window.detrend().normalize()
    #
    #     # for windowed_st in station_stream.slide(window_length=sequence_length, step=1):
    #     #     window = windowed_st.copy()
    #     #     window.detrend().normalize()
    #     trace_z = np.float32(window[0])
    #     trace_n = np.float32(window[1])
    #     trace_e = np.float32(window[2])
    #     example = np.stack((trace_z, trace_n, trace_e))[:, 0:-1]
    #     example = torch.from_numpy(example[None])
    #     output = model(example)
    #     _, predicted = torch.max(output, 1)
    #
    #     outs[i: i + 4] = outs[i: i + 4] + np.full(4, predicted[0])
    #     # print(outs)
    # #    print(windowed_st)
    #
    # #    print("---")
    #
    # print(outs)
    # out_times = date2num(out_times)
    #
    # tr1 = station_stream[0]
    # tr2 = station_stream[1]
    # tr3 = station_stream[2]
    #
    # fig, ax = plt.subplots(4, sharex=True)
    # ax[0].plot(tr1.times("matplotlib"), tr1.data, color="black", linewidth=0.4)
    # ax[0].xaxis_date()
    # ax[0].axvline(date2num(UTCDateTime(p_pick)), color="blue")
    #
    # ax[1].plot(tr2.times("matplotlib"), tr2.data, color="black", linewidth=0.4)
    # ax[1].xaxis_date()
    # ax[1].axvline(date2num(UTCDateTime(p_pick)), color="blue")
    #
    # ax[2].plot(tr3.times("matplotlib"), tr3.data, color="black", linewidth=0.4)
    # ax[2].xaxis_date()
    # ax[2].axvline(date2num(UTCDateTime(p_pick)), color="blue")
    #
    # ax[3].plot(out_times, outs, "r-")
    # ax[3].xaxis_date()
    # fig.autofmt_xdate()
    # plt.show()
    # plt.savefig("current_plot")


# learn(cp, hp, mp)
predict(catalog_path=cp, hdf5_path=hp,
        checkpoint_path="../tb_logs/detection/version_8/checkpoints/epoch=22-step=91.ckpt")
# test(catalog_path=cp,hdf5_path=hp,checkpoint_path="../tb_logs/detection/version_2/checkpoints/epoch=22-step=91.ckpt",hparams_file="../tb_logs/detection/version_2/hparams.yaml")
# test_one(catalog_path=cp, hdf5_path=hp,checkpoint_path="../tb_logs/detection/version_8/checkpoints/epoch=22-step=91.ckpt")
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
             checkpoint_path=args.checkpoint_path,
             hparams_file=args.hparams_file,
             hdf5_path=args.hdf5_path
             )
    if action == "test_one":
        test_one(catalog_path=args.catalog_path,
                 checkpoint_path=args.checkpoint_path,
                 hdf5_path=args.hdf5_path
                 )
    if action == 'predict':
        predict(catalog_path=args.catalog_path,
                hdf5_path=args.hdf5_path,
                checkpoint_path=args.checkpoint_path,
                )
