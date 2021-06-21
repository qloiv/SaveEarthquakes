from __future__ import print_function, division

import argparse
from random import randrange

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from datasets_distance import obspy_detrend, normalize_stream
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
    dm = LitDataModule(catalog_path=catalog_path, hdf5_path=hdf5_path, batch_size=128)
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


def test_one(catalog_path, checkpoint_path, hdf5_path):
    print("hey")
    # load catalog with random test event
    catalog = pd.read_csv(catalog_path)
    test_catalog = catalog[catalog["SPLIT"] == "TEST"]
    idx = randrange(0, len(test_catalog))
    print(idx)
    event, station, distance, p, s = test_catalog.iloc[idx][
        ["EVENT", "STATION", "DIST", "P_PICK", "S_PICK"]
    ]

    dist = np.array([1, 600000])
    print(max(test_catalog["DIST"]))
    assert max(test_catalog["DIST"]) <= 600000
    assert min(test_catalog["DIST"]) >= 1
    scaler = MinMaxScaler()
    scaler.fit(dist.reshape(-1, 1))
    ts_dist = scaler.transform(distance.reshape(1, -1))
    label = np.float32(ts_dist.squeeze())
    print(label)
    # load network
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.freeze()

    raw_waveform = np.array(h5data.get(event + "/" + station))
    seq_len = 20 * 100  # *sampling rate 20 sec window
    p_pick_array = 3000
    random_point = np.random.randint(seq_len)
    waveform = raw_waveform[
               :, p_pick_array - random_point: p_pick_array + (seq_len - random_point)
               ]
    print(np.shape(waveform))

    spick = False
    if s and (s - p) * 100 < (seq_len - random_point):
        print("S Pick included, diff: ", (s - p), (seq_len - random_point) / 100)
        spick = True

    fig, axs = plt.subplots(3)
    fig.suptitle("Input of Distance Network - full Trace")
    axs[0].plot(raw_waveform[0], "r")
    axs[1].plot(raw_waveform[1], "b")
    axs[2].plot(raw_waveform[2], "g")
    fig.savefig("TestOne:Full Trace")

    fig, axs = plt.subplots(3)
    fig.suptitle("Input to Network")
    axs[0].plot(waveform[0], "r")
    axs[1].plot(waveform[1], "b")
    axs[2].plot(waveform[2], "g")
    fig.savefig("TestOne: Input")
    d0 = obspy_detrend(waveform[0])
    d1 = obspy_detrend(waveform[1])
    d2 = obspy_detrend(waveform[2])
    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending")
    axs[0].plot(d0, "r")
    axs[1].plot(d1, "b")
    axs[2].plot(d2, "g")
    fig.savefig("TestOne:Detrended")

    filt = signal.butter(2, 2, btype="highpass", fs=100, output="sos")
    f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
    f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
    f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)

    # set low pass filter
    lfilt = signal.butter(2, 35, btype="lowpass", fs=100, output="sos")
    g0 = signal.sosfilt(lfilt, f0, axis=-1).astype(np.float32)
    g1 = signal.sosfilt(lfilt, f1, axis=-1).astype(np.float32)
    g2 = signal.sosfilt(lfilt, f2, axis=-1).astype(np.float32)

    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending, then Filtering")
    axs[0].plot(f0, "r")
    axs[1].plot(f1, "b")
    axs[2].plot(f2, "g")
    fig.savefig("TestOne:Detrended and Filtered")
    waveform = np.stack((g0, g1, g2))
    waveform, _ = normalize_stream(waveform)
    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending->Filtering->Normalizing")
    axs[0].plot(waveform[0], "r")
    axs[1].plot(waveform[1], "b")
    axs[2].plot(waveform[2], "g")
    fig.savefig("TestOne: Detrended, Filtered and Normalized")

    station_stream = torch.from_numpy(waveform[None])
    outputs = model(station_stream)
    learned = outputs[0]
    var = outputs[1]
    print(learned, var)

    sigma = np.sqrt(var)

    r_learned = scaler.inverse_transform(learned.reshape(1, -1))[0]
    # r_var = (scaler.inverse_transform(var.reshape(1, -1))[0])
    r_sigma = scaler.inverse_transform(sigma.reshape(1, -1))[0]
    print(r_learned, r_sigma)

    fig, axs = plt.subplots(5, sharex=True)
    # axs[4].yaxis.set_major_formatter(FormatStrFormatter('%d'))
    fig.suptitle(
        "Target:~"
        + str(int(distance / 1000))
        + "km, learned value:~"
        + str(int(r_learned / 1000))
        + "km\n 68% conf to be between "
        + str(int((r_learned + r_sigma) / 1000))
        + "km and "
        + str(int((r_learned - r_sigma) / 1000))
        + "km, S-Pick included:"
        + str(spick)
    )

    axs[0].plot(waveform[0], "r")

    axs[1].plot(waveform[1], "b")

    axs[2].plot(waveform[2], "g")

    axs[3].axhline(label, color="black", linestyle="dashed")
    axs[3].axhline(learned, color="green", alpha=0.7)
    axs[3].axhline(learned + 3 * sigma, color="green", alpha=0.001, linewidth=0.001)
    axs[3].axhline(learned - 3 * sigma, color="green", alpha=0.001, linewidth=0.001)

    t = np.linspace(1, 2000, num=2000)
    var1 = learned + var
    var2 = learned - var
    axs[3].fill_between(t, learned, learned + 3 * sigma, alpha=0.01, color="green")
    axs[3].fill_between(t, learned, learned - 3 * sigma, alpha=0.01, color="green")
    axs[3].fill_between(t, learned, learned + 2 * sigma, alpha=0.2, color="green")
    axs[3].fill_between(t, learned, learned - 2 * sigma, alpha=0.2, color="green")
    axs[3].fill_between(t, learned, learned + sigma, alpha=0.4, color="green")
    axs[3].fill_between(t, learned, learned - sigma, alpha=0.4, color="green")

    scale_y = 1000  # metres to km
    ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_y))
    axs[4].yaxis.set_major_formatter(ticks_y)
    axs[4].axhline(distance, color="black", linestyle="dashed")
    axs[4].axhline(r_learned, color="green", alpha=0.7)
    print(r_learned, r_learned + 3 * r_sigma)
    axs[4].fill_between(
        t, r_learned, r_learned + 3 * r_sigma, alpha=0.01, color="green"
    )
    axs[4].fill_between(
        t, r_learned, r_learned - 3 * r_sigma, alpha=0.01, color="green"
    )
    axs[4].fill_between(t, r_learned, r_learned + 2 * r_sigma, alpha=0.2, color="green")
    axs[4].fill_between(t, r_learned, r_learned - 2 * r_sigma, alpha=0.2, color="green")
    axs[4].fill_between(t, r_learned, r_learned + r_sigma, alpha=0.4, color="green")
    axs[4].fill_between(t, r_learned, r_learned - r_sigma, alpha=0.4, color="green")
    axs[4].axhline(r_learned + 3 * r_sigma, color="green", alpha=0.001, linewidth=0.001)
    axs[4].axhline(r_learned - 3 * r_sigma, color="green", alpha=0.001, linewidth=0.001)

    axs[0].axvline(random_point, color="black", linestyle="dashed", linewidth=0.5)
    axs[1].axvline(random_point, color="black", linestyle="dashed", linewidth=0.5)
    axs[2].axvline(random_point, color="black", linestyle="dashed", linewidth=0.5)
    if spick:
        axs[0].axvline(
            random_point + (s - p) * 100, color="red", linestyle="dashed", linewidth=0.5
        )
        axs[1].axvline(
            random_point + (s - p) * 100, color="red", linestyle="dashed", linewidth=0.5
        )
        axs[2].axvline(
            random_point + (s - p) * 100, color="red", linestyle="dashed", linewidth=0.5
        )

    fig.savefig("TestOne: Results")
    h5data.close()


def test(catalog_path, hdf5_path, checkpoint_path, hparams_file):
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams_file=hparams_file,
        map_location=None,
    )
    dm = LitDataModule(catalog_path, hdf5_path, batch_size=128)
    # init trainer with whatever options
    trainer = pl.Trainer(gpus=[0])

    # test (pass in the model)
    trainer.test(model, datamodule=dm)


def predict(
        catalog_path, hdf5_path, checkpoint_path
):  # TODO put sequence length into variable
    # load catalog with random test event
    catalog = pd.read_csv(catalog_path)
    test_catalog = catalog[catalog["SPLIT"] == "TEST"]
    idx = randrange(0, len(test_catalog))
    print(idx)
    event, station, distance, p, s = test_catalog.iloc[idx][
        ["EVENT", "STATION", "DIST", "P_PICK", "S_PICK"]
    ]

    dist = np.array([1, 600000])
    print(max(test_catalog["DIST"]))
    assert max(test_catalog["DIST"]) <= 600000
    assert min(test_catalog["DIST"]) >= 1
    scaler = MinMaxScaler()
    scaler.fit(dist.reshape(-1, 1))
    ts_dist = scaler.transform(distance.reshape(1, -1))
    label = np.float32(ts_dist.squeeze())
    print(label)
    # load network
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.freeze()

    waveform = np.array(h5data.get(event + "/" + station))
    filt = signal.butter(2, 2, btype="highpass", fs=100, output="sos")
    # set low pass filter
    lfilt = signal.butter(2, 35, btype="lowpass", fs=100, output="sos")
    real_output = np.zeros(6000)
    # real_labels = np.zeros(6000)
    s_output = np.zeros(6000)
    # s_labels = np.zeros(6000)
    real_sig = np.zeros(6000)
    s_sig = np.zeros(6000)

    for i in tqdm(range(0, 6000 - 20 * 100)):
        station_stream = waveform[:, i: i + 20 * 100]
        d0 = obspy_detrend(station_stream[0])
        d1 = obspy_detrend(station_stream[1])
        d2 = obspy_detrend(station_stream[2])

        f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
        f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
        f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)

        g0 = signal.sosfilt(lfilt, f0, axis=-1).astype(np.float32)
        g1 = signal.sosfilt(lfilt, f1, axis=-1).astype(np.float32)
        g2 = signal.sosfilt(lfilt, f2, axis=-1).astype(np.float32)

        station_stream = np.stack((g0, g1, g2))
        station_stream, _ = normalize_stream(station_stream)
        station_stream = torch.from_numpy(station_stream[None])

        outputs = model(station_stream)
        learned = outputs[:, 0]
        var = outputs.data[:, 1]
        sigma = np.sqrt(var)
        s_sig[i + 2000] = sigma
        real_sig[i + 2000] = scaler.inverse_transform(sigma.reshape(1, -1))[0]
        s_output[i + 2000] = learned
        # s_labels[i+2000] = ts_dist
        real_output[i + 2000] = scaler.inverse_transform(learned.reshape(1, -1))[0]
        # real_labels[i+2000] = distance

    print(real_output)
    # print(real_labels)
    print(s_output)
    # print(s_labels)
    # print(mean_squared_error(real_output, real_labels))
    # print(mean_squared_error(s_output, s_labels))
    t = np.linspace(0, 6000, num=6000)
    fig, axs = plt.subplots(5, sharex=True)
    fig.suptitle(
        "Predict Plot, Distance:~"
        + str(int(distance / 1000))
        + "km \nLearned distance ranges from ~"
        + str(int(min(real_output[2000:]) / 1000))
        + "km to ~"
        + str(int(max(real_output[2000:]) / 1000))
        + "km"
    )
    real_output[0:2000] = np.nan

    waveform = np.array(h5data.get(event + "/" + station))

    d0 = obspy_detrend(waveform[0])
    d1 = obspy_detrend(waveform[1])
    d2 = obspy_detrend(waveform[2])

    filt = signal.butter(2, 2, btype="highpass", fs=100, output="sos")
    f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
    f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
    f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)

    # set low pass filter
    lfilt = signal.butter(2, 35, btype="lowpass", fs=100, output="sos")
    g0 = signal.sosfilt(lfilt, f0, axis=-1).astype(np.float32)
    g1 = signal.sosfilt(lfilt, f1, axis=-1).astype(np.float32)
    g2 = signal.sosfilt(lfilt, f2, axis=-1).astype(np.float32)

    waveform = np.stack((g0, g1, g2))
    waveform, _ = normalize_stream(waveform)

    axs[0].plot(waveform[0], "r")
    axs[1].plot(waveform[1], "b")
    axs[2].plot(waveform[2], "g")
    axs[0].axvline(3000, color="black", linestyle="dashed", linewidth=0.5)
    axs[1].axvline(3000, color="black", linestyle="dashed", linewidth=0.5)
    axs[2].axvline(3000, color="black", linestyle="dashed", linewidth=0.5)
    if s and (s - p) < 30:
        axs[0].axvline(
            3000 + (s - p) * 100, color="black", linestyle="dashed", linewidth=0.5
        )
        axs[1].axvline(
            3000 + (s - p) * 100, color="black", linestyle="dashed", linewidth=0.5
        )
        axs[2].axvline(
            3000 + (s - p) * 100, color="black", linestyle="dashed", linewidth=0.5
        )
    real_labels = np.full(6000, distance)
    print(np.shape(real_labels), np.shape(real_output))
    # axs[3].title.set_text("Absolute Error in m")
    # scale_y = 1000 #metres to km
    # ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
    # axs[3].yaxis.set_major_formatter(ticks_y)
    # axs[3].plot(t,np.abs(real_output-real_labels))

    # axs[4].title.set_text("Target, learned value and 68% conf interval")
    scale_y = 1000  # metres to km
    ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_y))
    axs[4].yaxis.set_major_formatter(ticks_y)
    axs[4].axhline(distance, color="black", linestyle="dashed")
    axs[4].plot(t, real_output, color="green", alpha=0.7)
    # axs[4].plot(t,real_output+real_sig, color = "green", alpha = 0.3)
    # axs[4].plot(t,real_output-real_sig, color = "green", alpha = 0.3)
    axs[4].fill_between(
        t, real_output, real_output + real_sig, alpha=0.3, color="green"
    )
    axs[4].fill_between(
        t, real_output, real_output - real_sig, alpha=0.3, color="green"
    )

    scale_y = 1000  # metres to km
    ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_y))
    axs[3].yaxis.set_major_formatter(ticks_y)
    axs[3].axhline(distance, color="black", linestyle="dashed")
    axs[3].plot(t, real_output, color="green", alpha=0.7)

    # plt.plot(t,mean_squared_error(s_output,s_labels),":")
    fig.savefig("predict plot")
    h5data.close()


learn(catalog_path=cp, hdf5_path=hp, model_path=mp)
# predict(cp, hp, chp)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--catalog_path", type=str)
    parser.add_argument("--hdf5_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--hparams_file", type=str)
    args = parser.parse_args()
    action = args.action
    if action == "test_one":
        test_one(
            catalog_path=args.catalog_path,
            checkpoint_path=args.checkpoint_path,
            hdf5_path=args.hdf5_path,
        )
    if action == "learn":
        learn(
            catalog_path=args.catalog_path,
            hdf5_path=args.hdf5_path,
            model_path=args.model_path,
        )
    if action == "test":
        test(
            catalog_path=args.catalog_path,
            hdf5_path=args.hdf5_path,
            checkpoint_path=args.checkpoint_path,
            hparams_file=args.hparams_file,
        )
    if action == "predict":
        predict(
            catalog_path=args.catalog_path,
            hdf5_path=args.hdf5_path,
            checkpoint_path=args.checkpoint_path,
        )
