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
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import signal
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from datasets_distance import obspy_detrend, normalize_stream
from litdatamodule_distance import LitDataModule
from litnetwork_distance import LitNetwork

cp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog_sensitivity.csv"
wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset_sensitivity.h5"
mp = "/home/viola/WS2021/Code/Models"
chp = "/home/viola/WS2021/Code/tb_logs/distance/version_47/checkpoints/epoch=19-step=319.ckpt"
hf = ("/home/viola/WS2021/Code/tb_logs/distance/version_47/hparams.yaml",)


# checkpoint_path = "/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_8/checkpoints/epoch=33-step=3093.ckpt",
# hparams_file = "/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_8/hparams.yaml",
# map_location = None,


def learn(catalog_path, hdf5_path, model_path):
    network = LitNetwork()
    dm = LitDataModule(catalog_path=catalog_path, hdf5_path=hdf5_path, batch_size=1024)
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


def timespan_iteration(catalog_path, checkpoint_path, hdf5_path, timespan_array):
    for t in timespan_array:
        predtrue_timespan(catalog_path, checkpoint_path, hdf5_path, t)


def predtrue_timespan(catalog_path, checkpoint_path, hdf5_path, timespan=None):
    # load catalog
    catalog = pd.read_csv(catalog_path)
    test_catalog = catalog[catalog["SPLIT"] == "TEST"]

    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)

    # load model
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.freeze()

    # load scaler
    dist = np.array([1, 600000])
    # print(max(test_catalog["DIST"]))
    assert max(test_catalog["DIST"]) <= 600000
    assert min(test_catalog["DIST"]) >= 1
    scaler = MinMaxScaler()
    scaler.fit(dist.reshape(-1, 1))

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.freeze()
    model.to(device)

    # list for storing mean and variance
    learn = torch.zeros(1, device=device)
    var = torch.zeros(1, device=device)
    learn_s = torch.zeros(1, device=device)
    var_s = torch.zeros(1, device=device)
    true, true_s = [], []
    # preload filters
    filt = signal.butter(2, 2, btype="highpass", fs=100, output="sos")
    lfilt = signal.butter(2, 35, btype="lowpass", fs=100, output="sos")

    # iterate through catalogue
    with torch.no_grad():
        for idx in tqdm(range(0, len(test_catalog))):
            event, station, distance, p, s = test_catalog.iloc[idx][
                ["EVENT", "STATION", "DIST", "P_PICK", "S_PICK"]
            ]

            # load subsequent waveform
            raw_waveform = np.array(h5data.get(event + "/" + station))
            seq_len = 20 * 100  # *sampling rate 20 sec window
            p_pick_array = 3000
            if timespan is None:
                random_point = np.random.randint(seq_len)
            else:
                random_point = int(seq_len - timespan * 100)
            waveform = raw_waveform[
                       :, p_pick_array - random_point: p_pick_array + (seq_len - random_point)
                       ]
            # modify waveform for input
            d0 = obspy_detrend(waveform[0])
            d1 = obspy_detrend(waveform[1])
            d2 = obspy_detrend(waveform[2])

            f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
            f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
            f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)

            g0 = signal.sosfilt(lfilt, f0, axis=-1).astype(np.float32)
            g1 = signal.sosfilt(lfilt, f1, axis=-1).astype(np.float32)
            g2 = signal.sosfilt(lfilt, f2, axis=-1).astype(np.float32)

            waveform = np.stack((g0, g1, g2))
            waveform, _ = normalize_stream(waveform)

            # evaluate stream
            station_stream = torch.from_numpy(waveform[None])
            station_stream = station_stream.to(device)
            outputs = model(station_stream)
            learned = outputs[0][0]
            variance = outputs[1][0]

            # check if the s pick already arrived
            if s and (s - p) * 100 < (seq_len - random_point):
                # print("S Pick included, diff: ", (s - p), (seq_len - random_point) / 100)
                learn_s = torch.cat((learn_s, learned), 0)
                # var_s = torch.cat((var_s, variance), 0)
                true_s = true_s + [distance]

            else:
                learn = torch.cat((learn, learned), 0)
                # var = torch.cat((var, variance), 0)
                true = true + [distance]

        learn = learn.cpu()
        # var = var.cpu()

        learn = np.delete(learn, 0)
        # var = np.delete(var, 0)
        # sig = np.sqrt(var)
        pred = scaler.inverse_transform(learn.reshape(-1, 1)).squeeze()

        learn_s = learn_s.cpu()
        # var_s = var_s.cpu()

        # if learn_s.shape != torch.Size([1]):  # no element was added during loop
        learn_s = np.delete(learn_s, 0)
        pred_s = scaler.inverse_transform(learn_s.reshape(-1, 1)).squeeze()
    # if learn_s.shape == 0:
    #    rsmes = np.round(mean_squared_error(np.array(true_s) / 1000, (pred_s) / 1000, squared=False),
    #                     decimals=2)
    # else:
    #    pred_s = np.zeros((0))
    #    rsmes = -1

    # Plot with differentiation between S and no S Arrivals
    fig, axs = plt.subplots(1)
    axs.tick_params(axis="both", labelsize=8)
    fig.suptitle(
        "Predicted and true distance values, \ndifferentiating between recordings with and without a S-Wave arrival",
        fontsize=10,
    )

    x = np.array(true) / 1000
    y = pred / 1000
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    cm = plt.cm.get_cmap("autumn")
    x, y, z = x[idx], y[idx], z[idx]
    axs.scatter(
        x,
        y,
        c=z,
        cmap=cm,
        marker="D",
        s=0.3,
        alpha=0.3,
        lw=0,
        label="Recordings without a S-Wave arrival",
    )

    x = np.array(true_s) / 1000
    y = pred_s / 1000
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    cm = plt.cm.get_cmap("winter")

    axs.scatter(
        x,
        y,
        c=z,
        cmap=cm,
        marker="s",
        s=0.3,
        alpha=0.3,
        lw=0,
        label="Recordings in which there is a S-Wave arrival",
    )
    if timespan is not None:
        axs.legend(
            title=str(timespan) + " seconds after P-Wave arrival",
            loc="best",
            fontsize=7,
            title_fontsize=8,
        )
    else:
        axs.legend(loc=0)
    axs.axline((0, 0), (100, 100), linewidth=0.5, color="black")
    plt.axis("square")
    plt.xlabel("True distance[km]", fontsize=8)
    plt.ylabel("Predicted distance[km]", fontsize=8)
    if timespan is not None:
        fig.savefig(
            "Distance:PredVSTrue_" + str(timespan).replace(".", "_") + "sec", dpi=600
        )
    else:
        fig.savefig("Distance:PredVSTrue", dpi=600)
        # Plot with differentiation between S and no S Arrivals
    fig, axs = plt.subplots(1)
    axs.tick_params(axis="both", labelsize=8)
    fig.suptitle(
        "Predicted and true distance values, \ndifferentiating between recordings with and without a S-Wave arrival",
        fontsize=10,
    )

    x = np.array(true) / 1000
    y = pred / 1000
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    cm = plt.cm.get_cmap("cividis")
    x, y, z = x[idx], y[idx], z[idx]
    axs.scatter(
        x,
        y,
        c=z,
        cmap=cm,
        s=0.5,
        marker="s",
        lw=0,
        alpha=0.2,
        label="Recordings without a S-Wave arrival",
    )

    x = np.array(true_s) / 1000
    y = pred_s / 1000
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    cm = plt.cm.get_cmap("spring")

    axs.scatter(
        x,
        y,
        s=0.5,
        c=z,
        cmap=cm,
        marker="D",
        lw=0,
        alpha=0.2,
        label="Recordings in which there is a S-Wave arrival",
    )
    if timespan is not None:
        axs.legend(
            title=str(timespan) + " seconds after P-Wave arrival",
            loc="best",
            fontsize=7,
            title_fontsize=8,
        )
    else:
        axs.legend(loc=0)
    axs.axline((0, 0), (100, 100), linewidth=0.5, color="black")
    plt.axis("square")
    plt.xlabel("True distance[km]", fontsize=8)
    plt.ylabel("Predicted distance[km]", fontsize=8)
    if timespan is not None:
        fig.savefig(
            "Distance:PredVSTrue2_" + str(timespan).replace(".", "_") + "sec", dpi=600
        )
    else:
        fig.savefig("Distance:PredVSTrue2", dpi=600)
        # Plot with differentiation between S and no S Arrivals
    fig, axs = plt.subplots(1)
    axs.tick_params(axis="both", labelsize=8)
    fig.suptitle(
        "Predicted and true distance values, \ndifferentiating between recordings with and without a S-Wave arrival",
        fontsize=10,
    )

    x = np.array(true) / 1000
    y = pred / 1000
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    cm = plt.cm.get_cmap("autumn")
    x, y, z = x[idx], y[idx], z[idx]
    axs.scatter(
        x,
        y,
        c=z,
        cmap=cm,
        s=3,
        marker="s",
        lw=0,
        alpha=0.05,
        label="Recordings without a S-Wave arrival",
    )

    x = np.array(true_s) / 1000
    y = pred_s / 1000
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    cm = plt.cm.get_cmap("winter")

    axs.scatter(
        x,
        y,
        s=3,
        c=z,
        cmap=cm,
        marker="D",
        lw=0,
        alpha=0.05,
        label="Recordings in which there is a S-Wave arrival",
    )
    if timespan is not None:
        axs.legend(
            title=str(timespan) + " seconds after P-Wave arrival",
            loc="best",
            fontsize=7,
            title_fontsize=8,
        )
    else:
        axs.legend(loc=0)
    axs.axline((0, 0), (100, 100), linewidth=0.5, color="black")
    plt.axis("square")
    plt.xlabel("True distance[km]", fontsize=8)
    plt.ylabel("Predicted distance[km]", fontsize=8)
    if timespan is not None:
        fig.savefig(
            "Distance:PredVSTrue3_" + str(timespan).replace(".", "_") + "sec", dpi=600
        )
    else:
        fig.savefig("Distance:PredVSTrue3", dpi=600)
    # Plot without differentiation
    fig, axs = plt.subplots(1)
    axs.tick_params(axis="both", labelsize=8)
    fig.suptitle("Predicted and true distance values")
    # " \nRSME = " + str(
    #    rsme), fontsize=10)

    x = np.array(true + true_s) / 1000
    y = np.append(pred, pred_s) / 1000
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    cm = plt.cm.get_cmap("plasma")
    x, y, z = x[idx], y[idx], z[idx]
    axs.scatter(
        x, y, c=z, cmap=cm, s=2, marker="o", alpha=0.05, label="Test set recordings"
    )

    if timespan is not None:
        axs.legend(
            title=str(timespan) + " seconds after P-Wave arrival",
            loc="best",
            fontsize=8,
            title_fontsize=8,
        )
    else:
        axs.legend(loc=0)
    axs.axline((0, 0), (100, 100), linewidth=0.5, color="black")
    plt.axis("square")
    plt.xlabel("True distance[km]", fontsize=8)
    plt.ylabel("Predicted distance[km]", fontsize=8)
    if timespan is not None:
        fig.savefig(
            "Distance:PredVSTrue_simple_" + str(timespan).replace(".", "_") + "sec",
            dpi=600,
        )
    else:
        fig.savefig("Distance:PredVSTrue_simple", dpi=600)


def rsme_timespan(catalog_path, checkpoint_path, hdf5_path):
    # load catalog
    catalog = pd.read_csv(catalog_path)
    test_catalog = catalog[catalog["SPLIT"] == "TEST"]
    s_times = test_catalog["S_PICK"]
    p_times = test_catalog["P_PICK"]
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)

    # iterate through catalogue
    timespan = np.linspace(0, 20, num=41)

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.freeze()
    model.to(device)

    # load scaler
    distmax = 600000
    distmin = 1
    # print(max(test_catalog["DIST"]))
    assert max(test_catalog["DIST"]) <= distmax
    assert min(test_catalog["DIST"]) >= distmin
    # scaler = MinMaxScaler()
    # scaler.fit(np.array([distmin,distmax]).reshape(-1, 1))

    # list for storing rsme
    rsme = torch.empty(len(timespan), device=device)
    rsme_s = torch.empty(len(timespan), device=device)
    rsme_p = torch.empty(len(timespan), device=device)
    # variance_all = torch.empty(len(timespan), device=device)

    # timespan = 1

    # preload filters
    filt = signal.butter(2, 2, btype="highpass", fs=100, output="sos")
    lfilt = signal.butter(2, 35, btype="lowpass", fs=100, output="sos")

    with torch.no_grad():
        for t in tqdm(timespan):
            learn = torch.zeros(1, device=device)
            # var = torch.empty(1, device=device)
            learn_s = torch.zeros(1, device=device)
            # var_s = torch.empty(1, device=device)
            true_s = torch.zeros(1, device=device)
            true = torch.zeros(1, device=device)
            dist = np.array(test_catalog["DIST"])
            dist = [[d] for d in dist]
            dist = torch.tensor(dist, device=device)
            for idx in range(0, len(test_catalog)):
                event, station, distance, p, s = test_catalog.iloc[idx][
                    ["EVENT", "STATION", "DIST", "P_PICK", "S_PICK"]
                ]
                # load subsequent waveform
                raw_waveform = np.array(h5data.get(event + "/" + station))
                seq_len = 20 * 100  # *sampling rate 20 sec window
                p_pick_array = 3000
                random_point = int(seq_len - t * 100)
                waveform = raw_waveform[
                           :,
                           p_pick_array
                           - random_point: p_pick_array
                                           + (seq_len - random_point),
                           ]

                # modify waveform for input
                d0 = obspy_detrend(waveform[0])
                d1 = obspy_detrend(waveform[1])
                d2 = obspy_detrend(waveform[2])

                f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
                f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
                f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)

                g0 = signal.sosfilt(lfilt, f0, axis=-1).astype(np.float32)
                g1 = signal.sosfilt(lfilt, f1, axis=-1).astype(np.float32)
                g2 = signal.sosfilt(lfilt, f2, axis=-1).astype(np.float32)

                waveform = np.stack((g0, g1, g2))
                waveform, _ = normalize_stream(waveform)

                # evaluate stream
                station_stream = torch.from_numpy(waveform[None])
                station_stream = station_stream.to(device)
                outputs = model(station_stream)
                learned = outputs[0][0]
                variance = outputs[1][0]

                # check if the s pick already arrived
                if s and (s - p) * 100 < (seq_len - random_point):
                    # print("S Pick included, diff: ", (s - p), (seq_len - random_point) / 100)
                    learn_s = torch.cat((learn_s, learned))
                    # var_s = torch.cat((var_s, variance))
                    true_s = torch.cat((true_s, dist[idx]))
                else:
                    learn = torch.cat((learn, learned))
                    # var = torch.cat((var, variance))
                    true = torch.cat((true, dist[idx]))

            # delete the dummy variable
            learn_s = learn_s[learn_s != learn_s[0]]
            # var_s = var_s[var_s != var_s[0]]
            learn = learn[learn != learn[0]]
            # var = var[var != var[0]]
            true = true[true != true[0]]
            true_s = true_s[true_s != true_s[0]]

            # scale back to km
            learn_scaled = torch.add(torch.mul(learn, distmax), distmin)
            learn_s_scaled = torch.add(torch.mul(learn_s, distmax), distmin)
            # variance_scaled = torch.add(torch.mul(var, 600000), 1)
            # variance_s_scaled = torch.add(torch.mul(var_s, 600000), 1)

            i = np.where(timespan == t)[0][0]
            rsme_p[i] = torch.sqrt(F.mse_loss(learn_scaled, true))
            rsme_s[i] = torch.sqrt(F.mse_loss(learn_s_scaled, true_s))
            rsme[i] = torch.sqrt(
                F.mse_loss(
                    torch.cat((learn_scaled, learn_s_scaled)), torch.cat((true, true_s))
                )
            )
            # variance_all[i]= torch.cat((variance_all,torch.cat((variance_scaled,variance_s_scaled))))

    rsme = rsme.cpu()
    rsme_p = rsme_p.cpu()
    rsme_s = rsme_s.cpu()
    fig, axs = plt.subplots(1)
    axs.tick_params(axis="both", labelsize=8)
    fig.suptitle("RSME after the P-arrival depending on S-arrivals", fontsize=10)
    axs.plot(
        timespan,
        np.array(rsme_p) / 1000,
        linewidth=0.7,
        label="Recordings without a S-Wave arrival",
        color="steelblue",
    )
    axs.plot(
        timespan,
        np.array(rsme_s) / 1000,
        linewidth=0.7,
        label="Recordings in which there is a S-Wave arrival",
        color="crimson",
    )
    axs.plot(
        timespan,
        np.array(rsme) / 1000,
        linewidth=0.7,
        label="All recordings",
        color="rebeccapurple",
    )
    axs.legend(fontsize=8, loc="best")
    plt.xlabel("Time after P-Wave arrival[sec]", fontsize=8)
    plt.ylabel("RSME[km]", fontsize=8)
    fig.savefig("Distance:RSME", dpi=600)


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
    learned = outputs[0][0]
    var = outputs[1][0]
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
    print(t, learned)
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
    # h5data.close()


def test(catalog_path, hdf5_path, checkpoint_path, hparams_file):
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        # hparams_file=hparams_file,
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
        # print("outputs", outputs)
        learned = outputs[0]
        var = outputs[1]
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

    fig, axs = plt.subplots(2, sharex=True)
    axs[1].set_xlabel("Time in milliseconds", fontdict={"fontsize": 8})
    fig.suptitle(
        "Distance prediction plot, the real distance ist about "
        + str(np.int(distance / 1000))
        + "km"
    )
    axs[0].plot(waveform[0], "r", linewidth=0.5)
    axs[0].plot(waveform[1], "b", linewidth=0.5)
    axs[0].plot(waveform[2], "g", linewidth=0.5)
    axs[0].axvline(3000, color="black", linestyle="dashed", linewidth=0.5)
    if s and (s - p) < 30:
        axs[0].axvline(
            3000 + (s - p) * 100, color="black", linestyle="dashed", linewidth=0.5
        )
    axs[0].set_title(
        "Normalized and filtered input for Z(red), N(blue) and E(green) with P-pick (and S-Pick, if there)",
        fontdict={"fontsize": 8},
    )

    # axs[1,0].axvline(random_point, color="black")
    # axs[2,0].axvline(random_point, color="black")
    # axs[1].set_title("Computed distance versus real distance", fontdict={"fontsize": 8})
    # scale_y = 1000  # metres to km
    # ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_y))
    # axs[1].yaxis.set_major_formatter(ticks_y)
    # axs[1].axhline(distance, color="black", linestyle="dashed")
    # axs[1].plot(t, real_output, color="green", alpha=0.7, linewidth = 0.5)
    # axs[1].set_ylabel("Distance in km",fontdict={"fontsize": 8})

    scale_y = 1000  # metres to km
    ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_y))
    axs[1].set_ylabel("Distance in km", fontdict={"fontsize": 8})
    axs[1].set_title("Distance and standard deviation", fontdict={"fontsize": 8})
    axs[1].yaxis.set_major_formatter(ticks_y)
    axs[1].axhline(distance, color="black", linestyle="dashed")
    axs[1].plot(t, real_output, color="green", alpha=0.7, linewidth=0.7)
    # axs[4].plot(t,real_output+real_sig, color = "green", alpha = 0.3)
    # axs[4].plot(t,real_output-real_sig, color = "green", alpha = 0.3)
    axs[1].fill_between(
        t, real_output, real_output + real_sig, alpha=0.3, color="green"
    )
    axs[1].fill_between(
        t, real_output, real_output - real_sig, alpha=0.3, color="green"
    )

    fig.tight_layout()
    fig.savefig("Prediction Plot", dpi=600)
    # plt.plot(t,mean_squared_error(s_output,s_labels),":")
    # h5data.close()


# learn(catalog_path=cp, hdf5_path=hp, model_path=mp)
# predict(cp, hp, chp)

# rsme_timespan(cp, chp, hp)
# predtrue_s_waves(cp, chp, hp)
# predtrue_timespan(cp, chp, hp)
# test(catalog_path=cp,hdf5_path=hp, checkpoint_path=chp, hparams_file=hf)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--catalog_path", type=str)
    parser.add_argument("--hdf5_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--hparams_file", type=str)
    parser.add_argument("--time", type=str)
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
    if action == "predtrue":
        predtrue_timespan(
            catalog_path=args.catalog_path,
            hdf5_path=args.hdf5_path,
            checkpoint_path=args.checkpoint_path,
        )
    if action == "rsme":
        rsme_timespan(
            catalog_path=args.catalog_path,
            hdf5_path=args.hdf5_path,
            checkpoint_path=args.checkpoint_path,
        )
    if action == "timespan":
        timespan_iteration(
            catalog_path=args.catalog_path,
            hdf5_path=args.hdf5_path,
            checkpoint_path=args.checkpoint_path,
            timespan_array=args.time,
        )
