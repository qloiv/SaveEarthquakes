from __future__ import print_function, division

import argparse
import datetime
from random import randrange

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import signal
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from datasets_magnitude import normalize_stream, obspy_detrend
from litdatamodule_magnitude import LitDataModule
from litnetwork_magnitude import LitNetwork

cp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog_sensitivity.csv"
wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
wpa = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset_sensitivity.h5"
mp = "/home/viola/WS2021/Code/Models"
chp = "/home/viola/WS2021/Code/tb_logs/distance/version_47/checkpoints/epoch=19-step=319.ckpt"
hf = ("/home/viola/WS2021/Code/tb_logs/distance/version_47/hparams.yaml",)
ip = "/home/viola/WS2021/Code/Daten/Chile_small/inventory.xml"

#cp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog.csv"
#wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
#hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset.h5"
#mp = "/home/viola/WS2021/Code/Models"
#chp = "/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/magnitude/version_76/checkpoints/epoch=5-step=95.ckpt"
cp = "../../new_catalogue_sensitivity.csv"
wp = "../../../data/earthquake/waveforms_long_full/"
wpa ="../../../data/earthquake/waveforms_long_additional/"
hp = "../../new_h5data_sensitivity.h5"
chp = "../tb_logs/magnitude/version_26/checkpoints/epoch=353-step=412055.ckpt"
ip = "../../inventory.xml"


# checkpoint_path = "/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_8/checkpoints/epoch=33-step=3093.ckpt",
# hparams_file = "/home/viola/WS2021/Code/SaveEarthquakes/tb_logs/my_model/version_8/hparams.yaml",
# map_location = None,


class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics["epoch"] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)


def learn(catalog_path, hdf5_path, model_path):
    network = LitNetwork()
    dm = LitDataModule(catalog_path=catalog_path, hdf5_path=hdf5_path, batch_size=128)
    logger = TensorBoardLogger("../tb_logs", name="magnitude")
    checkpoint_callback = ModelCheckpoint()
    # ch2 = ModelCheckpointAtEpochEnd()
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
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
    dm = LitDataModule(catalog_path, hdf5_path, batch_size=128)
    # init trainer with whatever options
    trainer = pl.Trainer(gpus=[0])

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

    test_catalog = catalog[catalog["SPLIT"] == "TEST"]
    above = True
    if above is True:
        test_catalog=test_catalog[test_catalog["MA"]>=5]
    idx = randrange(0, len(test_catalog))
    event, station, ma, s, p = test_catalog.iloc[idx][["EVENT", "STATION", "MA", "S_PICK", "P_PICK"]]
    print("MA", ma)
    waveform = np.array(h5data.get(event + "/" + station))
    filt = signal.butter(2, 2, btype="highpass", fs=100, output="sos")
    # set low pass filter
    lfilt = signal.butter(2, 35, btype="lowpass", fs=100, output="sos")

    output = np.zeros(6000)
    labels = np.zeros(6000)
    labels.fill(ma)
    output[0:2000] = np.nan
    for i in tqdm(range(2000, 6000)):
        station_stream = waveform[:, i - 2000: i]
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
        station_stream, max_stream = normalize_stream(station_stream)
        station_stream = torch.from_numpy(station_stream[None])
        ms = np.float32(0.001 * np.log(max_stream))
        ms = torch.tensor(ms).unsqueeze(-1)
        model_input = (station_stream, ms)
        # print(type(model_input))
        # print(type(model_input[0]))
        # print(type(model_input[1]), model_input[1].shape)
        # print((model_input[0].shape))
        # print((model_input[1].shape))

        predicted = model(model_input).squeeze()
        # _, predicted = torch.max(out.data, 1)
        output[i] = predicted

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
    fig.suptitle(
        "Magnitude prediction for one example.", fontsize=10
    )
    spick = False
    if s and (s - p) < 30:
        spick = True

    axs[1].tick_params(axis="both", labelsize=8)
    axs[0].tick_params(axis="both", labelsize=8)

    axs[1].set_ylabel("Magnitude", fontsize=8)
    axs[0].plot(waveform[1], "tab:orange", linewidth=0.5, alpha=0.8)
    axs[0].plot(waveform[2], "tab:green", linewidth=0.5, alpha=0.8)
    axs[0].plot(waveform[0], "tab:blue", linewidth=0.5, alpha=0.8)
    axs[1].set_xlabel("Time[sec]", fontdict={"fontsize": 8})

    axs[0].axvline(3000, color="black", linestyle="dotted", linewidth=0.5)
    axs[1].axvline(3000, color="black", linestyle="dotted", linewidth=0.5)

    ymin, ymax = axs[0].get_ylim()
    axs[0].annotate("P-Pick", xy=(3000, ymin), xytext=(-4, 2), textcoords='offset points',
                    annotation_clip=False, fontsize=6, rotation=90, va='bottom', ha='center')
    # ymin, ymax = axs[1].get_ylim()
    # axs[1].annotate("P-Pick", xy=(3000, ymin), xytext=(-4, 2), textcoords='offset points',
    #                 annotation_clip=False, fontsize=6, rotation=90, va='bottom', ha='center')

    if spick:
        ymin, ymax = axs[0].get_ylim()
        axs[0].annotate("S-Pick", xy=(3000 + (s - p) * 100, ymin), xytext=(-4, 2), textcoords='offset points',
                        annotation_clip=False, fontsize=6, rotation=90, va='bottom', ha='center')
        # ymin, ymax = axs[1].get_ylim()
        # axs[1].annotate("S-Pick", xy=(3000 + (s - p) * 100, ymin), xytext=(-4, 2), textcoords='offset points',
        #                 annotation_clip=False, fontsize=6, rotation=90, va='bottom', ha='center')

        axs[0].axvline(
            3000 + (s - p) * 100, color="red", linestyle="dotted", linewidth=0.5
        )
        axs[1].axvline(
            3000 + (s - p) * 100, color="red", linestyle="dotted", linewidth=0.5
        )

    axs[0].set_title(
        "Normalized and filtered input for Z(blue), N(orange) and E(green)",
        fontdict={"fontsize": 8},
    )
    t = np.linspace(1, 6000, num=6000)
    scale_x = 100  # milliscds to scd
    ticks_x = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_x))
    axs[1].xaxis.set_major_formatter(ticks_x)
    # fig.tight_layout()

    axs[1].axhline(ma, color="indigo", linestyle="dashed", linewidth=0.7)
    axs[1].plot(output, color="mediumvioletred", alpha=1, linewidth=0.7)
    xmin, xmax = axs[1].get_xlim()
    axs[1].annotate("Real magnitude(" + str(np.round(ma, decimals=2)) + ")", xy=(xmin, ma),
                    xytext=(2, 2), textcoords='offset points',
                    annotation_clip=False, fontsize=6)

    # axs[1,0].axvline(random_point, color="black")
    # axs[2,0].axvline(random_point, color="black")
    # axs[1].set_title("Computed distance versus real distance", fontdict={"fontsize": 8})
    # scale_y = 1000  # metres to km
    # ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_y))
    # axs[1].yaxis.set_major_formatter(ticks_y)
    # axs[1].axhline(distance, color="black", linestyle="dashed")
    # axs[1].plot(t, real_output, color="green", alpha=0.7, linewidth = 0.5)
    # axs[1].set_ylabel("Distance in km",fontdict={"fontsize": 8})

    axs[1].set_ylabel("Magnitude", fontdict={"fontsize": 8})
    axs[1].set_title("Predicted magnitude", fontdict={"fontsize": 8})
    fig.tight_layout()

    # plt.plot(t,mean_squared_error(s_output,s_labels),":")
    fig.savefig("MA:Prediction Plot" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), dpi=600)


def timespan_iteration(catalog_path, checkpoint_path, hdf5_path, timespan_array):

    for t in timespan_array:
        t = int(t)
        predtrue_timespan(catalog_path, checkpoint_path, hdf5_path, t)


def predtrue_timespan(catalog_path, checkpoint_path, hdf5_path, timespan=None):
    # load catalog
    catalog = pd.read_csv(catalog_path)
    test_catalog = catalog[catalog["SPLIT"] == "TEST"]
    above = True
    if above is True:
        test_catalog=test_catalog[test_catalog["MA"]>=5]
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)

    # load model
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.freeze()

    # load scaler
    mag = np.array([0, 9])
    # print(max(test_catalog["DIST"]))
    assert max(test_catalog["MA"]) <= 9
    assert min(test_catalog["MA"]) >= 0
    scaler = MinMaxScaler()
    scaler.fit(mag.reshape(-1, 1))

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
            event, station, magnitude, p, s = test_catalog.iloc[idx][
                ["EVENT", "STATION", "MA", "P_PICK", "S_PICK"]
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

            station_stream, max_stream = normalize_stream(waveform)
            station_stream = torch.from_numpy(station_stream[None])
            station_stream = station_stream.to(device)
            ms = np.float32(0.001 * np.log(max_stream))
            ms = torch.tensor(ms).unsqueeze(-1)
            ms = ms.to(device)
            model_input = (station_stream,  ms)
            outputs = model(model_input).squeeze()
            learned = outputs.unsqueeze(-1)
            

            # check if the s pick already arrived
            if s and (s - p) * 100 < (seq_len - random_point):
                # print("S Pick included, diff: ", (s - p), (seq_len - random_point) / 100)
                learn_s = torch.cat((learn_s, learned), 0)
                # var_s = torch.cat((var_s, variance), 0)
                true_s = true_s + [magnitude]

            else:
                learn = torch.cat((learn, learned), 0)
                # var = torch.cat((var, variance), 0)
                true = true + [magnitude]

        learn = learn.cpu()
        # var = var.cpu()

        learn = np.delete(learn, 0)
        # var = np.delete(var, 0)
        # sig = np.sqrt(var)
        pred = scaler.inverse_transform(learn.reshape(-1, 1)).squeeze()

        learn_s = learn_s.cpu()
        # var_s = var_s.cpu()
        swaves=False
        if learn_s.shape != torch.Size([1]):  # no element was added during loop
            learn_s = np.delete(learn_s, 0)
            pred_s = scaler.inverse_transform(learn_s.reshape(-1, 1)).squeeze()
            swaves=True

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
        "Predicted and true magnitude values, \ndifferentiating between recordings with and without a S-Wave arrival",
        fontsize=10,
    )
    if above is True:
        fig.suptitle(
        "Predicted and true magnitude values for magnitudes above 5, \ndifferentiating between recordings with and without a S-Wave arrival",
        fontsize=10,
    )   
    x = np.array(true)
    y = pred
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    cm = plt.cm.get_cmap("spring")
    x, y, z = x[idx], y[idx], z[idx]
    z *= len(x) / z.max()

    a = axs.scatter(
        x,
        y,
        c=z,
        cmap=cm,
        s=0.2,
        marker="s",
        lw=0,
        alpha=0.5,
    )
    if swaves is True:
        x = np.array(true_s)
        y = pred_s
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        cm = plt.cm.get_cmap("cividis")
        z *= len(x) / z.max()

        b = axs.scatter(
            x,
            y,
            s=0.2,
            c=z,
            cmap=cm,
            marker="D",
            lw=0,
            alpha=0.5,
        )
        bc = fig.colorbar(b)
        bc.ax.tick_params(labelsize=8)
        bc.ax.set_ylabel('S-Waves arrived', fontsize=8)
        
    if timespan is not None:
        axs.legend(
            title=str(timespan) + " seconds after P-Wave arrival",
            loc="best",
            fontsize=7,
            title_fontsize=8,
        )
    axs.axline((0, 0), (9, 9), linewidth=0.3, color="black")
    plt.axis("square")
    plt.xlabel("True magnitude", fontsize=8)
    plt.ylabel("Predicted magnitude", fontsize=8)
    ac = fig.colorbar(a, fraction=0.046, pad=0.04)
    # ac.ax.shrink = 0.8
    ac.ax.tick_params(labelsize=8)
    ac.ax.set_ylabel('No S-Waves present', fontsize=8)
    
    if timespan is not None:
        fig.savefig(
            "Magnitude:PredVSTrue_" +str(above)+"_" + str(timespan).replace(".", "_") + "sec", dpi=600
        )
    else:
        fig.savefig("Magnitude:PredVSTrue_" + str(above), dpi=600)

    # Plot without differentiation
    fig, axs = plt.subplots(1)
    axs.tick_params(axis="both", labelsize=8)
    fig.suptitle("Predicted and true magnitude values")
    # " \nRSME = " + str(
    #    rsme), fontsize=10)
    if above is True:
        fig.suptitle("Predicted and true magnitude values for magnitudes above 5")
    if swaves is True:
        x = np.array(true + true_s)
        y = np.append(pred, pred_s)
    else:
        x = np.array(true)
        y = pred       
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    cm = plt.cm.get_cmap("plasma")
    x, y, z = x[idx], y[idx], z[idx]
    z *= len(x) / z.max()

    a = axs.scatter(
        x,
        y,
        c=z,
        cmap=cm,
        s=0.2,
        marker="s",
        lw=0,
        alpha=0.5,
    )
    ac = fig.colorbar(a, fraction=0.046, pad=0.04)
    # ac.ax.shrink = 0.8
    ac.ax.tick_params(labelsize=8)
    ac.ax.set_ylabel('Number of points', fontsize=8)
    if timespan is not None:
        axs.legend(
            title=str(timespan) + " seconds after P-Wave arrival",
            loc="best",
            fontsize=8,
            title_fontsize=8,
        )
    axs.axline((0, 0), (9, 9), linewidth=0.5, color="black")
    plt.axis("square")
    plt.xlabel("True magnitude", fontsize=8)
    plt.ylabel("Predicted magnitude", fontsize=8)
    if timespan is not None:
        fig.savefig(
            "Magnitude:PredVSTrue_simple_"+str(above)+"_" + str(timespan).replace(".", "_") + "sec",
            dpi=600,
        )
    else:
        fig.savefig("Magnitude:PredVSTrue_simple_"+str(above), dpi=600)


def rsme_timespan(catalog_path, checkpoint_path, hdf5_path):
    # load catalog
    catalog = pd.read_csv(catalog_path)
    test_catalog = catalog[catalog["SPLIT"] == "TEST"]
    above = True
    if above is True:
        test_catalog=test_catalog[test_catalog["MA"]>=5]
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
    magmax = 9
    magmin = 0
    # print(max(test_catalog["DIST"]))
    assert max(test_catalog["MA"]) <= magmax
    assert min(test_catalog["MA"]) >= magmin
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
            mag = np.array(test_catalog["MA"])
            mag = [[m] for m in mag]
            mag = torch.tensor(mag, device=device)
            for idx in range(0, len(test_catalog)):
                event, station, magnitude, p, s = test_catalog.iloc[idx][
                    ["EVENT", "STATION", "MA", "P_PICK", "S_PICK"]
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

                station_stream, max_stream = normalize_stream(waveform)
                station_stream = torch.from_numpy(station_stream[None])
                station_stream = station_stream.to(device)
                ms = np.float32(0.001 * np.log(max_stream))
                ms = torch.tensor(ms).unsqueeze(-1)
                ms = ms.to(device)
                model_input = (station_stream, ms)
                outputs = model(model_input).squeeze()
                learned = outputs.unsqueeze(-1)

                # check if the s pick already arrived
                if s and (s - p) * 100 < (seq_len - random_point):
                    # print("S Pick included, diff: ", (s - p), (seq_len - random_point) / 100)
                    learn_s = torch.cat((learn_s, learned))
                    # var_s = torch.cat((var_s, variance))
                    true_s = torch.cat((true_s, mag[idx]))
                else:
                    learn = torch.cat((learn, learned))
                    # var = torch.cat((var, variance))
                    true = torch.cat((true, mag[idx]))

            # delete the dummy variable
            learn_s = learn_s[learn_s != learn_s[0]]
            # var_s = var_s[var_s != var_s[0]]
            learn = learn[learn != learn[0]]
            # var = var[var != var[0]]
            true = true[true != true[0]]
            true_s = true_s[true_s != true_s[0]]

            # scale back to km
            learn_scaled = torch.add(torch.mul(learn, magmax), magmin)
            learn_s_scaled = torch.add(torch.mul(learn_s, magmax), magmin)
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
    if above is True:
        fig.suptitle("RSME after the P-arrival depending on S-arrivals for magnitudes above 5", fontsize=10)
    
    axs.plot(
        timespan,
        np.array(rsme_p),
        linewidth=0.7,
        label="Recordings without a S-Wave arrival",
        color="steelblue",
    )
    axs.plot(
        timespan,
        np.array(rsme_s),
        linewidth=0.7,
        label="Recordings in which there is a S-Wave arrival",
        color="mediumvioletred",
    )
    axs.plot(
        timespan,
        np.array(rsme),
        linewidth=0.7,
        label="All recordings",
        color="springgreen",
    )
    axs.legend(fontsize=8, loc="best")
    plt.xlabel("Time after P-Wave arrival[sec]", fontsize=8)
    plt.ylabel("RSME", fontsize=8)
    fig.savefig("Magnitude:RSME_"+str(above), dpi=600)


# learn(cp, hp, mp)
#predict(cp, hp, chp)

# predtrue_timespan(catalog_path=cp, checkpoint_path=chp, hdf5_path=hp, timespan = 4)
timespan_iteration(cp, chp, hp, timespan_array=[2,4,8, 16])
#rsme_timespan(cp,chp,hp)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--catalog_path", type=str)
    parser.add_argument("--hdf5_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--hparams_file", type=str)
    parser.add_argument("--timespan", type=list)
    parser.add_argument("--time", type= int)
    args = parser.parse_args()
    action = args.action

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
    if action == "timespan":
        timespan_iteration(
            catalog_path=args.catalog_path,
            hdf5_path=args.hdf5_path,
            checkpoint_path=args.checkpoint_path,
            timespan_array=args.timespan,
        )
    if action == "predtrue":
        predtrue_timespan(
            catalog_path=args.catalog_path,
            hdf5_path=args.hdf5_path,
            checkpoint_path=args.checkpoint_path,
            timespan=args.timespan,
        )