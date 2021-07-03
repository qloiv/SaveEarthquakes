from __future__ import print_function, division

import argparse
import os
from random import randrange

import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import pytorch_lightning as pl
import torch
from obspy import UTCDateTime
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import signal

from datasets_detection import obspy_detrend, normalize_stream
from litdatamodule_detection import LitDataModule
from litnetwork_detection import LitNetwork

cp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog.csv"
wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset.h5"
mp = "/home/viola/WS2021/Code/Models"


def learn(catalog_path, hdf5_path, model_path):
    network = LitNetwork()
    dm = LitDataModule(catalog_path=catalog_path, hdf5_path=hdf5_path, batch_size=128)
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
    dm = LitDataModule(catalog_path, hdf5_path, batch_size=128)
    # init trainer with whatever options
    trainer = pl.Trainer(gpus=[0])

    # test (pass in the model)
    trainer.test(model, datamodule=dm)


def test_one_displacement(
        catalog_path, checkpoint_path, hdf5_path, waveform_path, waveform_path_add, inv_path
):
    # load catalog with random test event
    catalog = pd.read_csv(catalog_path)
    test_catalog = catalog[catalog["SPLIT"] == "TEST"]
    idx = randrange(0, len(test_catalog))
    print(idx)
    event, station, p_pick = test_catalog.iloc[idx][["EVENT", "STATION", "P_PICK"]]

    # load network
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.freeze()

    # load hdf5 waveform
    raw_waveform = np.array(h5data.get(event + "/" + station))
    seq_len = 4 * 100  # *sampling rate
    p_pick_array = 3000  # ist bei 3000 weil obspy null indiziert arbeitet, also die Startzeit beginnt bei array 0
    # wir haben eine millisekunde zu viel, weil ich im preprocessing 30s vor und nach dem p Pick auswähle
    random_point = np.random.randint(seq_len)
    waveform = raw_waveform[
               :, p_pick_array - random_point: p_pick_array + (seq_len - random_point)
               ]

    # load obpsy waveforms
    o_raw_waveform = (obspy.read(os.path.join(waveform_path, f"{event}.mseed"))) + (
        obspy.read(os.path.join(waveform_path_add, f"{event}.mseed"))
    )

    o_waveform = o_raw_waveform.select(station=station, channel="HH*")
    o_station_stream = o_waveform.slice(
        starttime=UTCDateTime(p_pick) - random_point / 100,  #
        endtime=UTCDateTime(p_pick) + (4.00 - random_point / 100) - 0.01,
    )  # -0.01 deletes the last item, therefore enforcing array indexing

    # load inventory
    inv = obspy.read_inventory(inv_path)
    inv_selection = inv.select(station=station, channel="HH*")

    new_stream = o_station_stream.copy()
    assert np.all(o_station_stream[0].data == waveform[2])
    # new_stream[0].data = waveform[2]
    assert np.all(o_station_stream[1].data == waveform[1])
    # new_stream[1].data = waveform[1]
    assert np.all(o_station_stream[2].data == waveform[0])
    # new_stream[2].data = waveform[0]

    # the same routines as in test_one
    fig, axs = plt.subplots(3)
    fig.suptitle("Input of Detection Network - full Trace")
    axs[0].plot(raw_waveform[0], "r")
    axs[1].plot(raw_waveform[1], "b")
    axs[2].plot(raw_waveform[2], "g")
    fig.savefig("TestOneD:Full Trace")

    fig, axs = plt.subplots(3)
    fig.suptitle("Cut Out Input")
    axs[0].plot(waveform[0], "r")
    axs[1].plot(waveform[1], "b")
    axs[2].plot(waveform[2], "g")
    fig.savefig("TestOneD: Input")
    d0 = obspy_detrend(waveform[0])
    d1 = obspy_detrend(waveform[1])
    d2 = obspy_detrend(waveform[2])
    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending")
    axs[0].plot(d0, "r")
    axs[1].plot(d1, "b")
    axs[2].plot(d2, "g")
    fig.savefig("TestOneD:Detrended")

    # set high pass filter
    sampling_rate = 100
    filt = signal.butter(2, 2, btype="highpass", fs=sampling_rate, output="sos")
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
    axs[0].plot(g0, "r")
    axs[1].plot(g1, "b")
    axs[2].plot(g2, "g")
    fig.savefig("TestOneD:Detrended and Filtered")

    waveform = np.stack((g0, g1, g2))
    waveform, _ = normalize_stream(waveform)
    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending->Filtering->Normalizing")

    axs[0].plot(waveform[0], "r")
    axs[1].plot(waveform[1], "b")
    axs[2].plot(waveform[2], "g")
    fig.savefig("TestOneD: Detrended, Filtered and Normalized")

    station_stream = torch.from_numpy(waveform[None])
    out = model(station_stream)
    _, predicted = torch.max(out, 1)
    print(predicted)

    # compute and plot displacement
    # relying on asserts from before
    new_stream[0].data = waveform[2]
    new_stream[1].data = waveform[1]
    new_stream[2].data = waveform[0]
    # TODO warum umsortiert
    disp = new_stream.copy().remove_response(
        inventory=inv_selection, pre_filt=None, output="DISP", water_level=30
    )
    disp.plot()

    fig, axs = plt.subplots(6)
    fig.suptitle(
        "Modified data with P-Pick, was detected as P-Wave? " + str(bool(predicted))
    )
    axs[0].plot(waveform[0], "r")
    axs[1].plot(waveform[1], "b")
    axs[2].plot(waveform[2], "g")
    axs[0].axvline(random_point, color="black")
    axs[1].axvline(random_point, color="black")
    axs[2].axvline(random_point, color="black")
    axs[3].plot(disp[0].data)
    axs[4].plot(disp[1].data)
    axs[5].plot(disp[2].data)

    fig.savefig("TestOneD: Results")


def test_displacement(
        catalog_path, checkpoint_path, hdf5_path, waveform_path, waveform_path_add, inv_path
):
    # load catalog with random test event
    catalog = pd.read_csv(catalog_path)
    test_catalog = catalog[catalog["SPLIT"] == "TEST"]
    idx = randrange(0, len(test_catalog))
    # idx = 141
    print(idx)
    event, station, p_pick = test_catalog.iloc[idx][["EVENT", "STATION", "P_PICK"]]

    # load network
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.freeze()

    # load hdf5 waveform
    raw_waveform = np.array(h5data.get(event + "/" + station))
    seq_len = 4 * 100  # *sampling rate
    p_pick_array = 3000  # ist bei 3000 weil obspy null indiziert arbeitet, also die Startzeit beginnt bei array 0
    # wir haben eine millisekunde zu viel, weil ich im preprocessing 30s vor und nach dem p Pick auswähle
    random_point = np.random.randint(seq_len)
    # random_point = 150
    waveform = raw_waveform[
               :, p_pick_array - random_point: p_pick_array + (seq_len - random_point)
               ]

    # load obpsy waveforms
    o_raw_waveform = (obspy.read(os.path.join(waveform_path, f"{event}.mseed"))) + (
        obspy.read(os.path.join(waveform_path_add, f"{event}.mseed"))
    )

    o_waveform = o_raw_waveform.select(station=station, channel="HH*")
    o_station_stream = o_waveform.slice(
        starttime=UTCDateTime(p_pick) - random_point / 100,  #
        endtime=UTCDateTime(p_pick) + (4.00 - random_point / 100) - 0.01,
    )  # -0.01 deletes the last item, therefore enforcing array indexing

    # load inventory
    inv = obspy.read_inventory(inv_path)
    inv_selection = inv.select(station=station, channel="HH*")

    new_stream_w60 = o_station_stream.copy()
    new_stream_w30 = o_station_stream.copy()
    new_stream_w30_high = o_station_stream.copy()
    new_stream_w60_high = o_station_stream.copy()
    assert np.all(o_station_stream[0].data == waveform[2])
    # new_stream_w60[0].data = waveform[2]
    assert np.all(o_station_stream[1].data == waveform[1])
    # new_stream_w60[1].data = waveform[1]
    assert np.all(o_station_stream[2].data == waveform[0])
    # new_stream_w60[2].data = waveform[0]

    # the same routines as in test_one
    fig, axs = plt.subplots(3)
    fig.suptitle("Input of Detection Network - full Trace")
    axs[0].plot(raw_waveform[0], "r")
    axs[1].plot(raw_waveform[1], "b")
    axs[2].plot(raw_waveform[2], "g")
    fig.savefig("TestDisp:Full Trace")

    fig, axs = plt.subplots(3)
    fig.suptitle("Cut Out Input")
    axs[0].plot(waveform[0], "r")
    axs[1].plot(waveform[1], "b")
    axs[2].plot(waveform[2], "g")
    fig.savefig("TestDisp: Input")
    d0 = obspy_detrend(waveform[0])
    d1 = obspy_detrend(waveform[1])
    d2 = obspy_detrend(waveform[2])
    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending")
    axs[0].plot(d0, "r")
    axs[1].plot(d1, "b")
    axs[2].plot(d2, "g")
    fig.savefig("TestDisp:Detrended")

    # set high pass filter
    hfilt = signal.butter(2, 2, btype="highpass", fs=100, output="sos")
    f0 = signal.sosfilt(hfilt, d0, axis=-1).astype(np.float32)
    f1 = signal.sosfilt(hfilt, d1, axis=-1).astype(np.float32)
    f2 = signal.sosfilt(hfilt, d2, axis=-1).astype(np.float32)
    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending, then High-Pass filtering")
    axs[0].plot(f0, "r")
    axs[1].plot(f1, "b")
    axs[2].plot(f2, "g")
    fig.savefig("TestDisp:Detrended and High-Pass Filtered")

    # set low pass filter
    lfilt = signal.butter(2, 35, btype="lowpass", fs=100, output="sos")
    g0 = signal.sosfilt(lfilt, f0, axis=-1).astype(np.float32)
    g1 = signal.sosfilt(lfilt, f1, axis=-1).astype(np.float32)
    g2 = signal.sosfilt(lfilt, f2, axis=-1).astype(np.float32)
    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending, then High and Low-Pass filtering")
    axs[0].plot(g0, "r")
    axs[1].plot(g1, "b")
    axs[2].plot(g2, "g")
    fig.savefig("TestDisp:Detrended and High and Low-Pass Filtered")

    # build the two final waveforms, with and without lowpass
    waveform = np.stack((g0, g1, g2))
    waveform, _ = normalize_stream(waveform)
    fig, axs = plt.subplots(3)
    fig.suptitle("After Detrending->Filtering->Normalizing")

    waveform_highpass = np.stack((f0, f1, f2))
    waveform_highpass, _ = normalize_stream(waveform_highpass)

    axs[0].plot(waveform[0], "r")
    axs[1].plot(waveform[1], "b")
    axs[2].plot(waveform[2], "g")
    fig.savefig("TestDisp: Detrended, Filtered and Normalized")

    # evaluate both waveforms with the model
    station_stream = torch.from_numpy(waveform[None])
    out = model(station_stream)
    _, predicted = torch.max(out, 1)
    print(predicted)

    station_stream_highpass = torch.from_numpy(waveform_highpass[None])
    out = model(station_stream_highpass)
    _, predicted_highpass = torch.max(out, 1)

    # compute and plot displacement for the just highpass waveforms for two water levels,30 and 60
    # relying on asserts from before
    new_stream_w60_high[0].data = waveform_highpass[
        2
    ]  # changing obspy ordering to mine which is ZNE = 123
    new_stream_w60_high[1].data = waveform_highpass[1]
    new_stream_w60_high[2].data = waveform_highpass[0]
    disp_w60_high = new_stream_w60_high.copy().remove_response(
        inventory=inv_selection, pre_filt=None, output="DISP"
    )
    disp_w60_high.plot()

    fig, axs = plt.subplots(4, sharex=True)
    axs[3].set_xlabel("Time in seconds")
    fig.suptitle(
        "Modified data with P-Pick, was detected as P-Wave? "
        + str(bool(predicted_highpass))
        + "\nExample:"
        + str(idx)
        + ", Waterlevel:60"
        + " and only high-pass filter"
    )
    axs[0].plot(waveform_highpass[0], "r", linewidth=0.5)
    axs[0].plot(waveform_highpass[1], "b", linewidth=0.5)
    axs[0].plot(waveform_highpass[2], "g", linewidth=0.5)
    axs[0].axvline(random_point, color="black", linewidth=0.5)
    axs[0].set_title(
        "Normalized and filtered input for Z(red), N(blue) and E(green)",
        fontdict={"fontsize": 8},
    )

    # axs[1,0].axvline(random_point, color="black")
    # axs[2,0].axvline(random_point, color="black")
    axs[1].set_title("Displacement for Z, N, E", fontdict={"fontsize": 8})
    axs[1].plot(disp_w60_high[0].data, linewidth=0.5)
    axs[2].plot(disp_w60_high[1].data, linewidth=0.5)
    axs[3].plot(disp_w60_high[2].data, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(
        "TestDisp: Results with water level 60 and only High-Pass (2Hz) filter", dpi=600
    )

    # compute and plot displacement
    # relying on asserts from before
    new_stream_w30_high[0].data = waveform_highpass[2]
    new_stream_w30_high[1].data = waveform_highpass[1]
    new_stream_w30_high[2].data = waveform_highpass[0]
    disp_w30_high = new_stream_w30_high.copy().remove_response(
        inventory=inv_selection, pre_filt=None, output="DISP", water_level=30
    )
    disp_w30_high.plot()

    fig, axs = plt.subplots(4, sharex=True)
    axs[3].set_xlabel("Time in seconds")
    fig.suptitle(
        "Modified data with P-Pick, was detected as P-Wave? "
        + str(bool(predicted_highpass))
        + "\nExample:"
        + str(idx)
        + " Waterlevel:30"
        + " and only high-pass (2Hz) filter"
    )
    axs[0].plot(waveform_highpass[0], "r", linewidth=0.5)
    axs[0].plot(waveform_highpass[1], "b", linewidth=0.5)
    axs[0].plot(waveform_highpass[2], "g", linewidth=0.5)
    axs[0].axvline(random_point, color="black", linewidth=0.5)
    axs[0].set_title(
        "Normalized and filtered input for Z(red), N(blue) and E(green)",
        fontdict={"fontsize": 8},
    )

    # axs[1,0].axvline(random_point, color="black")
    # axs[2,0].axvline(random_point, color="black")
    axs[1].set_title("Displacement for Z, N, E", fontdict={"fontsize": 8})
    axs[1].plot(disp_w30_high[0].data, linewidth=0.5)
    axs[2].plot(disp_w30_high[1].data, linewidth=0.5)
    axs[3].plot(disp_w30_high[2].data, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(
        "TestDisp: Results with water level 30 and only high-pass filter", dpi=600
    )

    # compute and plot displacement for high and low pass and both water levels
    # relying on asserts from before
    new_stream_w60[0].data = waveform[2]
    new_stream_w60[1].data = waveform[1]
    new_stream_w60[2].data = waveform[0]
    disp_w60 = new_stream_w60.copy().remove_response(
        inventory=inv_selection, pre_filt=None, output="DISP"
    )
    disp_w60.plot()

    fig, axs = plt.subplots(4, sharex=True)
    axs[3].set_xlabel("Time in seconds")
    fig.suptitle(
        "Modified data with P-Pick, was detected as P-Wave? "
        + str(bool(predicted))
        + "\nExample:"
        + str(idx)
        + " Waterlevel:60 and high (2Hz) and low pass(35Hz) filter"
    )
    axs[0].plot(waveform[0], "r", linewidth=0.5)
    axs[0].plot(waveform[1], "b", linewidth=0.5)
    axs[0].plot(waveform[2], "g", linewidth=0.5)
    axs[0].axvline(random_point, color="black", linewidth=0.5)
    axs[0].set_title(
        "Normalized and filtered input for Z(red), N(blue) and E(green)",
        fontdict={"fontsize": 8},
    )

    # axs[1,0].axvline(random_point, color="black")
    # axs[2,0].axvline(random_point, color="black")
    axs[1].set_title("Displacement for Z, N, E", fontdict={"fontsize": 8})
    axs[1].plot(disp_w60[0].data, linewidth=0.5)
    axs[2].plot(disp_w60[1].data, linewidth=0.5)
    axs[3].plot(disp_w60[2].data, linewidth=0.5)
    fig.tight_layout()

    fig.savefig("TestDisp: Results with water level 60", dpi=600)

    # compute and plot displacement
    # relying on asserts from before
    new_stream_w30[0].data = waveform[2]
    new_stream_w30[1].data = waveform[1]
    new_stream_w30[2].data = waveform[0]
    disp_w30 = new_stream_w30.copy().remove_response(
        inventory=inv_selection, pre_filt=None, output="DISP", water_level=30
    )
    disp_w30.plot()

    fig, axs = plt.subplots(4, sharex=True)
    axs[3].set_xlabel("Time in seconds")
    fig.suptitle(
        "Modified data with P-Pick, was detected as P-Wave? "
        + str(bool(predicted))
        + "\nExample:"
        + str(idx)
        + " Waterlevel:30 and high (2Hz) and low pass(35) filter"
    )
    axs[0].plot(waveform[0], "r", linewidth=0.5)
    axs[0].plot(waveform[1], "b", linewidth=0.5)
    axs[0].plot(waveform[2], "g", linewidth=0.5)
    axs[0].axvline(random_point, color="black", linewidth=0.5)
    axs[0].set_title(
        "Normalized and filtered input for Z(red), N(blue) and E(green)",
        fontdict={"fontsize": 8},
    )

    # axs[1,0].axvline(random_point, color="black")
    # axs[2,0].axvline(random_point, color="black")
    axs[1].set_title("Displacement for Z, N, E", fontdict={"fontsize": 8})
    axs[1].plot(disp_w30[0].data, linewidth=0.5)
    axs[2].plot(disp_w30[1].data, linewidth=0.5)
    axs[3].plot(disp_w30[2].data, linewidth=0.5)
    fig.tight_layout()

    fig.savefig("TestDisp: Results with water level 30", dpi=600)


def test_one(catalog_path, checkpoint_path, hdf5_path):
    # load catalog with random test event
    catalog = pd.read_csv(catalog_path)
    test_catalog = catalog[catalog["SPLIT"] == "TEST"]
    idx = randrange(0, len(test_catalog))
    print(idx)
    event, station, p_pick = test_catalog.iloc[idx][["EVENT", "STATION", "P_PICK"]]

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
    p_pick_array = 3000
    random_point = np.random.randint(seq_len)
    waveform = raw_waveform[
               :, p_pick_array - random_point: p_pick_array + (seq_len - random_point)
               ]
    fig, axs = plt.subplots(3)
    fig.suptitle("Input of Detection Network - full Trace")
    axs[0].plot(raw_waveform[0], "r")
    axs[1].plot(raw_waveform[1], "b")
    axs[2].plot(raw_waveform[2], "g")
    fig.savefig("TestOne:Full Trace")

    fig, axs = plt.subplots(3)
    fig.suptitle("Cut Out Input")
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

    # set high pass filter
    sampling_rate = 100
    filt = signal.butter(2, 2, btype="highpass", fs=sampling_rate, output="sos")
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
    axs[0].plot(g0, "r")
    axs[1].plot(g1, "b")
    axs[2].plot(g2, "g")
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
    out = model(station_stream)
    _, predicted = torch.max(out, 1)
    print(predicted)
    fig, axs = plt.subplots(3, sharex = True)
    axs[2].set_xlabel("Time in hundredths of seconds")
    fig.suptitle(
        "Modified data with P-Pick, was detected as P-Wave? " + str(bool(predicted))
    )
    axs[0].plot(waveform[0], "r")
    axs[1].plot(waveform[1], "b")
    axs[2].plot(waveform[2], "g")
    axs[0].axvline(random_point, color="black")
    axs[1].axvline(random_point, color="black")
    axs[2].axvline(random_point, color="black")

    fig.savefig("TestOne: Results")


def predict(catalog_path, checkpoint_path, hdf5_path):
    # load catalog with random test event
    catalog = pd.read_csv(catalog_path)
    test_catalog = catalog[catalog["SPLIT"] == "TEST"]
    idx = randrange(0, len(test_catalog))
    print(idx)
    event, station, p_pick = test_catalog.iloc[idx][["EVENT", "STATION", "P_PICK"]]

    # load network
    split_key = "test_files"
    file_path = hdf5_path
    h5data = h5py.File(file_path, "r").get(split_key)
    model = LitNetwork.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
    )
    model.freeze()

    raw_waveform = np.array(h5data.get(event + "/" + station))
    filt = signal.butter(2, 2, btype="highpass", fs=100, output="sos")
    lfilt = signal.butter(2, 35, btype="lowpass", fs=100, output="sos")
    outs = np.zeros(len(raw_waveform[0]))
    for i in range(0, len(raw_waveform[0]) - 400):
        raw_waveform = np.array(h5data.get(event + "/" + station))  # reload stream
        window = raw_waveform[:, i: i + 400]
        d0 = obspy_detrend(window[0])
        d1 = obspy_detrend(window[1])
        d2 = obspy_detrend(window[2])

        f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
        f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
        f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)
        g0 = signal.sosfilt(lfilt, f0, axis=-1).astype(np.float32)
        g1 = signal.sosfilt(lfilt, f1, axis=-1).astype(np.float32)
        g2 = signal.sosfilt(lfilt, f2, axis=-1).astype(np.float32)

        station_stream = np.stack((g0, g1, g2))
        station_stream, _ = normalize_stream(station_stream)
        station_stream = torch.from_numpy(station_stream[None])
        out = model(station_stream)
        _, predicted = torch.max(out, 1)

        outs[i: i + 400] = outs[i: i + 400] + np.full(400, predicted[0])
    # print(outs)
    #    print(windowed_st)

    #    print("---")

    fig, axs = plt.subplots(4, constrained_layout=True)
    fig.suptitle("Input: Raw data with P-Pick with detection vector ")
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

    axs[0].plot(station_stream[0], "r")
    axs[1].plot(station_stream[1], "b")
    axs[2].plot(station_stream[2], "g")
    axs[3].plot(outs)
    axs[0].axvline(3000, color="black")
    axs[1].axvline(3000, color="black")
    axs[2].axvline(3000, color="black")
    axs[3].axvline(3000, color="black")
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


#test_displacement(
#    catalog_path=cp,
#    hdf5_path=hp,
#    checkpoint_path="../tb_logs/detection/version_8/checkpoints/epoch=22-step=91.ckpt",
#    waveform_path="/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/",
#    waveform_path_add="/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/",
#    inv_path="/home/viola/WS2021/Code/Daten/Chile_small/inventory.xml",
#)

# learn(cp, hp, mp)
# predict(catalog_path=cp, hdf5_path=hp,
# checkpoint_path="../tb_logs/detection/version_8/checkpoints/epoch=22-step=91.ckpt")
# test(catalog_path=cp,hdf5_path=hp,checkpoint_path="../tb_logs/detection/version_2/checkpoints/epoch=22-step=91.ckpt",hparams_file="../tb_logs/detection/version_2/hparams.yaml")
# test_one(catalog_path=cp, hdf5_path=hp,checkpoint_path="../tb_logs/detection/version_8/checkpoints/epoch=22-step=91.ckpt")
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

    if action == "learn":
        learn(
            catalog_path=args.catalog_path,
            hdf5_path=args.hdf5_path,
            model_path=args.model_path,
        )
    if action == "test":
        test(
            catalog_path=args.catalog_path,
            checkpoint_path=args.checkpoint_path,
            hparams_file=args.hparams_file,
            hdf5_path=args.hdf5_path,
        )
    if action == "test_one":
        test_one(
            catalog_path=args.catalog_path,
            checkpoint_path=args.checkpoint_path,
            hdf5_path=args.hdf5_path,
        )
    if action == "predict":
        predict(
            catalog_path=args.catalog_path,
            hdf5_path=args.hdf5_path,
            checkpoint_path=args.checkpoint_path,
        )
