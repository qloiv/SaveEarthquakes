import os

import h5py
import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from scipy import signal
from tqdm import tqdm


def obspy_detrend(data):
    # based on obspys detrend ("simple") function
    if not np.issubdtype(data.dtype, np.floating):
        data = np.require(data, dtype=np.float64)
    ndat = len(data)
    x1, x2 = data[0], data[-1]
    data -= x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1)
    return data


def normalize_stream(stream, global_max=False):
    stream_max = np.abs(stream).max()
    if global_max is True:
        stream /= stream_max
    else:
        i = 0
        for tr in stream:
            ma_tr = np.abs(tr).max()
            if ma_tr == 0:
                i += 1
            else:
                tr /= ma_tr
        if i == 3:
            print("Der gesamte Stream ist 0")
    return stream, stream_max


cp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog_sensitivity.csv"
wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset_sensitivity.h5"
mp = "/home/viola/WS2021/Code/Models"
chp = "/home/viola/WS2021/Code/tb_logs/distance/version_47/checkpoints/epoch=19-step=319.ckpt"
hf = "/home/viola/WS2021/Code/tb_logs/distance/version_47/hparams.yaml",
catalog_path = cp
hdf5_path = hp
waveform_path = wp
# waveform_path_add =
inv_path = "/home/viola/WS2021/Jannes Daten/Daten2/inventory.xml"

# load train catalog

catalog = pd.read_csv(catalog_path)
train_catalog = catalog[catalog["SPLIT"] == "TRAIN"]
split_key = "train_files"
file_path = hdf5_path
h5data = h5py.File(file_path, "r").get(split_key)

# load magnitude
magnitude = np.array(train_catalog["MA"])
# load distance
distance = np.array(train_catalog["DIST"])

# for every item in catalog
# locate 4 seconds after p Pick --> this is clear as its 3000-3003
# compute peak displacement
# load catalog with random test event

max_disp = np.zeros(len(train_catalog))
# set high pass filter
hfilt = signal.butter(2, 2, btype="highpass", fs=100, output="sos")
# set low pass filter
lfilt = signal.butter(2, 35, btype="lowpass", fs=100, output="sos")
for idx in tqdm(range(0, len(train_catalog))):
    event, station, p_pick = train_catalog.iloc[idx][["EVENT", "STATION", "P_PICK"]]

    # load hdf5 waveform
    raw_waveform = np.array(h5data.get(event + "/" + station))
    seq_len = 4 * 100  # 4seconds *sampling rate
    p_pick_array = 3000  # ist bei 3000 weil obspy null indiziert arbeitet, also die Startzeit beginnt bei array 0

    # load obpsy waveforms
    o_raw_waveform = (obspy.read(os.path.join(waveform_path, f"{event}.mseed")))  # + (
    # obspy.read(os.path.join(waveform_path_add, f"{event}.mseed"))
    # )

    o_waveform = o_raw_waveform.select(station=station, channel="HHZ")
    o_station_stream = o_waveform.slice(
        starttime=UTCDateTime(p_pick),  #
        endtime=UTCDateTime(p_pick) + 3.99,
    )  # -0.01 deletes the last item, therefore enforcing array indexing

    # load inventory
    inv = obspy.read_inventory(inv_path)
    inv_selection = inv.select(station=station, channel="HHZ")

    new_stream_w30 = o_station_stream.copy()

    disp_w30 = new_stream_w30.copy().remove_response(
        inventory=inv_selection, pre_filt=None, output="DISP", water_level=60
    )
    # disp_w30.plot()
    m = disp_w30.max()  # already returns absolute maximum amplitude
    max_disp[idx] = m[0]

max_disp = max_disp / 100
print(max_disp)
np.save("max_disp.npy", max_disp)
