import os
from multiprocessing import Pool

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
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


# cp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog_sensitivity.csv"
wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
# hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset_sensitivity.h5"
# mp = "/home/viola/WS2021/Code/Models"
# chp = "/home/viola/WS2021/Code/tb_logs/distance/version_47/checkpoints/epoch=19-step=319.ckpt"
# hf = "/home/viola/WS2021/Code/tb_logs/distance/version_47/hparams.yaml",
ip = "/home/viola/WS2021/Code/Daten/Chile_small/inventory.xml"
# cp = "../new_catalogue_sensitivity.csv"
# wp = "../../data/earthquake/waveforms_long_full"
wp2 = "../../data/earthquake/waveforms_long_additional"


# ip = "../inventory.xml"


def formulate(x, train_catalog):
    # load train catalog
    # catalog_path = cp
    # hdf5_path = hp
    waveform_path = wp
    waveform_path_add = wp
    inv_path = ip

    #    split_key = "train_files"
    #  file_path = hdf5_path
    #  h5data = h5py.File(file_path, "r").get(split_key)

    # for every item in catalog
    # locate 4 seconds after p Pick --> this is clear as its 3000-3003
    # compute peak displacement
    # load catalog with random test event

    event, station, p_pick, distance, magnitude = train_catalog[
        ["EVENT", "STATION", "P_PICK", "DIST", "MA"]
    ]
    # print(idx)
    # load hdf5 waveform
    # raw_waveform = np.array(h5data.get(event + "/" + station))
    #  seq_len = 4 * 100  # 4seconds *sampling rate
    # p_pick_array = 3000  # ist bei 3000 weil obspy null indiziert arbeitet, also die Startzeit beginnt bei array 0

    # load obpsy waveforms
    if os.path.getsize(os.path.join(waveform_path_add, f"{event}.mseed")) > 0:
        o_raw_waveform = obspy.read(
            os.path.join(waveform_path, f"{event}.mseed")
        ) + obspy.read(os.path.join(waveform_path_add, f"{event}.mseed"))
    else:
        o_raw_waveform = obspy.read(os.path.join(waveform_path, f"{event}.mseed"))

    o_waveform = o_raw_waveform.select(station=station, channel="HHZ")
    o_station_stream = o_waveform.slice(
        starttime=UTCDateTime(p_pick),  #
        endtime=UTCDateTime(p_pick) + 3.99,
    )  # -0.01 deletes the last item, therefore enforcing array indexing

    # load inventory
    inv = obspy.read_inventory(inv_path)
    inv_selection = inv.select(station=station, channel="HHZ")

    new_stream_w30 = o_station_stream.copy()

    disp_w30 = new_stream_w30.remove_response(
        inventory=inv_selection, pre_filt=None, output="DISP", water_level=30
    )
    # disp_w30.plot()
    m = np.max(np.abs(disp_w30))  # already returns absolute maximum amplitude
    # print(m*100)
    # assert(m>=0)
    return (m * 100, distance, magnitude)


pool = Pool(4)
catalog_path = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog_sensitivity.csv"
catalog = pd.read_csv(catalog_path)
train_catalog = catalog[catalog["SPLIT"] == "TRAIN"]
len_train = len(train_catalog)
results = [
    pool.apply_async(formulate, args=(x, train_catalog.iloc[x]))
    for x in tqdm(np.arange(0, len_train))
]
output = [p.get() for p in tqdm(results)]
print(output)
np.save("output2.npy", output)
