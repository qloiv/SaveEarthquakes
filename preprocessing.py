# creates a hdf5 file and a new catalogue
# guarantees that the stations in the event waveforms are the same, as the station and events in the catalog
# it also exludes waveforms, which are too short
# additionally every waveform is resampled to 100Hz
# no data augmentation or filtering

import argparse
import os

import h5py
import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy import read_inventory
from tqdm import tqdm


#
# def filter_missing_files(data, events, input_dirs):
#     misses = 0
#     for event in events:
#         found = False
#         for waveform_path in input_dirs:
#             path = os.path.join(waveform_path, f"{event}.mseed")
#             if os.path.isfile(path):
#                 found = True
#                 # print(f'Missing file: {path}')
#         if not found:
#             misses += 1
#             events.remove(event)
#     if misses:
#         print(f"Could not find {misses} files")
#         data = data[data["EVENT"].isin(events)]
#     return data


def resample_trace(trace, sampling_rate):
    if trace.stats.sampling_rate == sampling_rate:
        return
    if trace.stats.sampling_rate % sampling_rate == 0:
        trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
    else:
        trace.resample(sampling_rate)


cp = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
csvp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog_sensitivity.csv"
hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset_sensitivity.h5"
inv_path = "/home/viola/WS2021/Code/Daten/Chile_small/inventory.xml"


def preprocess(catalog_path, waveform_path, new_catalog_path, hdf5_path, inventory):
    # os.remove(csv_path)
    if os.path.exists(
        hdf5_path
    ):  # we need this, because 'a' is always going to append?
        os.remove(hdf5_path)
    hf = h5py.File(hdf5_path, "a")
    train_hf = hf.create_group("train_files")
    test_hf = hf.create_group("test_files")
    dev_hf = hf.create_group("dev_files")

    data = pd.read_csv(catalog_path)
    print("Original data length: ", len(data))
    original_length = len(data)
    data = data[data["STATION"] != "LVC"]
    events = sorted(data["EVENT"].unique())
    print(
        "Percentage of data left after deleting faulty station and duplicate events: ",
        len(data) / original_length,
    )
    waves = []
    for file in os.listdir(waveform_path):
        if file.endswith(".mseed"):
            waves.append(file.strip(".mseed"))
    # also add files in waveforms_long_additional
    # for file in os.listdir(waveform_path_add):
    #    if file.endswith(".mseed"):
    #        waves.append(file.strip(".mseed"))
    print("Number of waveforms found: ", len(waves))
    intersect = list(set(waves).intersection(set(events)))
    data = data[data["EVENT"].isin(intersect)]
    print(
        "Percentage of data left after deleting events which are missing from waveform files or the other way round: ",
        len(data) / original_length,
    )
    stream_miss = 0
    short_pick = 0
    trace_miss = 0
    new_frame = data.copy()
    data_length = len(data)
    print("Data length before having a look at every trace: ", data_length)
    inv = read_inventory(inventory)
    for idx in tqdm(range(0, data_length)):
        current_row = data.iloc[idx]
        event, station, p_pick, split = current_row[
            ["EVENT", "STATION", "P_PICK", "SPLIT"]
        ]

        # we don t need to check whether this exists, because one of these should definitely exist because we tested before and you can add to empty streams?
        waveform = obspy.read(
            os.path.join(waveform_path, f"{event}.mseed")
        )  # +(obspy.read(os.path.join(waveform_path_add, f"{event}.mseed")))

        assert waveform is not None

        station_stream = waveform.select(
            station=station, channel="HH*"
        )  # HH = high frequency
        slice_length = 40
        # P Pick Zeit - 10 Sek. Falls das zu kurz ist, ist der gesamte Trace zu kurz und fliegt dann später mit raus.

        station_stream = station_stream.slice(
            starttime=UTCDateTime(p_pick - 30), endtime=UTCDateTime(p_pick + 30)
        )
        # station_stream.plot()
        # inv_select = inv.select(station=station, channel="HH*",starttime=UTCDateTime(p_pick - 30), endtime=UTCDateTime(p_pick + 30))
        station_stream.remove_sensitivity(inv)
        # station_stream.plot()
        if len(station_stream) < 3:
            stream_miss = stream_miss + 1
            # print("Percentage of data left because stream or trace is missing from the waveform file:
            # ",(original_length - stream_miss) / original_length)
            new_frame.drop(current_row.name, inplace=True)  # is index of dataframe
            continue

        for trace in station_stream:
            resample_trace(trace, sampling_rate=100)
        trace_z = np.array(station_stream.select(component="Z"))
        trace_n = np.array(station_stream.select(component="N"))
        trace_e = np.array(station_stream.select(component="E"))
        if (
                np.shape(trace_z) != (1, 6001)
                or np.shape(trace_e) != (1, 6001)
                or np.shape(trace_n) != (1, 6001)
        ):
            trace_miss = trace_miss + 1
            # print("Percentage of data left because trace shape was wrong",
            # (original_length - trace_miss) / original_length,)
            new_frame.drop(current_row.name, inplace=True)
            continue

        waveform_np = np.squeeze(np.float32(np.stack((trace_z, trace_n, trace_e))))
        key = str(event) + "/" + str(station)
        if split == "TRAIN":
            train_hf.create_dataset(key, data=waveform_np)
        elif split == "TEST":
            test_hf.create_dataset(key, data=waveform_np)
        elif split == "DEV":
            dev_hf.create_dataset(key, data=waveform_np)
        else:
            print("Not a valid split value")
            new_frame.drop(current_row.name, inplace=True)
            continue

    hf.close()
    new_frame.to_csv(new_catalog_path)
    print(
        "After having a look at every trace ",
        len(new_frame) / data_length,
        "% of the data is remaining, this is ",
        len(new_frame) / original_length,
        "of the original data.\n",
    )
    print(
        stream_miss / original_length,
        "% of data  was sorted out because the stream or trace is missing from the "
        "waveform file.\n",
    )
    print(
        trace_miss / original_length,
        "% data was sorted out because the trace shape was wrong.",
    )
    grouped_frame = (new_frame.groupby("SPLIT").count())["EVENT"]
    print(
        "There are",
        grouped_frame["TRAIN"],
        "train,",
        grouped_frame["TEST"],
        "test and",
        grouped_frame["DEV"],
        "dev examples.\n",
    )


preprocess(cp, wp, csvp, hp, inv_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--catalog_path", type=str, default=cp)
    parser.add_argument("--waveform_path", type=str, default=wp)
    # parser.add_argument("--waveform_path_add", type=str)
    parser.add_argument("--csv_path", type=str, default=csvp)
    parser.add_argument("--hdf5_path", type=str, default=hp)
    parser.add_argument("--inv_path", type=str, default=inv_path)
    args = parser.parse_args()
    action = args.action

    if action == "preprocess":
        preprocess(
            catalog_path=args.catalog_path,
            waveform_path=args.waveform_path,
            #           waveform_path_add= args.waveform_path,
            new_catalog_path=args.csv_path,
            hdf5_path=args.hdf5_path,
            inventory=args.inv_path,
        )
