# creates a hdf5 file and a new catalogue

import argparse
import os

import h5py
import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from tqdm import tqdm


def filter_missing_files(data, events, input_dirs):
    misses = 0
    for event in events:
        found = False
        for waveform_path in input_dirs:
            path = os.path.join(waveform_path, f"{event}.mseed")
            if os.path.isfile(path):
                found = True
                # print(f'Missing file: {path}')
        if not found:
            misses += 1
            events.remove(event)
    if misses:
        print(f"Could not find {misses} files")
        data = data[data["EVENT"].isin(events)]
    return data


def resample_trace(trace, sampling_rate):
    if trace.stats.sampling_rate == sampling_rate:
        return
    if trace.stats.sampling_rate % sampling_rate == 0:
        trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
    else:
        trace.resample(sampling_rate)


cp = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
wp = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
csvp = "/home/viola/WS2021/Code/Daten/Chile_small/new_catalog.csv"
hp = "/home/viola/WS2021/Code/Daten/Chile_small/hdf5_dataset.h5"


def preprocess(catalog_path, waveform_path, csv_path, hdf5_path):
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
    for idx in tqdm(range(0, data_length)):
        current_row = data.iloc[idx]
        event, station, p_pick, split = current_row[
            ["EVENT", "STATION", "P_PICK", "SPLIT"]
        ]

        waveform = obspy.read(
            os.path.join(waveform_path, f"{event}.mseed")
            # we don t need to check whether this exists, because we filtered by waveforms before
        )

        station_stream = waveform.select(
            station=station, channel="HH*"
        )  # HH = high frequency
        slice_length = 40
        # P Pick Zeit - 10 Sekunden. Falls das zu kurz ist, erstmal rauswerfen, weil ich ja noise example brauche.
        # Könnte man aber auch noch ändern.

        # if UTCDateTime(p_pick - 8) > UTCDateTime(time):  # angenommen time ist die startzeit
        #    short_pick = short_pick+1
        # print("Time before p pick is to short, percentage: ", short_pick/len(data))
        #    continue

        station_stream = station_stream.slice(
            starttime=UTCDateTime(p_pick - 10), endtime=UTCDateTime(p_pick + 30)
        )

        if len(station_stream) < 3:
            stream_miss = stream_miss + 1
            # print("Percentage of data left because stream or trace is missing from the waveform file: ",(original_length - stream_miss) / original_length)
            new_frame.drop(current_row[0], inplace=True)  # is index of dataframe
            continue

        for trace in station_stream:
            resample_trace(trace, sampling_rate=100)
        trace_z = np.array(station_stream.select(component="Z"))
        trace_n = np.array(station_stream.select(component="N"))
        trace_e = np.array(station_stream.select(component="E"))
        if (
                np.shape(trace_z) != (1, 4001)
                or np.shape(trace_e) != (1, 4001)
                or np.shape(trace_n) != (1, 4001)
        ):
            trace_miss = trace_miss + 1
            # print("Percentage of data left because trace shape was wrong",(original_length - trace_miss) / original_length,)
            new_frame.drop(current_row[0], inplace=True)
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
            new_frame.drop(current_row[0], inplace=True)
            continue

    hf.close()
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
    new_frame.to_csv(csv_path)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--catalog_path", type=str, default=cp)
    parser.add_argument("--waveform_path", type=str, default=wp)
    parser.add_argument("--csv_path", type=str, default=csvp)
    parser.add_argument("--hdf5_path", type=str, default=hp)
    args = parser.parse_args()
    action = args.action

    if action == "preprocess":
        preprocess(
            catalog_path=args.catalog_path,
            waveform_path=args.waveform_path,
            csv_path=args.csv_path,
            hdf5_path=args.hdf5_path,
        )
