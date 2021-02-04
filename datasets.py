from __future__ import print_function, division

import os

import numpy as np
import obspy
import pandas as pd
import torch
from obspy import UTCDateTime
from torch.utils.data import Dataset, DataLoader


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

class SeismoDataset(Dataset):
    def __init__(
            self,
            catalog_path,
            waveform_path,
            split,
            time_before=2,
            time_after=2,
            test_run=False,
    ):
        #         data = pd.read_csv(catalog_path)
        # #        print(len(data))
        #         data = data[data['STATION'] != 'LVC']
        #         events = sorted(data['EVENT'].unique())
        #         data = data[data['EVENT'].isin(events)]
        #         print(len(data))
        #         misses = 0
        #         for event in events:
        #             #found = False
        #             for waveform_paths in tqdm(waveform_path):
        #                 path = os.path.join(waveform_paths, f'{event}.mseed')
        #                 if not os.path.isfile(path):
        #                     if event in events:
        #                         events.remove(event)
        #                         misses += 1
        # #                        print(misses)
        #         if misses:
        #             print(f'Could not find {misses} files')
        #             data = data[data['EVENT'].isin(events)]
        #         print(len(data))
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
        print(len(data))
        if split is not None:
            if split in ["TRAIN", "DEV", "TEST"]:
                data = data[data["SPLIT"] == split]
        if test_run:
            data = data.iloc[:100]
        self.data = data
        self.waveform_path = waveform_path
        self.time_before = time_before
        self.time_after = time_after
        self.idx_changes = 0

    def __len__(self):
        return (
                len(self.data) * 2
        )  # because the noise examples are generated from the same data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        while True:
            if idx >= len(self.data):  # noise example
                label = np.int64(0)
                idx = idx % len(
                    self.data
                )  # get the same index like a waveform sample, just cut out another stream

                event, station, time = self.data.iloc[idx][["EVENT", "STATION", "TIME"]]
                waveform = obspy.read(
                    os.path.join(self.waveform_path, f"{event}.mseed")
                )
                station_stream = waveform.select(
                    station=station, channel="HH*"
                )  # high frequency
                station_stream = station_stream.slice(
                    starttime=UTCDateTime(time), endtime=UTCDateTime(time + 4)
                )
                station_stream.filter("highpass", freq=2, zerophase=True)

                station_stream.detrend().normalize()

            else:
                label = np.int64(1)
                event, station, p_pick = self.data.iloc[idx][
                    ["EVENT", "STATION", "P_PICK"]
                ]
                waveform = obspy.read(
                    os.path.join(self.waveform_path, f"{event}.mseed")
                )
                seq_len = self.time_before + self.time_after
                random_point = np.random.uniform(0, 4)
                start_time = UTCDateTime(p_pick - random_point)
                end_time = UTCDateTime(p_pick + (seq_len - random_point))
                station_stream = waveform.select(
                    station=station, channel="HH*"
                )  # high frequency
                station_stream = station_stream.slice(
                    starttime=start_time, endtime=end_time
                )
                station_stream.filter("highpass", freq=2, zerophase=True)
                station_stream.detrend().normalize()

            if len(station_stream) >=3:
                for trace in station_stream:
                    resample_trace(trace, sampling_rate=100)
                trace_z = np.array(station_stream.select(component="Z"))
                trace_n = np.array(station_stream.select(component="N"))
                trace_e = np.array(station_stream.select(component="E"))
                trace_z = trace_z[0]
                trace_z = trace_z[:400]
                trace_n = trace_n[0]
                trace_n = trace_n[:400]
                trace_e = trace_e[0]
                trace_e = trace_e[:400]
                waveform_np = np.float32(np.stack((trace_z, trace_n, trace_e)))
                # print(traces.shape)
                # waveform_np = np.squeeze(traces) #remove dummy dimension (3,1,401)
                # print(waveform_np.shape)
                # because I want a length of 400, not 401, I leave out the last element
                if waveform_np.shape == (3, 400):
                    if (np.any(waveform_np) is None) or (label is None):
                        print(waveform_np, label)
                    sample = {"waveform": waveform_np, "label": label}
                    #                print(self.idx_changes)
                    return sample
                else:
                    print("waveform does not have the right shape, instead:", waveform_np.shape)
                    rng = np.random.default_rng()
                    idx = rng.integers(0, len(self.data))
                    self.idx_changes = self.idx_changes + 1
                    if (self.idx_changes / (len(self.data) * 2)) % 2 == 0:
                        print("station missing in event  stream: ", self.idx_changes, 'percentage: ',
                              self.idx_changes / len(self.data))
                # really unfortunate hack, because in the small dataset there exist stations which are not listed
                # in the waveform streams
            else:
                rng = np.random.default_rng()
                idx = rng.integers(0, len(self.data))
                self.idx_changes = self.idx_changes + 1
                if (self.idx_changes/(len(self.data)*2)) %2 == 0:
                    print("station missing in event stream: ", self.idx_changes, 'percentage: ',
                          self.idx_changes / len(self.data))
                # really unfortunate hack, because in the small dataset there exist stations which are not listed
                # in the waveform streams


def get_data_loaders(
        catalog_path,
        waveform_path,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        test_run=False,
):
    if test_run:
        num_workers = 1

    training_data = SeismoDataset(
        catalog_path=catalog_path,
        waveform_path=waveform_path,
        split="TRAIN",
        test_run=test_run,
    )
    validation_data = SeismoDataset(
        catalog_path=catalog_path,
        waveform_path=waveform_path,
        split="DEV",
        test_run=test_run,
    )

    training_loader = DataLoader(
        training_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return training_loader, validation_loader


def get_test_loader(
        catalog_path, waveform_path, batch_size=64, num_workers=4, test_run=False
):
    if test_run:
        num_workers = 1
    test_data = SeismoDataset(
        catalog_path=catalog_path,
        waveform_path=waveform_path,
        split="TEST",
        test_run=test_run,
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return test_loader
