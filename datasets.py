from __future__ import print_function, division

import os

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import signal
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
            hdf5_path,
            split,
            time_before=2,
            time_after=2,
            test_run=False,
    ):
        self.split_key = str.lower(split) + "_files"
        self.file_path = hdf5_path
        self.h5data = None
        catalog = pd.read_csv(catalog_path)
        self.catalog = catalog[catalog["SPLIT"] == split]
        self.sampling_rate = 100
        self.p_pick = 1001
        self.split = str.lower(split) + "_files"
        self.time_before = time_before * self.sampling_rate
        self.time_after = time_after * self.sampling_rate
        self.idx_changes = 0


    def __len__(self):
        return (
                len(self.catalog) * 2
        )  # because the noise examples are generated from the same data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.h5data is None:
            self.h5data = h5py.File(self.file_path, "r").get(self.split_key)
        while True:
            if idx >= len(self.catalog):  # noise example
                label = np.int64(0)
                idx = idx % len(
                    self.catalog
                )  # get the same index like a waveform sample, just cut out another stream

                event, station = self.catalog.iloc[idx][["EVENT", "STATION"]]
                # in all waveforms in the hdf5 catalogue, the pick was placed at index 1001
                waveform = np.array(self.h5data.get(event + "/" + station))

                station_stream = waveform[:, self.p_pick - 500: self.p_pick - 100]

            else:
                label = np.int64(1)
                event, station = self.catalog.iloc[idx][["EVENT", "STATION"]]
                # in all waveforms in the hdf5 catalogue, the pick was placed at index 1001
                waveform = np.array(self.h5data.get(event + "/" + station))
                seq_len = self.time_before + self.time_after
                random_point = np.random.randint(seq_len)
                station_stream = waveform[:, self.p_pick - random_point: self.p_pick + (seq_len - random_point)]

            filt = signal.butter(
                2, 2, btype="highpass", fs=self.sampling_rate, output="sos"
            )
            station_stream = signal.sosfilt(filt, station_stream, axis=-1).astype(
                np.float32
            )
            station_stream = signal.detrend(station_stream)
            station_stream = signal.normalize(station_stream, 1)[0]
            sample = {"waveform": station_stream, "label": label}
            return sample


def get_data_loaders(
        catalog_path,
        hdf5_path,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        test_run=False,
):
    if test_run:
        num_workers = 1

    training_data = SeismoDataset(
        catalog_path=catalog_path,
        hdf5_path=hdf5_path,
        split="TRAIN",
        test_run=test_run,
    )
    validation_data = SeismoDataset(
        catalog_path=catalog_path,
        hdf5_path=hdf5_path,
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
        catalog_path, hdf5_path, batch_size=64, num_workers=4, test_run=False
):
    if test_run:
        num_workers = 1
    test_data = SeismoDataset(
        catalog_path=catalog_path,
        hdf5_path=hdf5_path,
        split="TEST",
        test_run=test_run,
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return test_loader
