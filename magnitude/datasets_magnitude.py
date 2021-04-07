from __future__ import print_function, division

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from torch.utils.data import Dataset


def obspy_detrend_simple(data):
    # Convert data if it's not a floating point type.
    if not np.issubdtype(data.dtype, np.floating):
        data = np.require(data, dtype=np.float64)
    ndat = len(data)
    x1, x2 = data[0], data[-1]
    data -= x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1)
    return data


def normalize_stream(stream, global_max=False):
    if global_max is True:
        ma = np.abs(stream).max()
        stream /= ma
    else:
        for tr in stream:
            ma_tr = np.abs(tr).max()
            tr /= ma_tr
    return stream


def resample_trace(trace, sampling_rate):
    if trace.stats.sampling_rate == sampling_rate:
        return
    if trace.stats.sampling_rate % sampling_rate == 0:
        trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
    else:
        trace.resample(sampling_rate)


def normalize_stream(stream, global_max=False):
    if global_max is True:
        ma = np.abs(stream).max()
        stream /= ma
    else:
        for tr in stream:
            ma_tr = np.abs(tr).max()
            tr /= ma_tr
    return stream


class DetectionDataset(Dataset):
    def __init__(
            self,
            catalog_path,
            hdf5_path,
            split,
            time_before=10,
            time_after=10,
            test_run=False,
    ):
        self.split_key = str.lower(split) + "_files"
        self.file_path = hdf5_path
        self.h5data = None
        catalog = pd.read_csv(catalog_path)
        self.catalog = catalog[catalog["SPLIT"] == split]
        self.sampling_rate = 100
        self.p_pick = 3001
        self.split = str.lower(split) + "_files"
        self.time_before = time_before * self.sampling_rate
        self.time_after = time_after * self.sampling_rate
        self.idx_changes = 0

    def __len__(self):
        return (
            len(self.catalog)
        )  # because the noise examples are generated from the same data

    def __getitem__(self, idx):
        if self.h5data is None:
            self.h5data = h5py.File(self.file_path, "r").get(self.split_key)

        event, station, magnitude = self.catalog.iloc[idx][["EVENT", "STATION", "ML"]]
        label = np.int64(magnitude)
        # in all waveforms in the hdf5 catalogue, the pick was placed at index 1001
        waveform = np.array(self.h5data.get(event + "/" + station))
        seq_len = self.time_before + self.time_after
        random_point = np.random.randint(seq_len)
        station_stream = waveform[:, self.p_pick - random_point: self.p_pick + (seq_len - random_point)]
        d0 = obspy_detrend_simple(station_stream[0])
        d1 = obspy_detrend_simple(station_stream[1])
        d2 = obspy_detrend_simple(station_stream[2])
        filt = signal.butter(
            2, 2, btype="highpass", fs=self.sampling_rate, output="sos"
        )
        f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
        f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
        f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)
        station_stream = np.stack((f0, f1, f2))
        station_stream = normalize_stream(station_stream)
        sample = {"waveform": station_stream, "label": label}
        return sample
