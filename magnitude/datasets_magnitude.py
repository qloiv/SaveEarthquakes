from __future__ import print_function, division

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def obspy_detrend(data):
    # Convert data if it's not a floating point type.
    if not np.issubdtype(data.dtype, np.floating):
        data = np.require(data, dtype=np.float64)
    ndat = len(data)
    x1, x2 = data[0], data[-1]
    data -= x1 + np.arange(ndat) * (x2 - x1) / float(ndat - 1)
    return data


def normalize_stream(stream, global_max=False):
    stream_max = np.float32(np.abs(stream).max())
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


def resample_trace(trace, sampling_rate):
    if trace.stats.sampling_rate == sampling_rate:
        return
    if trace.stats.sampling_rate % sampling_rate == 0:
        trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
    else:
        trace.resample(sampling_rate)


class DetectionDataset(Dataset):
    def __init__(
            self,
            catalog_path,
            hdf5_path,
            split,
            time_before=10,
            time_after=10,
    ):
        self.split_key = str.lower(split) + "_files"
        self.file_path = hdf5_path
        self.h5data = None
        catalog = pd.read_csv(catalog_path)
        self.catalog = catalog[catalog["SPLIT"] == split]

        train = catalog[catalog["SPLIT"] == "TRAIN"]
        mag = np.array([0, 9])
        assert max(np.array(train["MA"])) <= 9
        assert min(np.array(train["MA"])) >= 0

        self.scaler = MinMaxScaler()
        self.scaler.fit(mag.reshape(-1, 1))

        self.sampling_rate = 100
        self.p_pick = 3000
        self.split = str.lower(split) + "_files"
        self.time_before = time_before * self.sampling_rate
        self.time_after = time_after * self.sampling_rate
        self.idx_changes = 0

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        if self.h5data is None:
            self.h5data = h5py.File(self.file_path, "r").get(self.split_key)

        # in all waveforms in the hdf5 catalogue, the pick was placed at index 3001
        event, station, magnitude = self.catalog.iloc[idx][["EVENT", "STATION", "MA"]]
        magnitude = self.scaler.transform(magnitude.reshape(1, -1))
        label = np.float32(magnitude.squeeze())

        waveform = np.array(self.h5data.get(event + "/" + station))
        seq_len = self.time_before + self.time_after  # is 2000 if 20sec Window
        random_point = np.random.randint(seq_len)
        station_stream = waveform[
                         :, self.p_pick - random_point: self.p_pick + (seq_len - random_point)
                         ]
        d0 = obspy_detrend(station_stream[0])
        d1 = obspy_detrend(station_stream[1])
        d2 = obspy_detrend(station_stream[2])

        # set high pass filter
        filt = signal.butter(
            2, 2, btype="highpass", fs=self.sampling_rate, output="sos"
        )
        f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
        f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
        f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)

        # set low pass filter
        lfilt = signal.butter(2, 35, btype="lowpass", fs=100, output="sos")
        g0 = signal.sosfilt(lfilt, f0, axis=-1).astype(np.float32)
        g1 = signal.sosfilt(lfilt, f1, axis=-1).astype(np.float32)
        g2 = signal.sosfilt(lfilt, f2, axis=-1).astype(np.float32)

        station_stream = np.stack((g0, g1, g2))
        station_stream, max_stream = normalize_stream(station_stream)
        sample = {"waveform": (station_stream, 0.01 * np.log(max_stream)), "label": label}
        return sample
