from __future__ import print_function, division

from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
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


class DistanceDataset(Dataset):
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
        train = catalog[catalog["SPLIT"] == "TRAIN"]
        dist = np.array([1, 600000])
        assert max(np.array(train["DIST"])) <= 600000
        assert min(np.array(train["DIST"])) >= 1

        # # compute best lambda on train set
        # l, opt_lambda = stats.boxcox(dist)
        # self.lb = opt_lambda
        # transformed = stats.boxcox(dist,opt_lambda)
        #
        # self.scaler = MinMaxScaler()
        # self.scaler.fit(transformed.reshape(-1,1))
        # t_dist = self.scaler.transform(transformed.reshape(-1,1))

        self.scaler = MinMaxScaler()
        self.scaler.fit(dist.reshape(-1, 1))
        # t_dist = self.scaler.transform(dist.reshape(-1, 1))

        self.catalog = catalog[catalog["SPLIT"] == split]
        self.sampling_rate = 100
        self.p_pick = 3000
        self.split = str.lower(split) + "_files"
        self.time_before = time_before * self.sampling_rate
        self.time_after = time_after * self.sampling_rate
        self.idx_changes = 0

        self.h5dict = defaultdict(dict)
        with h5py.File(self.file_path, "r") as h5file:
            h5file = h5file.get(self.split_key)
            events = self.catalog["EVENT"]
            stations = self.catalog["STATION"]
            hp5index = list(zip(events, stations))
            # write new dict using this tuple list
            for event, station in tqdm(hp5index):
                waveform = np.array(h5file.get(event + "/" + station))
                self.h5dict[event][station] = waveform

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #    idx = idx.tolist()
        while True:
            event, station, distance = self.catalog.iloc[idx][
                ["EVENT", "STATION", "DIST"]
            ]

            # s_dist = (distance ** self.lb - 1) / self.lb
            # assert s_dist != np.nan
            #
            # ts_dist =self.scaler.transform(s_dist.reshape(1,-1))
            # label = np.float32(ts_dist.squeeze())
            # assert 0 <= label <= 1, str(label)

            ts_dist = self.scaler.transform(distance.reshape(1, -1))
            label = np.float32(ts_dist.squeeze())

            # in all waveforms in the hdf5 catalogue, the pick was placed at index 3000
            waveform = self.h5dict[event][station]
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

            station_stream, _ = normalize_stream(station_stream)
            sample = {"waveform": station_stream, "label": label}
            return sample
