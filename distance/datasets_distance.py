from __future__ import print_function, division

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def obspy_detrend(data):
    # based on obspys detrend ("simple") function
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
        i = 0
        for tr in stream:
            ma_tr = np.abs(tr).max()
            if (ma_tr == 0):
                i += 1
            else:
                tr /= ma_tr
        if i == 3:
            print("Der gesamte Stream ist 0")
    return stream, True


class DistanceDataset(Dataset):
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
        t_dist = self.scaler.transform(dist.reshape(-1, 1))

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
        )

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #    idx = idx.tolist()
        if self.h5data is None:
            self.h5data = h5py.File(self.file_path, "r").get(self.split_key)
        while True:
            event, station, distance = self.catalog.iloc[idx][["EVENT", "STATION", 'DIST']]

            # s_dist = (distance ** self.lb - 1) / self.lb
            # assert s_dist != np.nan
            #
            # ts_dist =self.scaler.transform(s_dist.reshape(1,-1))
            # label = np.float32(ts_dist.squeeze())
            # assert 0 <= label <= 1, str(label)

            ts_dist = self.scaler.transform(distance.reshape(1, -1))
            label = np.float32(ts_dist.squeeze())

            # in all waveforms in the hdf5 catalogue, the pick was placed at index 3001
            waveform = np.array(self.h5data.get(event + "/" + station))
            seq_len = self.time_before + self.time_after  # is 2000 if 20sec Window
            random_point = np.random.randint(seq_len)
            station_stream = waveform[:, self.p_pick - random_point: self.p_pick + (seq_len - random_point)]
            d0 = obspy_detrend(station_stream[0])
            d1 = obspy_detrend(station_stream[1])
            d2 = obspy_detrend(station_stream[2])
            filt = signal.butter(
                2, 2, btype="highpass", fs=self.sampling_rate, output="sos"
            )
            f0 = signal.sosfilt(filt, d0, axis=-1).astype(np.float32)
            f1 = signal.sosfilt(filt, d1, axis=-1).astype(np.float32)
            f2 = signal.sosfilt(filt, d2, axis=-1).astype(np.float32)
            station_stream = np.stack((f0, f1, f2))
            station_stream, bl = normalize_stream(station_stream)
            if bl is False:
                continue
            sample = {"waveform": station_stream, "label": label}
            return sample

#
# def get_data_loaders(
#         catalog_path,
#         hdf5_path,
#         batch_size=64,
#         num_workers=4,
#         shuffle=True,
#         test_run=False,
# ):
#     if test_run:
#         num_workers = 1
#
#     training_data = DistanceDataset(
#         catalog_path=catalog_path,
#         hdf5_path=hdf5_path,
#         split="TRAIN",
#         test_run=test_run,
#     )
#     validation_data = DistanceDataset(
#         catalog_path=catalog_path,
#         hdf5_path=hdf5_path,
#         split="DEV",
#         test_run=test_run,
#     )
#
#     training_loader = DataLoader(
#         training_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
#     )
#     validation_loader = DataLoader(
#         validation_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
#     )
#
#     return training_loader, validation_loader
#
#
# def get_test_loader(
#         catalog_path, hdf5_path, batch_size=64, num_workers=4, test_run=False
# ):
#     if test_run:
#         num_workers = 1
#     test_data = DistanceDataset(
#         catalog_path=catalog_path,
#         hdf5_path=hdf5_path,
#         split="TEST",
#         test_run=test_run,
#     )
#
#     test_loader = DataLoader(
#         test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
#     )
#
#     return test_loader
