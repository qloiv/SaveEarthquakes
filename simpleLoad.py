import pandas as pd
from tqdm import tqdm
import obspy
import os
import numpy as np
from obspy import UTCDateTime

catalog_path = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
numpy_array_path = "/home/viola/WS2021/Code/Daten/Chile_small/numpyarrays"
data = pd.read_csv(catalog_path)
data = data[data['STATION'] != 'LVC']  # Faulty station

waveform_path = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
events = sorted(data['EVENT'].unique())
data = data[data['EVENT'].isin(events)]
time_before = 2
time_after = 2
train_list = []
test_list = []
dev_list = []
dev_noise = []
test_noise = []
train_noise = []

for event_id, (event, event_data) in enumerate(tqdm(data.groupby('EVENT'))):
    # print(event, event_data)
    if os.path.isfile(os.path.join(waveform_path, f'{event}.mseed')):
        waveform = obspy.read(os.path.join(waveform_path, f'{event}.mseed'))
        for (station, p_pick, split, time) in event_data[['STATION', 'P_PICK', 'SPLIT', 'TIME']].values:
            # print(station, p_pick, split)
            start_time = UTCDateTime(p_pick - time_before)
            end_time = UTCDateTime(p_pick + time_after)
            station_stream = waveform.select(station=station, channel='HH*')
            noise_stream = station_stream.slice(starttime=UTCDateTime(time), endtime=UTCDateTime(time + 4))
            station_stream = station_stream.slice(starttime=start_time, endtime=end_time)
            if station_stream:
                trace_z = station_stream[0]
                trace_n = station_stream[1]
                trace_e = station_stream[2]
                example = np.stack((trace_z, trace_n, trace_e))[:,
                          0:-1]  # because I want a length of 400, not 401, I leave out the last element
                noise_z = noise_stream[0]
                noise_n = noise_stream[1]
                noise_e = noise_stream[2]
                noise_example = np.stack((noise_z, noise_n, noise_e))[:,
                                0:-1]
                if split == 'TRAIN':
                    train_list.append(example)
                    train_noise.append(noise_example)
                elif split == 'TEST':
                    test_list.append(example)
                    test_noise.append(noise_example)

                else:
                    dev_list.append(example)
                    dev_noise.append(noise_example)

train_array = np.array(train_list, dtype=np.float32)

print(train_array.shape)
test_array = np.array(test_list, dtype=np.float32)
print(test_array.shape)
dev_array = np.array(dev_list, dtype=np.float32)
print(dev_array.shape)
np.save('train_array.npy', train_array)
np.save('test_array.npy', test_array)
np.save('dev_array.npy', dev_array)
np.save('train_noise.npy', np.array(train_noise, dtype=np.float32))
np.save('test_noise.npy', np.array(test_noise, dtype=np.float32))
np.save('dev_noise.npy', np.array(dev_noise, dtype=np.float32))
