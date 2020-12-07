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
time_before = 2
time_after = 2
train_list = []
test_list = []
data = data[data['EVENT'].isin(events)]
for event_id, (event, event_data) in enumerate(tqdm(data.groupby('EVENT'))):
    print(event, event_data)
    waveform = obspy.read(os.path.join(waveform_path, f'{event}.mseed'))
    for (station, p_pick, split) in event_data[['STATION', 'P_PICK', 'SPLIT']].values:
        print(station, p_pick, split)
        start_time = UTCDateTime(p_pick - time_before)
        end_time = UTCDateTime(p_pick + time_after)
        station_stream = waveform.select(station=station, channel='HH*')
        station_stream = station_stream.slice(starttime=start_time, endtime=end_time)
        trace_z = station_stream[0]
        trace_n = station_stream[1]
        trace_e = station_stream[2]
        example = np.stack((trace_z, trace_n, trace_e))[:,
                  0:-1]  # because I want a length of 400, not 401, I leave out the last element
        if split == 'TRAIN':
            train_list.append(example)
        elif split == 'TEST':
            test_list.append(example)
        else:
            print('Error or valid?', split)

train_array = np.array(train_list)
print(train_array.shape)
test_array = np.array(test_list)
print(test_array.shape)
