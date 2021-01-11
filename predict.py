# load net
# load waveform
# cut waveform in little pieces, 0.5 seconds apart
# for each piece, let the network determine a label
# save the label and the start, and endtime, for printing
# start and endtimes, and labels should give a prediction function. can this be printing onto the original waveform?

# though for real time use, one wants to usher an alarm, when the label is above a threshold (for a certain time?)
# so maybe I should rather iterate through every piece, like I would get them in real time.
# This might be performance critical
import os
from random import randrange
import random
import numpy as np
import obspy
import pandas as pd
import torch
from network import Net
from obspy import UTCDateTime
from torch.utils.data import Dataset, DataLoader

# Load waveform from Test set

catalog_path = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
waveform_path = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
model_path = "/home/viola/WS2021/Code/Models"

sequence_length = 4

data = pd.read_csv(catalog_path)
#        print(len(data))
data = data[data['SPLIT'] == 'TEST']

data = data[data['STATION'] != 'LVC']
events = sorted(data['EVENT'].unique())
data = data[data['EVENT'].isin(events)]
#        print(len(data))
no_sample = True
idx = randrange(0, len(data))
event, station, p_pick, time = data.iloc[idx][
    ['EVENT', 'STATION', 'P_PICK', 'TIME']]  # TODO time means start time of event?
station_stream = []
while no_sample:
    idx = randrange(0, len(data))
    event, station, p_pick, time = data.iloc[idx][
        ['EVENT', 'STATION', 'P_PICK', 'TIME']]  # TODO time means start time of event?
    if os.path.isfile(os.path.join(waveform_path, f'{event}.mseed')):
        waveform = obspy.read(os.path.join(waveform_path, f'{event}.mseed'))
        station_stream = waveform.select(station=station, channel='HH*')  # high frequency
        if station_stream:
            no_sample = False

# to prevent loading the entire catalog, choose a random event and check, whether it is in the waveform folder

# print now maybe with highlight on the P Pick
# maybe slice out with beginning on start time?

# now start slicing with a given sequence length
# TODO maybe read that out of a setup file?
station_stream.plot()
model = Net()
model.load_state_dict(torch.load(os.path.join(model_path, 'GPD_net_2020-12-17 10:29.pth')))
model.eval()
outs = []
for windowed_st in station_stream.slide(window_length=sequence_length, step=1):
    window = windowed_st.copy()
    window.detrend().normalize()
    trace_z = np.float32(window[0])
    trace_n = np.float32(window[1])
    trace_e = np.float32(window[2])
    example = np.stack((trace_z, trace_n, trace_e))[:,
              0:-1]
    example = torch.from_numpy(example[None])
    output = model(example)
    _, predicted = torch.max(output, 1)
    outs.append(predicted)
#    print(windowed_st)

#    print("---")

print(outs)
