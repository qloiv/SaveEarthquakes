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
import matplotlib.pyplot as plt
import random
import numpy as np
import obspy
import pandas as pd
import torch
from matplotlib.dates import date2num
from network import Net
from obspy import UTCDateTime
from torch.utils.data import Dataset, DataLoader

# Load waveform from Test set

catalog_path = "/home/viola/WS2021/Code/Daten/Chile_small/catalog_ma.csv"
waveform_path = "/home/viola/WS2021/Code/Daten/Chile_small/mseedJan07/"
model_path = "/home/viola/WS2021/Code/Models"

sequence_length = 4
data = pd.read_csv(catalog_path)
print(len(data))
data = data[data['STATION'] != 'LVC']
events = sorted(data['EVENT'].unique())
waves = []
for file in os.listdir(waveform_path):
    if file.endswith(".mseed"):
        waves.append(file.strip('.mseed'))
intersect = list(set(waves).intersection(set(events)))
data = data[data['EVENT'].isin(intersect)]

idx = randrange(0, len(data))
event, station, p_pick, time = data.iloc[idx][
    ['EVENT', 'STATION', 'P_PICK', 'TIME']]  # TODO time means start time of event?
waveform = obspy.read(os.path.join(waveform_path, f'{event}.mseed'))
station_stream = waveform.select(station=station, channel='HH*')  # high frequency
while not station_stream:
    idx = randrange(0, len(data))
    event, station, p_pick, time = data.iloc[idx][
        ['EVENT', 'STATION', 'P_PICK', 'TIME']]  # TODO time means start time of event?
    waveform = obspy.read(os.path.join(waveform_path, f'{event}.mseed'))
    station_stream = waveform.select(station=station, channel='HH*')  # high frequency

station_stream.plot()
# to prevent loading the entire catalog, choose a random event and check, whether it is in the waveform folder

# print now maybe with highlight on the P Pick
# maybe slice out with beginning on start time?

# now start slicing with a given sequence length
# TODO maybe read that out of a setup file?
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

device = torch.device('cpu')
model = Net()
model.load_state_dict(torch.load(os.path.join(model_path, 'GPD_net_2021-01-11 15:42.pth'), map_location=device))
model.eval()

out_times = (np.arange(station_stream[0].stats.starttime, station_stream[0].stats.endtime + 1, 1))
outs = np.zeros(len(out_times))
for i in range(0, len(out_times) - 4):
    station_stream = waveform.select(station=station, channel='HH*')  # reload
    window = station_stream.slice(starttime=out_times[i], endtime=out_times[i] + 4)
    window.detrend().normalize()

    # for windowed_st in station_stream.slide(window_length=sequence_length, step=1):
    #     window = windowed_st.copy()
    #     window.detrend().normalize()
    trace_z = np.float32(window[0])
    trace_n = np.float32(window[1])
    trace_e = np.float32(window[2])
    example = np.stack((trace_z, trace_n, trace_e))[:,
              0:-1]
    example = torch.from_numpy(example[None])
    output = model(example)
    _, predicted = torch.max(output, 1)

    outs[i:i + 4] = outs[i:i + 4] + np.full(4, predicted[0])
    # print(outs)
#    print(windowed_st)

#    print("---")

print(outs)
out_times = date2num(out_times)

tr1 = station_stream[0]
tr2 = station_stream[1]
tr3 = station_stream[2]

fig, ax = plt.subplots(4, sharex=True)
ax[0].plot(tr1.times("matplotlib"), tr1.data, color='black', linewidth=0.4)
ax[0].xaxis_date()
ax[0].axvline(date2num(UTCDateTime(p_pick)), color='blue')

ax[1].plot(tr2.times("matplotlib"), tr2.data, color='black', linewidth=0.4)
ax[1].xaxis_date()
ax[1].axvline(date2num(UTCDateTime(p_pick)), color='blue')

ax[2].plot(tr3.times("matplotlib"), tr3.data, color='black', linewidth=0.4)
ax[2].xaxis_date()
ax[2].axvline(date2num(UTCDateTime(p_pick)), color='blue')

ax[3].plot(out_times, outs, 'r-')
ax[3].xaxis_date()
fig.autofmt_xdate()
plt.show()
